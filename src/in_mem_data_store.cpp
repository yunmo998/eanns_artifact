// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <memory>
#include <cstdlib>
#include <cstring>
#include "abstract_scratch.h"
#include "in_mem_data_store.h"

#include "utils.h"

#ifndef _WINDOWS
#include <numa.h>
#include <sched.h>
#endif

namespace diskann
{

#ifndef _WINDOWS
static inline bool disable_numa_replica_runtime()
{
    const char *value = std::getenv("DISKANN_ENABLE_NUMA_REPLICA");
    if (value == nullptr)
    {
        return true;
    }

    return !(std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0);
}

#define DISKANN_DISABLE_NUMA_REPLICA disable_numa_replica_runtime()

static inline int get_current_numa_node()
{
    if (numa_available() == -1)
    {
        return -1;
    }

    const int cpu = sched_getcpu();
    if (cpu < 0)
    {
        return -1;
    }

    return numa_node_of_cpu(cpu);
}
#endif

template <typename data_t>
InMemDataStore<data_t>::InMemDataStore(const location_t num_points, const size_t dim,
                                       std::unique_ptr<Distance<data_t>> distance_fn)
    : AbstractDataStore<data_t>(num_points, dim), _distance_fn(std::move(distance_fn))
{
    _aligned_dim = ROUND_UP(dim, _distance_fn->get_required_alignment());
#ifndef _WINDOWS
    _use_numa_replica = (!DISKANN_DISABLE_NUMA_REPLICA) && (numa_available() != -1);
    if (_use_numa_replica)
    {
        const size_t bytes = this->_capacity * _aligned_dim * sizeof(data_t);
        _data = static_cast<data_t *>(numa_alloc_onnode(bytes, _numa_node_primary));
        _data_replica = static_cast<data_t *>(numa_alloc_onnode(bytes, _numa_node_secondary));
        if (_data == nullptr || _data_replica == nullptr)
        {
            throw diskann::ANNException("Failed to allocate NUMA replicas for data store", -1, __FUNCSIG__, __FILE__,
                                        __LINE__);
        }
        std::memset(_data, 0, bytes);
        std::memset(_data_replica, 0, bytes);
    }
    else
    {
        alloc_aligned(((void **)&_data), this->_capacity * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
        std::memset(_data, 0, this->_capacity * _aligned_dim * sizeof(data_t));
    }
#else
    alloc_aligned(((void **)&_data), this->_capacity * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
    std::memset(_data, 0, this->_capacity * _aligned_dim * sizeof(data_t));
#endif
}

template <typename data_t> InMemDataStore<data_t>::~InMemDataStore()
{
#ifndef _WINDOWS
    if (_use_numa_replica)
    {
        const size_t bytes = this->_capacity * _aligned_dim * sizeof(data_t);
        if (_data != nullptr)
        {
            numa_free(_data, bytes);
        }
        if (_data_replica != nullptr)
        {
            numa_free(_data_replica, bytes);
        }
        return;
    }
#endif

    if (_data != nullptr)
    {
        aligned_free(this->_data);
    }
}

template <typename data_t> size_t InMemDataStore<data_t>::get_aligned_dim() const
{
    return _aligned_dim;
}

template <typename data_t> size_t InMemDataStore<data_t>::get_alignment_factor() const
{
    return _distance_fn->get_required_alignment();
}

template <typename data_t> location_t InMemDataStore<data_t>::load(const std::string &filename)
{
    return load_impl(filename);
}

#ifdef EXEC_ENV_OLS
template <typename data_t> location_t InMemDataStore<data_t>::load_impl(AlignedFileReader &reader)
{
    size_t file_dim, file_num_points;

    diskann::get_bin_metadata(reader, file_num_points, file_dim);

    if (file_dim != this->_dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << this->_dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > this->capacity())
    {
        this->resize((location_t)file_num_points);
    }
    copy_aligned_data_from_file<data_t>(reader, _data, file_num_points, file_dim, _aligned_dim);

    return (location_t)file_num_points;
}
#endif

template <typename data_t> location_t InMemDataStore<data_t>::load_impl(const std::string &filename)
{
    size_t file_dim, file_num_points;
    if (!file_exists(filename))
    {
        std::stringstream stream;
        stream << "ERROR: data file " << filename << " does not exist." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    diskann::get_bin_metadata(filename, file_num_points, file_dim);

    if (file_dim != this->_dim)
    {
        std::stringstream stream;
        stream << "ERROR: Driver requests loading " << this->_dim << " dimension,"
               << "but file has " << file_dim << " dimension." << std::endl;
        diskann::cerr << stream.str() << std::endl;
        aligned_free(_data);
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    if (file_num_points > this->capacity())
    {
        this->resize((location_t)file_num_points);
    }

    copy_aligned_data_from_file<data_t>(filename.c_str(), _data, file_num_points, file_dim, _aligned_dim);
    if (!DISKANN_DISABLE_NUMA_REPLICA && _data_replica != nullptr)
    {
        std::memcpy(_data_replica, _data, file_num_points * _aligned_dim * sizeof(data_t));
    }

    return (location_t)file_num_points;
}

template <typename data_t> size_t InMemDataStore<data_t>::save(const std::string &filename, const location_t num_points)
{
    return save_data_in_base_dimensions(filename, _data, num_points, this->get_dims(), this->get_aligned_dim(), 0U);
}

template <typename data_t> void InMemDataStore<data_t>::populate_data(const data_t *vectors, const location_t num_pts)
{
    memset(_data, 0, _aligned_dim * sizeof(data_t) * num_pts);
    for (location_t i = 0; i < num_pts; i++)
    {
        std::memmove(_data + i * _aligned_dim, vectors + i * this->_dim, this->_dim * sizeof(data_t));
    }

    if (_distance_fn->preprocessing_required())
    {
        _distance_fn->preprocess_base_points(_data, this->_aligned_dim, num_pts);
    }

    if (!DISKANN_DISABLE_NUMA_REPLICA && _data_replica != nullptr)
    {
        std::memcpy(_data_replica, _data, num_pts * _aligned_dim * sizeof(data_t));
    }
}

template <typename data_t> void InMemDataStore<data_t>::populate_data(const std::string &filename, const size_t offset)
{
    size_t npts, ndim;
    copy_aligned_data_from_file(filename.c_str(), _data, npts, ndim, _aligned_dim, offset);

    if ((location_t)npts > this->capacity())
    {
        std::stringstream ss;
        ss << "Number of points in the file: " << filename
           << " is greater than the capacity of data store: " << this->capacity()
           << ". Must invoke resize before calling populate_data()" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }

    if ((location_t)ndim != this->get_dims())
    {
        std::stringstream ss;
        ss << "Number of dimensions of a point in the file: " << filename
           << " is not equal to dimensions of data store: " << this->capacity() << "." << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }

    if (_distance_fn->preprocessing_required())
    {
        _distance_fn->preprocess_base_points(_data, this->_aligned_dim, this->capacity());
    }

    if (!DISKANN_DISABLE_NUMA_REPLICA && _data_replica != nullptr)
    {
        std::memcpy(_data_replica, _data, this->capacity() * _aligned_dim * sizeof(data_t));
    }
}

template <typename data_t>
void InMemDataStore<data_t>::extract_data_to_bin(const std::string &filename, const location_t num_points)
{
    save_data_in_base_dimensions(filename, _data, num_points, this->get_dims(), this->get_aligned_dim(), 0U);
}

template <typename data_t> void InMemDataStore<data_t>::get_vector(const location_t i, data_t *dest) const
{
    // REFACTOR TODO: Should we denormalize and return values?
    const data_t *source = _data;
#ifndef _WINDOWS
    if (!DISKANN_DISABLE_NUMA_REPLICA && _data_replica != nullptr)
    {
        const int node = get_current_numa_node();
        if (node == _numa_node_secondary)
        {
            source = _data_replica;
        }
    }
#endif
    memcpy(dest, source + i * _aligned_dim, this->_dim * sizeof(data_t));
}

template <typename data_t> void InMemDataStore<data_t>::set_vector(const location_t loc, const data_t *const vector)
{
    size_t offset_in_data = loc * _aligned_dim;
    memset(_data + offset_in_data, 0, _aligned_dim * sizeof(data_t));
    memcpy(_data + offset_in_data, vector, this->_dim * sizeof(data_t));
    if (_distance_fn->preprocessing_required())
    {
        _distance_fn->preprocess_base_points(_data + offset_in_data, _aligned_dim, 1);
    }

    if (!DISKANN_DISABLE_NUMA_REPLICA && _data_replica != nullptr)
    {
        memset(_data_replica + offset_in_data, 0, _aligned_dim * sizeof(data_t));
        memcpy(_data_replica + offset_in_data, _data + offset_in_data, _aligned_dim * sizeof(data_t));
    }
}

template <typename data_t> void InMemDataStore<data_t>::prefetch_vector(const location_t loc)
{
    const data_t *source = _data;
#ifndef _WINDOWS
    if (!DISKANN_DISABLE_NUMA_REPLICA && _data_replica != nullptr)
    {
        const int node = get_current_numa_node();
        if (node == _numa_node_secondary)
        {
            source = _data_replica;
        }
    }
#endif
    diskann::prefetch_vector((const char *)source + _aligned_dim * (size_t)loc * sizeof(data_t),
                             sizeof(data_t) * _aligned_dim);
}

template <typename data_t>
void InMemDataStore<data_t>::preprocess_query(const data_t *query, AbstractScratch<data_t> *query_scratch) const
{
    if (query_scratch != nullptr)
    {
        memcpy(query_scratch->aligned_query_T(), query, sizeof(data_t) * this->get_dims());
    }
    else
    {
        std::stringstream ss;
        ss << "In InMemDataStore::preprocess_query: Query scratch is null";
        diskann::cerr << ss.str() << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
}

template <typename data_t> float InMemDataStore<data_t>::get_distance(const data_t *query, const location_t loc) const
{
    const data_t *source = _data;
#ifndef _WINDOWS
    if (!DISKANN_DISABLE_NUMA_REPLICA && _data_replica != nullptr)
    {
        const int node = get_current_numa_node();
        if (node == _numa_node_secondary)
        {
            source = _data_replica;
        }
    }
#endif
    return _distance_fn->compare(query, source + _aligned_dim * loc, (uint32_t)_aligned_dim);
}

template <typename data_t>
void InMemDataStore<data_t>::get_distance(const data_t *query, const location_t *locations,
                                          const uint32_t location_count, float *distances,
                                          AbstractScratch<data_t> *scratch_space) const
{
    const data_t *source = _data;
#ifndef _WINDOWS
    if (!DISKANN_DISABLE_NUMA_REPLICA && _data_replica != nullptr)
    {
        const int node = get_current_numa_node();
        if (node == _numa_node_secondary)
        {
            source = _data_replica;
        }
    }
#endif
    for (location_t i = 0; i < location_count; i++)
    {
        distances[i] =
            _distance_fn->compare(query, source + locations[i] * _aligned_dim, (uint32_t)this->_aligned_dim);
    }
}

template <typename data_t>
float InMemDataStore<data_t>::get_distance(const location_t loc1, const location_t loc2) const
{
    const data_t *source = _data;
#ifndef _WINDOWS
    if (!DISKANN_DISABLE_NUMA_REPLICA && _data_replica != nullptr)
    {
        const int node = get_current_numa_node();
        if (node == _numa_node_secondary)
        {
            source = _data_replica;
        }
    }
#endif
    return _distance_fn->compare(source + loc1 * _aligned_dim, source + loc2 * _aligned_dim,
                                 (uint32_t)this->_aligned_dim);
}

template <typename data_t>
void InMemDataStore<data_t>::get_distance(const data_t *preprocessed_query, const std::vector<location_t> &ids,
                                          std::vector<float> &distances, AbstractScratch<data_t> *scratch_space) const
{
    const data_t *source = _data;
#ifndef _WINDOWS
    if (_data_replica != nullptr)
    {
        const int node = get_current_numa_node();
        if (node == _numa_node_secondary)
        {
            source = _data_replica;
        }
    }
#endif
    for (int i = 0; i < ids.size(); i++)
    {
        distances[i] = _distance_fn->compare(preprocessed_query, source + ids[i] * _aligned_dim,
                                              (uint32_t)this->_aligned_dim);
    }
}

template <typename data_t> location_t InMemDataStore<data_t>::expand(const location_t new_size)
{
    if (new_size == this->capacity())
    {
        return this->capacity();
    }
    else if (new_size < this->capacity())
    {
        std::stringstream ss;
        ss << "Cannot 'expand' datastore when new capacity (" << new_size << ") < existing capacity("
           << this->capacity() << ")" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
#ifndef _WINDOWS
    data_t *new_data;
    data_t *new_data_replica = nullptr;
    const size_t bytes = new_size * _aligned_dim * sizeof(data_t);
    if (_use_numa_replica)
    {
        new_data = static_cast<data_t *>(numa_alloc_onnode(bytes, _numa_node_primary));
        new_data_replica = static_cast<data_t *>(numa_alloc_onnode(bytes, _numa_node_secondary));
    }
    else
    {
        alloc_aligned((void **)&new_data, bytes, 8 * sizeof(data_t));
    }

    memcpy(new_data, _data, this->capacity() * _aligned_dim * sizeof(data_t));
    if (new_data_replica != nullptr)
    {
        memcpy(new_data_replica, _data_replica, this->capacity() * _aligned_dim * sizeof(data_t));
    }

    if (_use_numa_replica)
    {
        const size_t old_bytes = this->capacity() * _aligned_dim * sizeof(data_t);
        numa_free(_data, old_bytes);
        numa_free(_data_replica, old_bytes);
    }
    else
    {
        aligned_free(_data);
    }

    _data = new_data;
    _data_replica = new_data_replica;
#else
    realloc_aligned((void **)&_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
#endif
    this->_capacity = new_size;
    return this->_capacity;
}

template <typename data_t> location_t InMemDataStore<data_t>::shrink(const location_t new_size)
{
    if (new_size == this->capacity())
    {
        return this->capacity();
    }
    else if (new_size > this->capacity())
    {
        std::stringstream ss;
        ss << "Cannot 'shrink' datastore when new capacity (" << new_size << ") > existing capacity("
           << this->capacity() << ")" << std::endl;
        throw diskann::ANNException(ss.str(), -1);
    }
#ifndef _WINDOWS
    data_t *new_data;
    data_t *new_data_replica = nullptr;
    const size_t bytes = new_size * _aligned_dim * sizeof(data_t);
    if (_use_numa_replica)
    {
        new_data = static_cast<data_t *>(numa_alloc_onnode(bytes, _numa_node_primary));
        new_data_replica = static_cast<data_t *>(numa_alloc_onnode(bytes, _numa_node_secondary));
    }
    else
    {
        alloc_aligned((void **)&new_data, bytes, 8 * sizeof(data_t));
    }

    memcpy(new_data, _data, new_size * _aligned_dim * sizeof(data_t));
    if (new_data_replica != nullptr)
    {
        memcpy(new_data_replica, _data_replica, new_size * _aligned_dim * sizeof(data_t));
    }

    if (_use_numa_replica)
    {
        const size_t old_bytes = this->capacity() * _aligned_dim * sizeof(data_t);
        numa_free(_data, old_bytes);
        numa_free(_data_replica, old_bytes);
    }
    else
    {
        aligned_free(_data);
    }

    _data = new_data;
    _data_replica = new_data_replica;
#else
    realloc_aligned((void **)&_data, new_size * _aligned_dim * sizeof(data_t), 8 * sizeof(data_t));
#endif
    this->_capacity = new_size;
    return this->_capacity;
}

template <typename data_t>
void InMemDataStore<data_t>::move_vectors(const location_t old_location_start, const location_t new_location_start,
                                          const location_t num_locations)
{
    if (num_locations == 0 || old_location_start == new_location_start)
    {
        return;
    }

    /*    // Update pointers to the moved nodes. Note: the computation is correct
       even
        // when new_location_start < old_location_start given the C++ uint32_t
        // integer arithmetic rules.
        const uint32_t location_delta = new_location_start - old_location_start;
    */
    // The [start, end) interval which will contain obsolete points to be
    // cleared.
    uint32_t mem_clear_loc_start = old_location_start;
    uint32_t mem_clear_loc_end_limit = old_location_start + num_locations;

    if (new_location_start < old_location_start)
    {
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_start < new_location_start + num_locations)
        {
            // Clear only after the end of the new range.
            mem_clear_loc_start = new_location_start + num_locations;
        }
    }
    else
    {
        // If ranges are overlapping, make sure not to clear the newly copied
        // data.
        if (mem_clear_loc_end_limit > new_location_start)
        {
            // Clear only up to the beginning of the new range.
            mem_clear_loc_end_limit = new_location_start;
        }
    }

    // Use memmove to handle overlapping ranges.
    copy_vectors(old_location_start, new_location_start, num_locations);
    memset(_data + _aligned_dim * mem_clear_loc_start, 0,
           sizeof(data_t) * _aligned_dim * (mem_clear_loc_end_limit - mem_clear_loc_start));
    if (!DISKANN_DISABLE_NUMA_REPLICA && _data_replica != nullptr)
    {
        memset(_data_replica + _aligned_dim * mem_clear_loc_start, 0,
               sizeof(data_t) * _aligned_dim * (mem_clear_loc_end_limit - mem_clear_loc_start));
    }
}

template <typename data_t>
void InMemDataStore<data_t>::copy_vectors(const location_t from_loc, const location_t to_loc,
                                          const location_t num_points)
{
    assert(from_loc < this->_capacity);
    assert(to_loc < this->_capacity);
    assert(num_points < this->_capacity);
    memmove(_data + _aligned_dim * to_loc, _data + _aligned_dim * from_loc, num_points * _aligned_dim * sizeof(data_t));
    if (!DISKANN_DISABLE_NUMA_REPLICA && _data_replica != nullptr)
    {
        memmove(_data_replica + _aligned_dim * to_loc, _data_replica + _aligned_dim * from_loc,
                num_points * _aligned_dim * sizeof(data_t));
    }
}

template <typename data_t> location_t InMemDataStore<data_t>::calculate_medoid() const
{
    // allocate and init centroid
    float *center = new float[_aligned_dim];
    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] = 0;

    for (size_t i = 0; i < this->capacity(); i++)
        for (size_t j = 0; j < _aligned_dim; j++)
            center[j] += (float)_data[i * _aligned_dim + j];

    for (size_t j = 0; j < _aligned_dim; j++)
        center[j] /= (float)this->capacity();

    // compute all to one distance
    float *distances = new float[this->capacity()];

    // TODO: REFACTOR. Removing pragma might make this slow. Must revisit.
    //  Problem is that we need to pass num_threads here, it is not clear
    //  if data store must be aware of threads!
    // #pragma omp parallel for schedule(static, 65536)
    for (int64_t i = 0; i < (int64_t)this->capacity(); i++)
    {
        // extract point and distance reference
        float &dist = distances[i];
        const data_t *cur_vec = _data + (i * (size_t)_aligned_dim);
        dist = 0;
        float diff = 0;
        for (size_t j = 0; j < _aligned_dim; j++)
        {
            diff = (center[j] - (float)cur_vec[j]) * (center[j] - (float)cur_vec[j]);
            dist += diff;
        }
    }
    // find imin
    uint32_t min_idx = 0;
    float min_dist = distances[0];
    for (uint32_t i = 1; i < this->capacity(); i++)
    {
        if (distances[i] < min_dist)
        {
            min_idx = i;
            min_dist = distances[i];
        }
    }

    delete[] distances;
    delete[] center;
    return min_idx;
}

template <typename data_t> Distance<data_t> *InMemDataStore<data_t>::get_dist_fn() const
{
    return this->_distance_fn.get();
}

template DISKANN_DLLEXPORT class InMemDataStore<float>;
template DISKANN_DLLEXPORT class InMemDataStore<int8_t>;
template DISKANN_DLLEXPORT class InMemDataStore<uint8_t>;

} // namespace diskann