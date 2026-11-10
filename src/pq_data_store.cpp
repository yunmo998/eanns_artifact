#include <exception>
#include <cstdlib>
#include <cstring>

#include "pq_data_store.h"
#include "pq.h"
#include "pq_scratch.h"
#include "utils.h"
#include "distance.h"

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

// REFACTOR TODO: Assuming that num_pq_chunks is known already. Must verify if
// this is true.
template <typename data_t>
PQDataStore<data_t>::PQDataStore(size_t dim, location_t num_points, size_t num_pq_chunks,
                                 std::unique_ptr<Distance<data_t>> distance_fn,
                                 std::unique_ptr<QuantizedDistance<data_t>> pq_distance_fn)
    : AbstractDataStore<data_t>(num_points, dim), _quantized_data(nullptr), _num_chunks(num_pq_chunks),
      _distance_metric(distance_fn->get_metric())
{
    if (num_pq_chunks > dim)
    {
        throw diskann::ANNException("ERROR: num_pq_chunks > dim", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    _distance_fn = std::move(distance_fn);
    _pq_distance_fn = std::move(pq_distance_fn);

#ifndef _WINDOWS
    _use_numa_replica = (!DISKANN_DISABLE_NUMA_REPLICA) && (numa_available() != -1);
#endif
}

template <typename data_t> PQDataStore<data_t>::~PQDataStore()
{
    if (_quantized_data != nullptr)
    {
#ifndef _WINDOWS
        if (_use_numa_replica)
        {
            const size_t bytes = this->capacity() * _num_chunks * sizeof(uint8_t);
            numa_free(_quantized_data, bytes);
        }
        else
#endif
        {
            aligned_free(_quantized_data);
        }
        _quantized_data = nullptr;
    }

    if (_quantized_data_replica != nullptr)
    {
#ifndef _WINDOWS
        const size_t bytes = this->capacity() * _num_chunks * sizeof(uint8_t);
        numa_free(_quantized_data_replica, bytes);
#endif
        _quantized_data_replica = nullptr;
    }
}

template <typename data_t> location_t PQDataStore<data_t>::load(const std::string &filename)
{
    return load_impl(filename);
}
template <typename data_t> size_t PQDataStore<data_t>::save(const std::string &filename, const location_t num_points)
{
    return diskann::save_bin(filename, _quantized_data, this->capacity(), _num_chunks, 0);
}

template <typename data_t> size_t PQDataStore<data_t>::get_aligned_dim() const
{
    return this->get_dims();
}

// Populate quantized data from regular data.
template <typename data_t> void PQDataStore<data_t>::populate_data(const data_t *vectors, const location_t num_pts)
{
    throw std::logic_error("Not implemented yet");
}

template <typename data_t> void PQDataStore<data_t>::populate_data(const std::string &filename, const size_t offset)
{
    if (_quantized_data != nullptr)
    {
#ifndef _WINDOWS
        if (_use_numa_replica)
        {
            const size_t bytes = this->capacity() * _num_chunks * sizeof(uint8_t);
            numa_free(_quantized_data, bytes);
        }
        else
#endif
        {
            aligned_free(_quantized_data);
        }
    }
    if (_quantized_data_replica != nullptr)
    {
#ifndef _WINDOWS
        const size_t bytes = this->capacity() * _num_chunks * sizeof(uint8_t);
        numa_free(_quantized_data_replica, bytes);
#endif
        _quantized_data_replica = nullptr;
    }

    uint64_t file_num_points = 0, file_dim = 0;
    get_bin_metadata(filename, file_num_points, file_dim, offset);
    this->_capacity = (location_t)file_num_points;
    this->_dim = file_dim;

    double p_val = std::min(1.0, ((double)MAX_PQ_TRAINING_SET_SIZE / (double)file_num_points));

    auto pivots_file = _pq_distance_fn->get_pivot_data_filename(filename);
    auto compressed_file = _pq_distance_fn->get_quantized_vectors_filename(filename);

    generate_quantized_data<data_t>(filename, pivots_file, compressed_file, _distance_metric, p_val, _num_chunks,
                                    _pq_distance_fn->is_opq());

    // REFACTOR TODO: Not sure of the alignment. Just copying from index.cpp
    const size_t bytes = file_num_points * _num_chunks * sizeof(uint8_t);
#ifndef _WINDOWS
    if (_use_numa_replica)
    {
        _quantized_data = static_cast<uint8_t *>(numa_alloc_onnode(bytes, _numa_node_primary));
        _quantized_data_replica = static_cast<uint8_t *>(numa_alloc_onnode(bytes, _numa_node_secondary));
    }
    else
#endif
    {
        alloc_aligned(((void **)&_quantized_data), bytes, 1);
    }
    copy_aligned_data_from_file<uint8_t>(compressed_file.c_str(), _quantized_data, file_num_points, _num_chunks,
                                         _num_chunks);
    if (!DISKANN_DISABLE_NUMA_REPLICA && _quantized_data_replica != nullptr)
    {
        std::memcpy(_quantized_data_replica, _quantized_data, bytes);
    }
#ifdef EXEC_ENV_OLS
    throw ANNException("load_pq_centroid_bin should not be called when "
                       "EXEC_ENV_OLS is defined.",
                       -1, __FUNCSIG__, __FILE__, __LINE__);
#else
    _pq_distance_fn->load_pivot_data(pivots_file.c_str(), _num_chunks);
#endif
}

template <typename data_t>
void PQDataStore<data_t>::extract_data_to_bin(const std::string &filename, const location_t num_pts)
{
    throw std::logic_error("Not implemented yet");
}

template <typename data_t> void PQDataStore<data_t>::get_vector(const location_t i, data_t *target) const
{
    // REFACTOR TODO: Should we inflate the compressed vector here?
    if (i < this->capacity())
    {
        throw std::logic_error("Not implemented yet.");
    }
    else
    {
        std::stringstream ss;
        ss << "Requested vector " << i << " but only  " << this->capacity() << " vectors are present";
        throw diskann::ANNException(ss.str(), -1);
    }
}
template <typename data_t> void PQDataStore<data_t>::set_vector(const location_t i, const data_t *const vector)
{
    // REFACTOR TODO: Should we accept a normal vector and compress here?
    // memcpy (_data + i * _num_chunks, vector, _num_chunks * sizeof(data_t));
    throw std::logic_error("Not implemented yet");
}

template <typename data_t> void PQDataStore<data_t>::prefetch_vector(const location_t loc)
{
    const uint8_t *ptr = _quantized_data;
#ifndef _WINDOWS
    if (!DISKANN_DISABLE_NUMA_REPLICA && _quantized_data_replica != nullptr)
    {
        const int node = get_current_numa_node();
        if (node == _numa_node_secondary)
        {
            ptr = _quantized_data_replica;
        }
    }
#endif
    ptr = ptr + ((size_t)loc) * _num_chunks * sizeof(data_t);
    diskann::prefetch_vector((const char *)ptr, _num_chunks * sizeof(data_t));
}

template <typename data_t>
void PQDataStore<data_t>::move_vectors(const location_t old_location_start, const location_t new_location_start,
                                       const location_t num_points)
{
    // REFACTOR TODO: Moving vectors is only for in-mem fresh.
    throw std::logic_error("Not implemented yet");
}

template <typename data_t>
void PQDataStore<data_t>::copy_vectors(const location_t from_loc, const location_t to_loc, const location_t num_points)
{
    // REFACTOR TODO: Is the number of bytes correct?
    memcpy(_quantized_data + to_loc * _num_chunks, _quantized_data + from_loc * _num_chunks, _num_chunks * num_points);
    if (!DISKANN_DISABLE_NUMA_REPLICA && _quantized_data_replica != nullptr)
    {
        memcpy(_quantized_data_replica + to_loc * _num_chunks, _quantized_data_replica + from_loc * _num_chunks,
               _num_chunks * num_points);
    }
}

// REFACTOR TODO: Currently, we take aligned_query as parameter, but this
// function should also do the alignment.
template <typename data_t>
void PQDataStore<data_t>::preprocess_query(const data_t *aligned_query, AbstractScratch<data_t> *scratch) const
{
    if (scratch == nullptr)
    {
        throw diskann::ANNException("Scratch space is null", -1);
    }

    PQScratch<data_t> *pq_scratch = scratch->pq_scratch();

    if (pq_scratch == nullptr)
    {
        throw diskann::ANNException("PQScratch space has not been set in the scratch object.", -1);
    }

    _pq_distance_fn->preprocess_query(aligned_query, (location_t)this->get_dims(), *pq_scratch);
}

template <typename data_t> float PQDataStore<data_t>::get_distance(const data_t *query, const location_t loc) const
{
    throw std::logic_error("Not implemented yet");
}

template <typename data_t> float PQDataStore<data_t>::get_distance(const location_t loc1, const location_t loc2) const
{
    throw std::logic_error("Not implemented yet");
}

template <typename data_t>
void PQDataStore<data_t>::get_distance(const data_t *preprocessed_query, const location_t *locations,
                                       const uint32_t location_count, float *distances,
                                       AbstractScratch<data_t> *scratch_space) const
{
    if (scratch_space == nullptr)
    {
        throw diskann::ANNException("Scratch space is null", -1);
    }
    PQScratch<data_t> *pq_scratch = scratch_space->pq_scratch();
    if (pq_scratch == nullptr)
    {
        throw diskann::ANNException("PQScratch not set in scratch space.", -1);
    }
    const uint8_t *source = _quantized_data;
#ifndef _WINDOWS
    if (!DISKANN_DISABLE_NUMA_REPLICA && _quantized_data_replica != nullptr)
    {
        const int node = get_current_numa_node();
        if (node == _numa_node_secondary)
        {
            source = _quantized_data_replica;
        }
    }
#endif
    diskann::aggregate_coords(locations, location_count, source, this->_num_chunks, pq_scratch->aligned_pq_coord_scratch);
    _pq_distance_fn->preprocessed_distance(*pq_scratch, location_count, distances);
}

template <typename data_t>
void PQDataStore<data_t>::get_distance(const data_t *preprocessed_query, const std::vector<location_t> &ids,
                                       std::vector<float> &distances, AbstractScratch<data_t> *scratch_space) const
{
    if (scratch_space == nullptr)
    {
        throw diskann::ANNException("Scratch space is null", -1);
    }
    PQScratch<data_t> *pq_scratch = scratch_space->pq_scratch();
    if (pq_scratch == nullptr)
    {
        throw diskann::ANNException("PQScratch not set in scratch space.", -1);
    }
    const uint8_t *source = _quantized_data;
#ifndef _WINDOWS
    if (!DISKANN_DISABLE_NUMA_REPLICA && _quantized_data_replica != nullptr)
    {
        const int node = get_current_numa_node();
        if (node == _numa_node_secondary)
        {
            source = _quantized_data_replica;
        }
    }
#endif
    diskann::aggregate_coords(ids, source, this->_num_chunks, pq_scratch->aligned_pq_coord_scratch);
    _pq_distance_fn->preprocessed_distance(*pq_scratch, (location_t)ids.size(), distances);
}

template <typename data_t> location_t PQDataStore<data_t>::calculate_medoid() const
{
    // REFACTOR TODO: Must calculate this just like we do with data store.
    size_t r = (size_t)rand() * (size_t)RAND_MAX + (size_t)rand();
    return (uint32_t)(r % (size_t)this->capacity());
}

template <typename data_t> size_t PQDataStore<data_t>::get_alignment_factor() const
{
    return 1;
}

template <typename data_t> Distance<data_t> *PQDataStore<data_t>::get_dist_fn() const
{
    return _distance_fn.get();
}

template <typename data_t> location_t PQDataStore<data_t>::load_impl(const std::string &file_prefix)
{
    if (_quantized_data != nullptr)
    {
#ifndef _WINDOWS
        if (_use_numa_replica)
        {
            const size_t bytes = this->capacity() * _num_chunks * sizeof(uint8_t);
            numa_free(_quantized_data, bytes);
        }
        else
#endif
        {
            aligned_free(_quantized_data);
        }
    }
    if (_quantized_data_replica != nullptr)
    {
#ifndef _WINDOWS
        const size_t bytes = this->capacity() * _num_chunks * sizeof(uint8_t);
        numa_free(_quantized_data_replica, bytes);
#endif
        _quantized_data_replica = nullptr;
    }
    auto quantized_vectors_file = _pq_distance_fn->get_quantized_vectors_filename(file_prefix);

    size_t num_points;
    uint8_t *loaded = nullptr;
    load_aligned_bin(quantized_vectors_file, loaded, num_points, _num_chunks, _num_chunks);
    this->_capacity = (location_t)num_points;

    const size_t bytes = this->capacity() * _num_chunks * sizeof(uint8_t);
#ifndef _WINDOWS
    if (!DISKANN_DISABLE_NUMA_REPLICA && _use_numa_replica)
    {
        _quantized_data = static_cast<uint8_t *>(numa_alloc_onnode(bytes, _numa_node_primary));
        _quantized_data_replica = static_cast<uint8_t *>(numa_alloc_onnode(bytes, _numa_node_secondary));
        if (_quantized_data == nullptr || _quantized_data_replica == nullptr)
        {
            aligned_free(loaded);
            throw diskann::ANNException("Failed to allocate NUMA replicas for PQ data store", -1, __FUNCSIG__,
                                        __FILE__, __LINE__);
        }
        std::memcpy(_quantized_data, loaded, bytes);
        std::memcpy(_quantized_data_replica, loaded, bytes);
        aligned_free(loaded);
    }
    else
#endif
    {
        _quantized_data = loaded;
    }

    auto pivots_file = _pq_distance_fn->get_pivot_data_filename(file_prefix);
    _pq_distance_fn->load_pivot_data(pivots_file, _num_chunks);

    return this->_capacity;
}

template <typename data_t> location_t PQDataStore<data_t>::expand(const location_t new_size)
{
    throw std::logic_error("Not implemented yet");
}

template <typename data_t> location_t PQDataStore<data_t>::shrink(const location_t new_size)
{
    throw std::logic_error("Not implemented yet");
}

#ifdef EXEC_ENV_OLS
template <typename data_t> location_t PQDataStore<data_t>::load_impl(AlignedFileReader &reader)
{
}
#endif

template DISKANN_DLLEXPORT class PQDataStore<int8_t>;
template DISKANN_DLLEXPORT class PQDataStore<float>;
template DISKANN_DLLEXPORT class PQDataStore<uint8_t>;

} // namespace diskann