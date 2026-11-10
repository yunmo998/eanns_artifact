// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "numa_split_graph_store.h"

#include <algorithm>
#include <fstream>

#include "utils.h"

#if defined(__linux__)
#include <numa.h>
#endif

namespace diskann
{
namespace
{
#if defined(__linux__)
void bind_all_nodes()
{
    if (numa_available() != -1)
    {
        numa_run_on_node_mask(numa_all_nodes_ptr);
    }
}
#endif
} // namespace

NumaSplitGraphStore::NumaSplitGraphStore(const size_t total_pts, const size_t reserve_graph_degree, size_t split_id,
                                         int node0, int node1)
    : AbstractGraphStore(total_pts, reserve_graph_degree), _split_id(split_id), _node0(node0), _node1(node1)
{
    if (_split_id == 0 || _split_id > total_pts)
    {
        _split_id = total_pts / 2;
    }
    std::cout << "Create graph index on NUMA with split node: " << _split_id << ", node0: " << _node0 << ", node1: " << _node1 << std::endl; 

#if defined(__linux__)
    if (numa_available() == -1)
    {
        _split_id = total_pts;
        _store0 = std::make_unique<InMemGraphStore>(total_pts, reserve_graph_degree);
        return;
    }

    numa_run_on_node(_node0);
    _store0 = std::make_unique<InMemGraphStore>(_split_id, reserve_graph_degree);

    numa_run_on_node(_node1);
    _store1 = std::make_unique<InMemGraphStore>(total_pts - _split_id, reserve_graph_degree);

    bind_all_nodes();
#else
    _split_id = total_pts;
    _store0 = std::make_unique<InMemGraphStore>(total_pts, reserve_graph_degree);
#endif
}

bool NumaSplitGraphStore::is_store0(const location_t i) const
{
    return i < _split_id;
}

size_t NumaSplitGraphStore::local_index(const location_t i) const
{
    return is_store0(i) ? i : static_cast<size_t>(i - _split_id);
}

InMemGraphStore *NumaSplitGraphStore::store_for(const location_t i)
{
    return is_store0(i) ? _store0.get() : _store1.get();
}

const InMemGraphStore *NumaSplitGraphStore::store_for(const location_t i) const
{
    return is_store0(i) ? _store0.get() : _store1.get();
}

std::tuple<uint32_t, uint32_t, size_t> NumaSplitGraphStore::load(const std::string &index_path_prefix,
                                                                 const size_t num_points)
{
    size_t expected_file_size;
    size_t file_frozen_pts;
    uint32_t start;
    size_t file_offset = 0;

    std::ifstream in;
    in.exceptions(std::ios::badbit | std::ios::failbit);
    in.open(index_path_prefix, std::ios::binary);
    in.seekg(file_offset, in.beg);
    in.read((char *)&expected_file_size, sizeof(size_t));
    in.read((char *)&_max_observed_degree, sizeof(uint32_t));
    in.read((char *)&start, sizeof(uint32_t));
    in.read((char *)&file_frozen_pts, sizeof(size_t));
    size_t vamana_metadata_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

    diskann::cout << "From graph header, expected_file_size: " << expected_file_size
                  << ", _max_observed_degree: " << _max_observed_degree << ", _start: " << start
                  << ", file_frozen_pts: " << file_frozen_pts << std::endl;

    diskann::cout << "Loading vamana graph " << index_path_prefix << "..." << std::flush;

    if (get_total_points() < num_points)
    {
        diskann::cout << "resizing graph to " << num_points << std::endl;
        this->resize_graph(num_points);
    }

    size_t bytes_read = vamana_metadata_size;
    size_t cc = 0;
    uint32_t nodes_read = 0;
    _max_range_of_graph = 0;

    while (bytes_read != expected_file_size)
    {
        uint32_t k;
        in.read((char *)&k, sizeof(uint32_t));

        if (k == 0)
        {
            diskann::cerr << "ERROR: Point found with no out-neighbours, point#" << nodes_read << std::endl;
        }

        cc += k;
        ++nodes_read;
        std::vector<uint32_t> tmp(k);
        tmp.reserve(k);
        in.read((char *)tmp.data(), k * sizeof(uint32_t));

        auto store = store_for(nodes_read - 1);
        store->set_neighbours(static_cast<location_t>(local_index(nodes_read - 1)), tmp);

        bytes_read += sizeof(uint32_t) * ((size_t)k + 1);
        if (nodes_read % 10000000 == 0)
        {
            diskann::cout << "." << std::flush;
        }
        if (k > _max_range_of_graph)
        {
            _max_range_of_graph = k;
        }
    }

    diskann::cout << "done. Index has " << nodes_read << " nodes and " << cc << " out-edges, _start is set to " << start
                  << std::endl;
    return std::make_tuple(nodes_read, start, file_frozen_pts);
}

int NumaSplitGraphStore::store(const std::string &index_path_prefix, const size_t num_points,
                               const size_t num_frozen_points, const uint32_t start)
{
    std::ofstream out;
    open_file_to_write(out, index_path_prefix);

    size_t file_offset = 0;
    out.seekp(file_offset, out.beg);
    size_t index_size = 24;
    uint32_t max_degree = 0;
    out.write((char *)&index_size, sizeof(uint64_t));
    out.write((char *)&max_degree, sizeof(uint32_t));
    uint32_t ep_u32 = start;
    out.write((char *)&ep_u32, sizeof(uint32_t));
    out.write((char *)&num_frozen_points, sizeof(size_t));

    for (uint32_t i = 0; i < num_points; i++)
    {
        const auto &nbrs = get_neighbours(i);
        uint32_t GK = (uint32_t)nbrs.size();
        out.write((char *)&GK, sizeof(uint32_t));
        out.write((char *)nbrs.data(), GK * sizeof(uint32_t));
        max_degree = nbrs.size() > max_degree ? (uint32_t)nbrs.size() : max_degree;
        index_size += (size_t)(sizeof(uint32_t) * (GK + 1));
    }
    out.seekp(file_offset, out.beg);
    out.write((char *)&index_size, sizeof(uint64_t));
    out.write((char *)&max_degree, sizeof(uint32_t));
    out.close();
    return (int)index_size;
}

const std::vector<location_t> &NumaSplitGraphStore::get_neighbours(const location_t i) const
{
    return store_for(i)->get_neighbours(static_cast<location_t>(local_index(i)));
}

void NumaSplitGraphStore::add_neighbour(const location_t i, location_t neighbour_id)
{
    store_for(i)->add_neighbour(static_cast<location_t>(local_index(i)), neighbour_id);
}

void NumaSplitGraphStore::clear_neighbours(const location_t i)
{
    store_for(i)->clear_neighbours(static_cast<location_t>(local_index(i)));
}

void NumaSplitGraphStore::swap_neighbours(const location_t a, const location_t b)
{
    if (is_store0(a) == is_store0(b))
    {
        store_for(a)->swap_neighbours(static_cast<location_t>(local_index(a)), static_cast<location_t>(local_index(b)));
        return;
    }

    std::vector<location_t> a_copy = get_neighbours(a);
    std::vector<location_t> b_copy = get_neighbours(b);

    set_neighbours(a, b_copy);
    set_neighbours(b, a_copy);
}

void NumaSplitGraphStore::set_neighbours(const location_t i, std::vector<location_t> &neighbours)
{
    store_for(i)->set_neighbours(static_cast<location_t>(local_index(i)), neighbours);
}

size_t NumaSplitGraphStore::resize_graph(const size_t new_size)
{
    if (new_size <= _split_id)
    {
        _store0->resize_graph(new_size);
        if (_store1)
        {
            _store1->resize_graph(0);
        }
    }
    else
    {
        _store0->resize_graph(_split_id);
        if (_store1)
        {
            _store1->resize_graph(new_size - _split_id);
        }
        else
        {
            _store1 = std::make_unique<InMemGraphStore>(new_size - _split_id, get_reserve_graph_degree());
        }
    }

    set_total_points(new_size);
    return new_size;
}

void NumaSplitGraphStore::clear_graph()
{
    _store0->clear_graph();
    if (_store1)
    {
        _store1->clear_graph();
    }
}

size_t NumaSplitGraphStore::get_max_range_of_graph()
{
    return _max_range_of_graph;
}

uint32_t NumaSplitGraphStore::get_max_observed_degree()
{
    uint32_t max0 = _store0 ? _store0->get_max_observed_degree() : 0;
    uint32_t max1 = _store1 ? _store1->get_max_observed_degree() : 0;
    return std::max(max0, max1);
}

} // namespace diskann
