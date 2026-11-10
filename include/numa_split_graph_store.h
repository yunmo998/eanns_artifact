#pragma once

#include <memory>
#include <string>
#include <vector>

#include "abstract_graph_store.h"
#include "in_mem_graph_store.h"

namespace diskann
{

class NumaSplitGraphStore : public AbstractGraphStore
{
  public:
    NumaSplitGraphStore(const size_t total_pts, const size_t reserve_graph_degree, size_t split_id = 0, int node0 = 0,
                        int node1 = 1);

    // returns tuple of <nodes_read, start, num_frozen_points>
    std::tuple<uint32_t, uint32_t, size_t> load(const std::string &index_path_prefix,
                                                const size_t num_points) override;
    int store(const std::string &index_path_prefix, const size_t num_points, const size_t num_frozen_points,
              const uint32_t start) override;

    const std::vector<location_t> &get_neighbours(const location_t i) const override;
    void add_neighbour(const location_t i, location_t neighbour_id) override;
    void clear_neighbours(const location_t i) override;
    void swap_neighbours(const location_t a, location_t b) override;

    void set_neighbours(const location_t i, std::vector<location_t> &neighbours) override;

    size_t resize_graph(const size_t new_size) override;
    void clear_graph() override;

    size_t get_max_range_of_graph() override;
    uint32_t get_max_observed_degree() override;

  private:
    size_t _split_id;
    int _node0;
    int _node1;
    size_t _max_range_of_graph = 0;
    uint32_t _max_observed_degree = 0;

    std::unique_ptr<InMemGraphStore> _store0;
    std::unique_ptr<InMemGraphStore> _store1;

    bool is_store0(const location_t i) const;
    size_t local_index(const location_t i) const;
    InMemGraphStore *store_for(const location_t i);
    const InMemGraphStore *store_for(const location_t i) const;
};

} // namespace diskann
