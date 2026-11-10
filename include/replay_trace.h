// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <vector>

#include "tsl/robin_map.h"

namespace diskann
{
// Describes an expanded node in a previous query execution.
// Each expanded node owns a contiguous range of children in `children`.
struct ExpandedNode
{
    uint32_t id = 0;
    uint16_t child_start = 0;
    uint16_t child_end = 0;
    // uint16_t child_end = 0;
    // float distance = 0.0f;
    ExpandedNode() = default;
    
    ExpandedNode(int _id, uint16_t _start, uint16_t _end) 
        : id(_id), child_start(_start), child_end(_end) {}
};


// // Describes a previously visited node and its computed distance to the query.
struct VisitedNode
{
    uint32_t id = 0;
    float distance = 0.0f;

    VisitedNode() = default;

    VisitedNode(uint32_t node_id, float node_dist)
        : id(node_id), distance(node_dist)
    {}

    bool operator<(const VisitedNode& other) const {
        return this->id < other.id;
    }
};


struct ReplayTrace
{
    std::vector<ExpandedNode> expanded;
    NeighborPriorityQueue best_L_nodes;
    std::vector<VisitedNode> visited;
    // tsl::robin_map<uint32_t, float> visited_map;
};


struct UpdatedNodesInfo
{
    std::vector<uint8_t> changed_nodes;
    std::vector<float> farthest_ngh_distance;
    std::vector<uint64_t> added_ngh_bitmap;  
};

} // namespace diskann
