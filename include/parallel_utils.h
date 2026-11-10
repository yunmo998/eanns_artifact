// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <functional>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#endif

namespace diskann
{

// Parallel-for dispatch with NUMA-aware SMT scheduling.
//
// Usage (replaces `#pragma omp parallel for schedule(dynamic, 1)` in apps):
//
//   parallel_for_smt(0, query_num, num_threads_per_node,
//       [&](int64_t thread_idx, int64_t query_idx, int numa_node) {
//           // The calling thread is already pinned to the correct
//           // SMT slot on a core within this NUMA node.
//           index.search_with_tags_replay(...);
//       });
//
// Each physical core runs `smt_factor` threads (typically 2).
// Threads on the same physical core share L1/L2 caches and
// hide each other's memory stalls.
//
// num_threads_per_node: number of OpenMP-like "threads" the user requested
//   per NUMA node. Internally each is split into `smt_factor` sub-threads.
inline void parallel_for_smt(int64_t start, int64_t end,
                             int num_threads_per_node,
                             std::function<void(int64_t thread_idx, int64_t work_idx,
                                                int numa_node)> fn)
{
#if defined(__linux__)
    if (numa_available() == -1)
    {
        // Fallback: no NUMA awareness, just use std::thread
        int total = static_cast<int>(std::thread::hardware_concurrency());
        total = std::max(total, num_threads_per_node);
        std::vector<std::thread> threads;
        threads.reserve(total);
        for (int t = 0; t < total; ++t)
        {
            threads.emplace_back([&, t]() {
                for (int64_t i = start + t; i < end; i += total)
                    fn(t, i, -1);
            });
        }
        for (auto &th : threads) th.join();
        return;
    }

    int max_node = numa_max_node();
    int64_t total_work = end - start;

    for (int node = 0; node <= max_node; ++node)
    {
        // --- Detect SMT topology for this NUMA node ---
        struct bitmask *mask = numa_allocate_cpumask();
        if (mask == nullptr || numa_node_to_cpus(node, mask) != 0)
        {
            if (mask) numa_free_cpumask(mask);
            continue;
        }

        std::vector<int> cpus_on_node;
        for (unsigned int cpu = 0; cpu < mask->size; ++cpu)
            if (numa_bitmask_isbitset(mask, cpu))
                cpus_on_node.push_back(static_cast<int>(cpu));
        numa_free_cpumask(mask);

        if (cpus_on_node.empty())
            continue;

        // Detect SMT factor: on most x86 Linux, cpus_on_node looks like
        // [0,1, 2,3, 4,5, ...] where consecutive pairs are SMT siblings.
        int smt_factor = 1;
        for (size_t i = 1; i < cpus_on_node.size(); ++i)
        {
            if (cpus_on_node[i] == cpus_on_node[i - 1] + 1)
            {
                smt_factor = 2;
                break;
            }
        }
        int phys_cores_on_node =
            std::max(1, static_cast<int>(cpus_on_node.size()) / smt_factor);

        // The user requested num_threads_per_node "logical threads" worth
        // of parallelism per node. With SMT, we create SMT slots on top.
        int logical_slots = std::min(num_threads_per_node, phys_cores_on_node);
        int total_smt_threads = logical_slots * smt_factor;

        // --- Launch SMT threads for this NUMA node ---
        std::vector<std::thread> threads;
        threads.reserve(total_smt_threads);

        for (int t = 0; t < total_smt_threads; ++t)
        {
            threads.emplace_back([&, t, node, logical_slots, smt_factor,
                                  &cpus_on_node]() {
                int phys_core = (t / smt_factor) % logical_slots;
                int smt_slot = t % smt_factor;
                int cpu_idx = phys_core * smt_factor + smt_slot;
                if (cpu_idx >= static_cast<int>(cpus_on_node.size()))
                    cpu_idx = cpus_on_node.size() - 1;
                int target_cpu = cpus_on_node[cpu_idx];

                // Pin to specific logical CPU
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(target_cpu, &cpuset);
                sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);

                // NUMA memory policy: local allocation
                numa_run_on_node(node);

                // Work-stealing dispatch: each SMT thread takes work
                // items in a strided pattern.
                // All threads across all nodes share the work range.
                int global_stride = (max_node + 1) * total_smt_threads;
                int global_offset = node * total_smt_threads + t;

                for (int64_t i = start + global_offset; i < end;
                     i += global_stride)
                {
                    fn(t, i, node);
                }
            });
        }

        for (auto &th : threads) th.join();
    }
#else
    // Non-Linux: simple thread pool
    int total = static_cast<int>(std::thread::hardware_concurrency());
    total = std::max(total, num_threads_per_node);
    std::vector<std::thread> threads;
    threads.reserve(total);
    for (int t = 0; t < total; ++t)
    {
        threads.emplace_back([&, t]() {
            for (int64_t i = start + t; i < end; i += total)
                fn(t, i, -1);
        });
    }
    for (auto &th : threads) th.join();
#endif
}

} // namespace diskann
