# ================================================================
# EANNS — standalone build for the ICDE artifact.
#
# Builds the DiskANN core library (sources under ../src) plus the
# three programs needed to reproduce the paper:
#   - eanns                            : the EANNS system entry point
#   - test_insert_deletes_consolidate  : builds the first snapshot index
#   - compute_groundtruth              : computes ground truth
#
# No file in the parent repository is modified by this build.
#
# Usage:
#   cmake -S . -B build
#   cmake --build build -j
#
# Options:
#   -DPORTABLE=ON              disable -march=native (portable binaries)
#   -DMKL_INCLUDE_PATH=<dir>   override the MKL header location
#   -DOMP_PATH=<dir>           override the libiomp5 location
# ================================================================
cmake_minimum_required(VERSION 3.15)
project(eanns CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(PORTABLE "Disable -march=native (portable build)" OFF)

set(DISKANN_SRC  "${CMAKE_CURRENT_SOURCE_DIR}/../src")
set(DISKANN_INC  "${CMAKE_CURRENT_SOURCE_DIR}/../include")
set(DISKANN_APPS "${CMAKE_CURRENT_SOURCE_DIR}/../apps")

# --- Core library sources (same list as the parent src/CMakeLists.txt) ---
set(CPP_SOURCES
    ${DISKANN_SRC}/abstract_data_store.cpp
    ${DISKANN_SRC}/ann_exception.cpp
    ${DISKANN_SRC}/disk_utils.cpp
    ${DISKANN_SRC}/distance.cpp
    ${DISKANN_SRC}/index.cpp
    ${DISKANN_SRC}/in_mem_graph_store.cpp
    ${DISKANN_SRC}/numa_split_graph_store.cpp
    ${DISKANN_SRC}/in_mem_data_store.cpp
    ${DISKANN_SRC}/linux_aligned_file_reader.cpp
    ${DISKANN_SRC}/math_utils.cpp
    ${DISKANN_SRC}/natural_number_map.cpp
    ${DISKANN_SRC}/natural_number_set.cpp
    ${DISKANN_SRC}/memory_mapper.cpp
    ${DISKANN_SRC}/partition.cpp
    ${DISKANN_SRC}/pq.cpp
    ${DISKANN_SRC}/pq_flash_index.cpp
    ${DISKANN_SRC}/scratch.cpp
    ${DISKANN_SRC}/logger.cpp
    ${DISKANN_SRC}/utils.cpp
    ${DISKANN_SRC}/filter_utils.cpp
    ${DISKANN_SRC}/index_factory.cpp
    ${DISKANN_SRC}/abstract_index.cpp
    ${DISKANN_SRC}/pq_l2_distance.cpp
    ${DISKANN_SRC}/pq_data_store.cpp
)

include_directories(${DISKANN_INC})

# --- Dependencies ---------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

# --- Intel MKL header detection (same locations as the parent build) ------
if(NOT MKL_INCLUDE_PATH)
    foreach(P "/opt/intel/oneapi/mkl/latest/include" "/usr/include/mkl" "/opt/intel/mkl/include")
        if(EXISTS "${P}")
            set(MKL_INCLUDE_PATH "${P}")
            break()
        endif()
    endforeach()
endif()
if(NOT MKL_INCLUDE_PATH)
    message(FATAL_ERROR "Could not find MKL headers. Use -DMKL_INCLUDE_PATH=<dir>.")
endif()
include_directories(${MKL_INCLUDE_PATH})

# --- Intel OpenMP runtime (libiomp5 / LLVM libomp) detection ---------------
if(NOT OMP_PATH)
    foreach(P "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so"
             "/usr/lib/x86_64-linux-gnu/libiomp5.so"
             "/opt/intel/lib/intel64_lin/libiomp5.so")
        if(EXISTS "${P}")
            get_filename_component(OMP_PATH "${P}" DIRECTORY)
            break()
        endif()
    endforeach()
endif()
if(OMP_PATH)
    link_directories(${OMP_PATH})
endif()

# --- Compile options (mirrors the parent build) ---------------------------
add_definitions(-DMKL_ILP64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp-simd -funroll-loops -DUSE_AVX2")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -Ofast")
if(NOT PORTABLE)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
endif()

# --- Core library ---------------------------------------------------------
add_library(diskann STATIC ${CPP_SOURCES})

# MKL + Intel OpenMP runtime + numa + tcmalloc (as linked by the parent tools)
set(EANNS_LINK_LIBS mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 pthread m dl numa tcmalloc)

# --- EANNS -----------------------------------------------------------------
add_executable(eanns eanns.cpp)
target_link_libraries(eanns diskann Boost::program_options ${EANNS_LINK_LIBS})

# --- Snapshot builder ------------------------------------------------------
add_executable(test_insert_deletes_consolidate ${DISKANN_APPS}/test_insert_deletes_consolidate.cpp)
target_link_libraries(test_insert_deletes_consolidate diskann Boost::program_options ${EANNS_LINK_LIBS})

# --- Ground-truth computation ----------------------------------------------
add_executable(compute_groundtruth ${DISKANN_APPS}/utils/compute_groundtruth.cpp)
target_link_libraries(compute_groundtruth diskann Boost::program_options ${EANNS_LINK_LIBS})
