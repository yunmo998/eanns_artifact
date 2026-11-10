#!/usr/bin/env bash
# ================================================================
# EANNS — reproduction pipeline
#
# Steps:
#   1. build the standalone project (cmake)
#   2. construct the first snapshot index   (test_insert_deletes_consolidate)
#   3. compute ground truth for the snapshot (compute_groundtruth, range 0..pts01)
#   4. construct the full index              (test_insert_deletes_consolidate)
#   5. compute ground truth for the full range (compute_groundtruth, range 0..pts02)
#   6. run EANNS (load snapshot + insert delta + two searches)
#
# The EANNS program performs TWO searches internally:
#   search-1 (after loading the snapshot): GT = ${GT_PREFIX}0-${pts01}
#   search-2 (after inserting the delta):  GT = ${GT_PREFIX}0-${pts02}
#
# Assumption: the datasets already exist (set the paths below).
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

export OMP_PLACES=cores
export OMP_PROC_BIND=close

# ---------------- Configurable paths and parameters ----------------
# Data type: float | int8 | uint8
DATA_TYPE='float'
# Distance function: l2 | cosine | mips | fast_l2
DIST_FN='l2'

# First snapshot data (fbin) — points 0..pts01
DATA01='/path/to/snapshot_data.fbin'
# Full data (fbin) containing snapshot + delta — points 0..pts02
DATA02='/path/to/full_data.fbin'
# Query file (fbin)
QUERY='/path/to/query.fbin'

# Ground-truth naming: ${GT_PREFIX}{deletes}-{last_point_threshold}
GT_PREFIX='/path/to/gt/gt100_'

# Output locations
INDEX_PREFIX01='/path/to/index/snapshot/index'
INDEX_PREFIX02='/path/to/index/full/index'
RESULT_PATH='/path/to/results/evolving'

# Build/search parameters
R=64            # max graph degree
LB=300          # build L (graph build complexity)
ALPHA=1.2
THREADS=64
K=10            # search K (recall@K)
GT_K=100        # ground-truth K
L_SEARCH=200    # search L
# ----------------------------------------------------------------

get_npts() {
    python3 -c "import struct; f=open('$1','rb'); print(struct.unpack('<i', f.read(4))[0])"
}

pts01=$(get_npts "${DATA01}")
pts02=$(get_npts "${DATA02}")
echo "Snapshot points (DATA01): ${pts01}"
echo "Full points       (DATA02): ${pts02}"

index01="${INDEX_PREFIX01}.after-delete-del0-${pts01}"
index02="${INDEX_PREFIX02}.after-delete-del0-${pts02}"
gt_file01="${GT_PREFIX}0-${pts01}"
gt_file02="${GT_PREFIX}0-${pts02}"

mkdir -p "$(dirname "${INDEX_PREFIX01}")" "$(dirname "${INDEX_PREFIX02}")" \
         "$(dirname "${GT_PREFIX}")" "$(dirname "${RESULT_PATH}")"

# ================================================================
echo "===== Step 1: build ====="
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j

# ================================================================
echo "===== Step 2: construct first snapshot index ====="
"${BUILD_DIR}/test_insert_deletes_consolidate" \
    --data_type "${DATA_TYPE}" --dist_fn "${DIST_FN}" --data_path "${DATA01}" \
    --index_path_prefix "${INDEX_PREFIX01}" \
    -R "${R}" -L "${LB}" --alpha "${ALPHA}" -T "${THREADS}" \
    --points_to_skip 0 \
    --max_points_to_insert "${pts01}" \
    --beginning_index_size "${pts01}" \
    --points_per_checkpoint "${pts01}" \
    --checkpoints_per_snapshot 0 \
    --points_to_delete_from_beginning 0 \
    --start_deletes_after "${pts01}" \
    --do_concurrent false

echo "===== Step 3: compute GT for snapshot (0..pts01) ====="
"${BUILD_DIR}/compute_groundtruth" \
    --data_type "${DATA_TYPE}" --dist_fn "${DIST_FN}" \
    --base_file "${index01}.data" \
    --query_file "${QUERY}" \
    --K "${GT_K}" \
    --gt_file "${gt_file01}" \
    --tags_file "${index01}.tags"

# ================================================================
echo "===== Step 4: construct full index (for GT of full range) ====="
"${BUILD_DIR}/test_insert_deletes_consolidate" \
    --data_type "${DATA_TYPE}" --dist_fn "${DIST_FN}" --data_path "${DATA02}" \
    --index_path_prefix "${INDEX_PREFIX02}" \
    -R "${R}" -L "${LB}" --alpha "${ALPHA}" -T "${THREADS}" \
    --points_to_skip 0 \
    --max_points_to_insert "${pts02}" \
    --beginning_index_size "${pts02}" \
    --points_per_checkpoint "${pts02}" \
    --checkpoints_per_snapshot 0 \
    --points_to_delete_from_beginning 0 \
    --start_deletes_after "${pts02}" \
    --do_concurrent false

echo "===== Step 5: compute GT for full range (0..pts02) ====="
"${BUILD_DIR}/compute_groundtruth" \
    --data_type "${DATA_TYPE}" --dist_fn "${DIST_FN}" \
    --base_file "${index02}.data" \
    --query_file "${QUERY}" \
    --K "${GT_K}" \
    --gt_file "${gt_file02}" \
    --tags_file "${index02}.tags"

# ================================================================
echo "===== Step 6: run EANNS (snapshot + delta) ====="
delta_size=$((pts02 - pts01))
echo "Delta size (pts02 - pts01): ${delta_size}"

"${BUILD_DIR}/eanns" \
    --data_type "${DATA_TYPE}" --dist_fn "${DIST_FN}" \
    --data_path "${DATA02}" \
    --index_path_prefix "${INDEX_PREFIX01}" \
    -R "${R}" -L "${L_SEARCH}" --alpha "${ALPHA}" -T "${THREADS}" \
    --points_to_skip 0 \
    --max_points_to_insert "${pts02}" \
    --beginning_index_size 0 \
    --points_per_checkpoint "${delta_size}" \
    --checkpoints_per_snapshot 0 \
    --points_to_delete_from_beginning 0 \
    --start_deletes_after "${pts01}" \
    --do_concurrent false \
    --start_point_norm 2 \
    --result_path "${RESULT_PATH}" \
    --query_file "${QUERY}" \
    --gt_file_prefix "${GT_PREFIX}" \
    -K "${K}" -l "${L_SEARCH}" \
    --dynamic true --tags 1

echo "===== All done ====="
