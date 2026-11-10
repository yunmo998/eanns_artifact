#!/usr/bin/env bash
# ================================================================
# EANNS — consolidated run script
#
# Usage:
#   bash run_eanns.sh <dataset>
#
# Datasets:
#   redcap        two-snapshot  (2020_01 + 2020_02, float, L2)
#   redcap_daily  two-snapshot  (2020_01 + 2020_02_01 daily delta, float, L2)
#   dblp          two-snapshot  (snapshot_2023 + snapshot_2024, float, L2)
#   spacev   same-file 50% -> 51% (int8, L2)
#   gist     sliding window, 50% base + 1% delta (float, L2)
#   bigann   sliding window, 50% base + 1% delta (uint8, L2)
#
# Pipeline per dataset:
#   Step 1: build (cmake)
#   Step 2: construct the first snapshot index (test_insert_deletes_consolidate)
#   Step 3: compute ground truth for the snapshot (compute_groundtruth)
#   Step 4: construct the full index (for the second ground truth)
#   Step 5: compute ground truth for the full range
#   Step 6: run EANNS (load snapshot + insert delta + two searches)
#
# The EANNS program performs TWO searches internally:
#   search-1 (after loading the snapshot): GT = {gt_prefix}0-{pts01}
#   search-2 (after inserting the delta):  GT = {gt_prefix}{deletes}-{pts02}
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

if [ -z "${ANNS_DATA_PATH:-}" ]; then
    echo "ERROR: ANNS_DATA_PATH is not set"
    exit 1
fi

export OMP_PLACES=cores
export OMP_PROC_BIND=close

get_npts() {
    python3 -c "import struct; f=open('$1','rb'); print(struct.unpack('<i', f.read(4))[0])"
}

usage() {
    grep '^#   ' "$0" | sed 's/^#   //'
}

DATASET="${1:-}"
if [ -z "${DATASET}" ]; then
    usage
    echo ""
    echo "ERROR: no dataset given"
    exit 1
fi

# ---------------- Common parameters ----------------
R=64            # max graph degree
LB=300          # build L (graph build complexity)
ALPHA=1.2
T=64            # threads
K=10            # search K (recall@K)
GT_K=100        # ground-truth K
DIST_FN='l2'
WARMSTART='false'   # warm-start replay for the updated-index search

# ---------------- Per-dataset configuration ----------------
case "${DATASET}" in
    redcap)
        TYPE='float'
        DATA01="${ANNS_DATA_PATH}/red_cap/diskann_fbin/2020_01_accumulated.fbin"
        DATA02="${ANNS_DATA_PATH}/red_cap/diskann_fbin/2020_02_accumulated.fbin"
        QUERY="${ANNS_DATA_PATH}/red_cap/diskann_fbin/query_2020_latest_10k.fbin"
        GT_PREFIX="${ANNS_DATA_PATH}/red_cap/gt/gt100_"
        INDEX01="${ANNS_DATA_PATH}/red_cap/index/202001/index"
        INDEX02="${ANNS_DATA_PATH}/red_cap/index/202002/index"
        RESULT_BASE="${ANNS_DATA_PATH}/redcap/res/evolving"
        L=200
        MODE='snapshot'
        WARMSTART='true'
        ;;
    redcap_daily)
        TYPE='float'
        DATA01="${ANNS_DATA_PATH}/red_cap/diskann_fbin/2020_01_accumulated.fbin"
        DATA02="${ANNS_DATA_PATH}/red_cap/diskann_fbin/2020_02_01_accumulated.fbin"
        QUERY="${ANNS_DATA_PATH}/red_cap/diskann_fbin/query_2020_latest_10k.fbin"
        GT_PREFIX="${ANNS_DATA_PATH}/red_cap/gt/gt100_"
        INDEX01="${ANNS_DATA_PATH}/red_cap/index/202001/index"
        INDEX02="${ANNS_DATA_PATH}/red_cap/index/20200201/index"
        RESULT_BASE="${ANNS_DATA_PATH}/redcap/res/evolving"
        L=200
        MODE='snapshot'
        ;;
    dblp)
        TYPE='float'
        DATA01="${ANNS_DATA_PATH}/dblp/snapshot_2023.fbin"
        DATA02="${ANNS_DATA_PATH}/dblp/snapshot_2024.fbin"
        QUERY="${ANNS_DATA_PATH}/dblp/query_2025.fbin"
        GT_PREFIX="${ANNS_DATA_PATH}/dblp/gt/gt100_"
        INDEX01="${ANNS_DATA_PATH}/dblp/index/2023/index"
        INDEX02="${ANNS_DATA_PATH}/dblp/index/2024/index"
        RESULT_BASE="${ANNS_DATA_PATH}/dblp/res/evolving"
        L=100
        MODE='snapshot'
        ;;
    spacev)
        TYPE='int8'
        DATA01="${ANNS_DATA_PATH}/spacev/base.100M.i8bin"
        DATA02="${DATA01}"
        QUERY="${ANNS_DATA_PATH}/spacev/query.30K.i8bin"
        GT_PREFIX="${ANNS_DATA_PATH}/spacev/gt/gt100_"
        INDEX01="${ANNS_DATA_PATH}/spacev/index/index"
        INDEX02="${INDEX01}"
        RESULT_BASE="${ANNS_DATA_PATH}/spacev/res/evolving"
        L=100
        MODE='snapshot'
        SPACEV_FRAC='50:51'   # pts01 : pts02 as % of total points
        WARMSTART='true'
        ;;
    gist)
        TYPE='float'
        DATA="${ANNS_DATA_PATH}/gist/gist_learn.fbin"
        QUERY="${ANNS_DATA_PATH}/gist/gist_query.fbin"
        GT_PREFIX="${ANNS_DATA_PATH}/gist/gt100_learn-conc-"
        INDEX_PREFIX="${ANNS_DATA_PATH}/gist/index/index"
        RESULT_BASE="${ANNS_DATA_PATH}/gist/res/evolving"
        L=100
        NUM_POINTS=500000
        MODE='sliding'
        WARMSTART='true'
        ;;
    bigann)
        TYPE='uint8'
        DATA="${ANNS_DATA_PATH}/bigann/bigann100m_base.u8bin"
        QUERY="${ANNS_DATA_PATH}/bigann/bigann100m_query.u8bin"
        GT_PREFIX="${ANNS_DATA_PATH}/bigann/gt100_learn-conc-"
        INDEX_PREFIX="${ANNS_DATA_PATH}/bigann/index/index"
        RESULT_BASE="${ANNS_DATA_PATH}/bigann/res/evolving"
        L=100
        NUM_POINTS=100000000
        MODE='sliding'
        WARMSTART='true'
        ;;
    *)
        usage
        echo ""
        echo "ERROR: unknown dataset '${DATASET}'"
        exit 1
        ;;
esac

# ---------------- Point counts / delete parameters ----------------
if [ "${MODE}" = 'snapshot' ]; then
    if [ -n "${SPACEV_FRAC:-}" ]; then
        total_pts=$(get_npts "${DATA01}")
        frac1="${SPACEV_FRAC%%:*}"
        frac2="${SPACEV_FRAC##*:}"
        pts01=$((total_pts * frac1 / 100))
        pts02=$((total_pts * frac2 / 100))
    else
        pts01=$(get_npts "${DATA01}")
        pts02=$(get_npts "${DATA02}")
    fi
    deletes=0
    deletes_after=${pts01}
    delta_size=$((pts02 - pts01))
    # snapshot mode: eanns reads the full (second) data file
    data_eanns="${DATA02}"
    index_eanns="${INDEX01}"
    index_gt2="${INDEX02}"
else
    # sliding window: base = 50%, delta = 1%, delete 1% from the front
    base_size=$((NUM_POINTS / 2))
    delta_size=$((NUM_POINTS / 100))
    pts01=${base_size}
    pts02=$((base_size + delta_size))
    deletes=${delta_size}
    deletes_after=${base_size}
    data_eanns="${DATA}"
    index_eanns="${INDEX_PREFIX}"
    index_gt2="${INDEX_PREFIX}"
fi

index01="${index_eanns}.after-delete-del0-${pts01}"
gt_file01="${GT_PREFIX}0-${pts01}"
gt_file02="${GT_PREFIX}${deletes}-${pts02}"

mkdir -p "$(dirname "${index_eanns}")" "$(dirname "${index_gt2}")" \
         "$(dirname "${GT_PREFIX}")" "${RESULT_BASE}"

echo "Dataset : ${DATASET}"
echo "Mode    : ${MODE}"
echo "pts01   : ${pts01}"
echo "pts02   : ${pts02}"
echo "deletes : ${deletes}  (deletes_after: ${deletes_after})"
echo "delta   : ${delta_size}"

# ================================================================
echo "===== Step 1: build ====="
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j

# ================================================================
echo "===== Step 2: construct first snapshot index (0..${pts01}) ====="
"${BUILD_DIR}/test_insert_deletes_consolidate" \
    --data_type "${TYPE}" --dist_fn "${DIST_FN}" --data_path "${DATA01}" \
    --index_path_prefix "${index_eanns}" \
    -R "${R}" -L "${LB}" --alpha "${ALPHA}" -T "${T}" \
    --points_to_skip 0 \
    --max_points_to_insert "${pts01}" \
    --beginning_index_size "${pts01}" \
    --points_per_checkpoint "${pts01}" \
    --checkpoints_per_snapshot 0 \
    --points_to_delete_from_beginning 0 \
    --start_deletes_after "${pts01}" \
    --do_concurrent false

echo "===== Step 3: compute GT for snapshot (0..${pts01}) ====="
"${BUILD_DIR}/compute_groundtruth" \
    --data_type "${TYPE}" --dist_fn "${DIST_FN}" \
    --base_file "${index01}.data" \
    --query_file "${QUERY}" \
    --K "${GT_K}" \
    --gt_file "${gt_file01}" \
    --tags_file "${index01}.tags"

# ================================================================
echo "===== Step 4: construct full index (for GT of range 0..${pts02}) ====="
if [ "${MODE}" = 'snapshot' ]; then
    "${BUILD_DIR}/test_insert_deletes_consolidate" \
        --data_type "${TYPE}" --dist_fn "${DIST_FN}" --data_path "${DATA02}" \
        --index_path_prefix "${index_gt2}" \
        -R "${R}" -L "${LB}" --alpha "${ALPHA}" -T "${T}" \
        --points_to_skip 0 \
        --max_points_to_insert "${pts02}" \
        --beginning_index_size "${pts02}" \
        --points_per_checkpoint "${pts02}" \
        --checkpoints_per_snapshot 0 \
        --points_to_delete_from_beginning 0 \
        --start_deletes_after "${pts02}" \
        --do_concurrent false
    index_gt2_full="${index_gt2}.after-delete-del0-${pts02}"
else
    # Build the post-update index (insert 1%, delete 1% from the front),
    # which yields the .data/.tags needed for GT2 with original tags.
    "${BUILD_DIR}/test_insert_deletes_consolidate" \
        --data_type "${TYPE}" --dist_fn "${DIST_FN}" --data_path "${DATA}" \
        --index_path_prefix "${index_gt2}" \
        -R "${R}" -L "${LB}" --alpha "${ALPHA}" -T "${T}" \
        --points_to_skip 0 \
        --max_points_to_insert "${pts02}" \
        --beginning_index_size 0 \
        --points_per_checkpoint "${delta_size}" \
        --checkpoints_per_snapshot 0 \
        --points_to_delete_from_beginning "${deletes}" \
        --start_deletes_after "${deletes_after}" \
        --do_concurrent false \
        --start_point_norm 2
    index_gt2_full="${index_gt2}.after-delete-del${deletes}-${pts02}"
fi

echo "===== Step 5: compute GT for full range (0..${pts02}) ====="
"${BUILD_DIR}/compute_groundtruth" \
    --data_type "${TYPE}" --dist_fn "${DIST_FN}" \
    --base_file "${index_gt2_full}.data" \
    --query_file "${QUERY}" \
    --K "${GT_K}" \
    --gt_file "${gt_file02}" \
    --tags_file "${index_gt2_full}.tags"

# ================================================================
echo "===== Step 6: run EANNS (snapshot + delta) ====="
RESULT_PATH="${RESULT_BASE}/l${L}"
mkdir -p "${RESULT_PATH}"

"${BUILD_DIR}/eanns" \
    --data_type "${TYPE}" --dist_fn "${DIST_FN}" \
    --data_path "${data_eanns}" \
    --index_path_prefix "${index_eanns}" \
    -R "${R}" -L "${L}" --alpha "${ALPHA}" -T "${T}" \
    --points_to_skip 0 \
    --max_points_to_insert "${pts02}" \
    --beginning_index_size 0 \
    --points_per_checkpoint "${delta_size}" \
    --checkpoints_per_snapshot 0 \
    --points_to_delete_from_beginning "${deletes}" \
    --start_deletes_after "${deletes_after}" \
    --do_concurrent false \
    --warmstart "${WARMSTART}" \
    --start_point_norm 2 \
    --result_path "${RESULT_PATH}" \
    --query_file "${QUERY}" \
    --gt_file_prefix "${GT_PREFIX}" \
    -K "${K}" -l "${L}" \
    --dynamic true --tags 1

echo "===== All done ====="
