# EANNS — ICDE Artifact

Standalone, self-contained build and reproduction entry point for the **EANNS**
system described in the paper.

EANNS builds an in-memory dynamic graph index over a first snapshot, applies
incremental insert/delete updates, and searches with tags. The entry point is
the `eanns` program; two helper programs (`test_insert_deletes_consolidate`,
`compute_groundtruth`) are also built to construct the snapshot index and to
compute ground truth.

This directory is additive only: **no file in the parent repository is
modified.** All sources are referenced from `../src`, `../include`, and
`../apps`.

## Directory layout

```
eanns/
├── CMakeLists.txt       # standalone build
├── eanns.cpp            # EANNS program (entry point)
├── run_eanns.sh         # end-to-end reproduction pipeline
└── README.md
```

## Dependencies

- Linux (x86-64), g++ (C++17)
- CMake >= 3.15
- Boost (`libboost-program-options-dev`)
- OpenMP
- Intel MKL (`libmkl-intel-ilp64`, `libmkl-intel-thread`, `libmkl-core`),
  headers under `/usr/include/mkl` by default
- `libiomp5` / LLVM `libomp` (the Intel OpenMP runtime)
- `libtcmalloc` (`libgoogle-perftools-dev`)
- `libnuma` (`libnuma-dev`)

Ubuntu example:

```bash
sudo apt-get install -y build-essential cmake libboost-program-options-dev \
    intel-oneapi-mkl-devel libomp-dev libgoogle-perftools-dev libnuma-dev
```

If MKL or `libiomp5` are installed elsewhere, override with
`-DMKL_INCLUDE_PATH=<dir>` and `-DOMP_PATH=<dir>`.

## Build

```bash
cd eanns
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces three binaries in `build/`:

- `eanns`
- `test_insert_deletes_consolidate`
- `compute_groundtruth`

By default the build targets the current CPU (`-march=native`). For portable
binaries, add `-DPORTABLE=ON`.

> Note: the parent build enables `-Werror`/`-Wfatal-errors`; this standalone
> build omits them for portability across g++ versions. Runtime behavior and
> results are unaffected.

## Data format

All data files are `.fbin` (float binary): a 4-byte `int32` point count,
a 4-byte `int32` dimension, followed by the raw vectors.

## Run

Edit the path/parameter block at the top of `run_eanns.sh`, then:

```bash
bash run_eanns.sh
```

The script performs: build → construct first snapshot index → compute GT for
the snapshot range (`0..pts01`) → construct the full index → compute GT for the
full range (`0..pts02`) → run `eanns` (load snapshot + insert delta + two
searches).

Alternatively, run `eanns` directly:

```bash
./build/eanns \
    --data_type float --dist_fn l2 \
    --data_path /path/to/full_data.fbin \
    --index_path_prefix /path/to/snapshot/index \
    -R 64 -L 200 --alpha 1.2 -T 64 \
    --points_to_skip 0 \
    --max_points_to_insert <pts02> \
    --beginning_index_size 0 \
    --points_per_checkpoint <delta_size> \
    --checkpoints_per_snapshot 0 \
    --points_to_delete_from_beginning 0 \
    --start_deletes_after <pts01> \
    --do_concurrent false \
    --start_point_norm 2 \
    --result_path /path/to/results \
    --query_file /path/to/query.fbin \
    --gt_file_prefix /path/to/gt/gt100_ \
    -K 10 -l 200 \
    --dynamic true --tags 1
```

## Ground-truth naming convention

The `eanns` program expects ground-truth files named

```
{gt_file_prefix}{num_deleted}-{last_point_threshold}
```

- search-1 (after loading the snapshot): `{gt_file_prefix}0-{pts01}`
- search-2 (after inserting the delta):  `{gt_file_prefix}0-{pts02}`
