#!/usr/bin/env bash
# reproduce.sh - runs the UNet-Starlight scripts with the settings of the paper
#
#   1. simulate.py benchmark      simulated benchmark (only if unet_lenses_2000.npz is missing)
#   2. train_simulations.py       UNet-Starlight, UNet-Starlight-mini, UNet-Starlight-tiny, UNet-Radius-mini
#   3. train_benchmarks.py        1k benchmarks R, S and D, training seeds 1-3
#   4. simulate.py radius-set, train_simulations.py, einstein_radius.py
#                                 Einstein radius of the real lenses (needs modeling_lens_mass.csv)
#   5. paper_results.py           tables of the paper, in results/tables/
#
# Folders: EUCLID_SAGE_DATA_DIR (working folder, default: current folder), EUCLID_SAGE_BENCHMARK_DIR (benchmarks),
# MEASURED_RADII (default: modeling_lens_mass.csv in the working folder).
#
# Usage
#   bash reproduce.sh            all steps
#   bash reproduce.sh --quick    short test run (results in folders ending in _quick)
set -euo pipefail

QUICK=""
if [ "${1:-}" = "--quick" ]; then QUICK="--quick"; fi
SUFFIX="${QUICK:+_quick}"                      # _quick for a quick run, empty otherwise
DATA_FOLDER="${EUCLID_SAGE_DATA_DIR:-.}"
RESULTS="$DATA_FOLDER/results"

BENCHMARK_FOLDER="${EUCLID_SAGE_BENCHMARK_DIR:-$DATA_FOLDER/benchmarks_vis}"
if [ ! -f "$BENCHMARK_FOLDER/EUCLID-SAGE-1k-R_VIS.csv" ]; then
    echo "EUCLID-SAGE benchmarks not found in $BENCHMARK_FOLDER; set EUCLID_SAGE_BENCHMARK_DIR (see README.md)." >&2
    exit 1
fi

if [ -n "$QUICK" ]; then
    LENSES=100; SIMULATED_BENCHMARK="$DATA_FOLDER/unet_lenses_200.npz"
    BENCHMARKS="--benchmarks EUCLID-SAGE-1k-R_VIS"; SEEDS="1 2"
    RADIUS_SET_LENSES=200; RADIUS_SEEDS="1"
else
    LENSES=1000; SIMULATED_BENCHMARK="$DATA_FOLDER/unet_lenses_2000.npz"
    BENCHMARKS=""; SEEDS="1 2 3"
    RADIUS_SET_LENSES=5000; RADIUS_SEEDS="1 2 3"
fi

# 1. Simulated benchmark
if [ ! -f "$SIMULATED_BENCHMARK" ]; then
    python simulate.py benchmark --n-lens $LENSES --n-nonlens $LENSES --out "$SIMULATED_BENCHMARK"
fi

# 2. Training on the simulated benchmark
for MODEL in starlight starlight-mini starlight-tiny radius-mini; do
    if [ ! -f "$RESULTS/simulations/$MODEL$SUFFIX/summary.json" ]; then
        python train_simulations.py --data "$SIMULATED_BENCHMARK" --model $MODEL $QUICK
    fi
done

# 3. Lens classification on the 1k benchmarks
for MODEL in starlight starlight-mini starlight-tiny; do
    python train_benchmarks.py --model $MODEL --seeds $SEEDS --save-weights $BENCHMARKS $QUICK
done

# 4. Einstein radius of the real lenses
MEASURED_RADII="${MEASURED_RADII:-$DATA_FOLDER/modeling_lens_mass.csv}"
if [ -f "$MEASURED_RADII" ]; then
    RADIUS_SET="$DATA_FOLDER/radius_training_$((2 * RADIUS_SET_LENSES)).npz"
    SECOND_SET_NETWORK="$RESULTS/simulations/radius-mini-10k$SUFFIX"
    if [ ! -f "$SECOND_SET_NETWORK/summary.json" ]; then
        if [ ! -f "$RADIUS_SET" ]; then
            python simulate.py radius-set --n-lens $RADIUS_SET_LENSES --n-nonlens $RADIUS_SET_LENSES --out "$RADIUS_SET"
        fi
        # --workers 4 as in the run of the paper
        python train_simulations.py --data "$RADIUS_SET" --model radius-mini --out "$SECOND_SET_NETWORK" \
            --workers 4 $QUICK
    fi
    python einstein_radius.py --measured "$MEASURED_RADII" --seeds $RADIUS_SEEDS \
        --networks "radius-mini=$RESULTS/simulations/radius-mini$SUFFIX/best_unet.pt" \
                   "radius-mini-10k=$SECOND_SET_NETWORK/best_unet.pt" \
        --out-name einstein_radius_real_10k $QUICK
else
    echo "Step 4 skipped: $MEASURED_RADII not found (modeling_lens_mass.csv, Zenodo record 15025832)." >&2
fi

# 5. Tables of the paper
python paper_results.py $QUICK
