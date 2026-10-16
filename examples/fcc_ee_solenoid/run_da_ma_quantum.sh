#!/usr/bin/env bash
# DA + MA with quantum radiation, 5000 turns (the 009/010 default), for
# SB and VarSol at 2T and 3T, with and without mid-bend trims, plus the
# solenoids-off baseline. Run from this directory:
#
#   ./run_da_ma_quantum.sh lattices   # step 1: rebuild the VarSol lattices (004c)
#   ./run_da_ma_quantum.sh track      # step 2: DA + MA, one case after another
#
# Commit the rebuilt VarSol lattices between the two steps, so the tracking
# runs record a clean commit. Tracking uses every core (omp_num_threads="auto"),
# so the cases run one after another; expect several hours per case.
# Logs go to data/logs_da_ma_quantum/. A failing case is reported and skipped.

set -u
cd "$(dirname "$0")"
LOG_DIR=data/logs_da_ma_quantum
mkdir -p "$LOG_DIR"

run() {
    local name=$1; shift
    echo "[$(date '+%F %T')] start $name: python $*"
    if python "$@" > "$LOG_DIR/$name.log" 2>&1; then
        echo "[$(date '+%F %T')] done  $name"
    else
        echo "[$(date '+%F %T')] FAILED $name (see $LOG_DIR/$name.log)"
    fi
}

case "${1:-}" in
lattices)
    # The VarSol lattices predate the W-matrix coupling targets (e749d4795).
    for b0 in 2.0 3.0; do
        tag=${b0%.0}T
        run "004c_varsol_${tag}" 004c_correct_solenoids_in_fcc_ring.py \
            --model varsol --b0 "$b0"
        run "004c_varsol_${tag}_notrims" 004c_correct_solenoids_in_fcc_ring.py \
            --model varsol --b0 "$b0" --no-midbend-trims \
            --output-tag notrims --input-tag ""
    done
    ;;
track)
    # Solenoids off: same design ring at every field and with or without
    # trims (all correction knobs are gated off), so it runs once.
    run da_ma_quantum_off 013_run_da_and_ma.py --b0 2.0 \
        --da-cases sb_off --ma-cases sb_off --no-emitt --radiation quantum
    for b0 in 2.0 3.0; do
        tag=${b0%.0}T
        run "da_ma_quantum_${tag}" 013_run_da_and_ma.py --b0 "$b0" \
            --da-cases sb_on varsol_on --ma-cases sb_on varsol_on \
            --no-emitt --radiation quantum
        run "da_ma_quantum_${tag}_notrims" 013_run_da_and_ma.py --b0 "$b0" \
            --output-tag notrims \
            --da-cases sb_on varsol_on --ma-cases sb_on varsol_on \
            --no-emitt --radiation quantum
    done
    ;;
*)
    echo "usage: $0 {lattices|track}" >&2
    exit 2
    ;;
esac
