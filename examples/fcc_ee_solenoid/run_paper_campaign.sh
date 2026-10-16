#!/usr/bin/env bash
# Long paper runs. Run from this directory:
#
#   ./run_paper_campaign.sh lattices      # rebuild the VarSol lattices (004c), done 2026-10-16
#   ./run_paper_campaign.sh track         # DA + MA, quantum radiation, 5000 turns:
#                                         #   SB and VarSol, 2T and 3T, with and
#                                         #   without mid-bend trims, + solenoids off
#   ./run_paper_campaign.sh calibrate     # time 60 turns of emittance + spin
#                                         #   (SB 3T, 1000 particles); ~6 min
#   ./run_paper_campaign.sh emitt N_PART  # emittance + spin in one tracking run,
#                                         #   SB, 10000 turns, 2T and 3T, with and
#                                         #   without trims, N_PART particles
#
# Tracking uses every core (omp_num_threads="auto"), so the cases run one after
# another; start each step with nohup. Logs go to data/logs_paper_campaign/.
# A failing case is reported and skipped.

set -u
cd "$(dirname "$0")"
LOG_DIR=data/logs_paper_campaign
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
calibrate)
    # Figures and data of this short run go to a scratch folder / are removed,
    # so nothing ends up in the paper folder or in data/.
    scratch=$(mktemp -d)
    start=$(date +%s)
    FCC_SOLENOID_PAPER_PLOT_DIR=$scratch MPLBACKEND=Agg \
        run calibrate 018_emittance_and_polarization.py --cases sb_on --b0 3.0 \
        --n-turns 60 --n-part 1000 --pol-fit-start-turn 10 --seed 1 --no-show
    rm -f data/EMIT_*_1000p_60t_* data/POL_*_1000p_60t_*
    rm -rf "$scratch"
    tr '\r' '\n' < "$LOG_DIR/calibrate.log" | grep -E "Tracking: 100%" | tail -1
    echo "total $(( $(date +%s) - start )) s. Reference (PCBE17534, 2026-10-16):"
    echo "  2.3 s/turn -> 6.5 h per case at 1000 particles, scaling linearly."
    ;;
emitt)
    n_part=${2:?usage: $0 emitt N_PART}
    for spec in "3.0 " "3.0 notrims" "2.0 " "2.0 notrims"; do
        set -- $spec
        tag=${1%.0}T${2:+_$2}
        run "emitt_${tag}_${n_part}p" 013_run_da_and_ma.py --b0 "$1" \
            ${2:+--output-tag "$2"} --emitt-only --emitt-cases sb_on \
            --n-part "$n_part" --emitt-n-turns 10000 --emitt-seed 1
    done
    ;;
*)
    echo "usage: $0 {lattices|track|calibrate|emitt N_PART}" >&2
    exit 2
    ;;
esac
