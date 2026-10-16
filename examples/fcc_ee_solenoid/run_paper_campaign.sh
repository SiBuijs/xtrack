#!/usr/bin/env bash
# Long paper runs. Run from this directory, each step with nohup:
#
#   ./run_paper_campaign.sh lattices         # rebuild the VarSol lattices (004c), done 2026-10-16
#   ./run_paper_campaign.sh calibrate        # time 60 turns of emittance + spin
#                                            #   (SB 3T, 1000 particles); ~6 min
#   ./run_paper_campaign.sh emitt N_PART     # emittance + spin in one tracking run,
#                                            #   SB with trims, 3T then 2T, 10000
#                                            #   turns, seed 1. Add 'all' to also run
#                                            #   the two no-trims lattices
#   ./run_paper_campaign.sh da_ma_sb         # DA + MA, quantum radiation, 5000 turns:
#                                            #   SB 3T/2T with and without trims, then
#                                            #   solenoids off (~14 h)
#   ./run_paper_campaign.sh da_ma_varsol     # the same for VarSol (~6.5 h)
#
# Plan (paper/TODO.md, 2026-10-16): emitt 5000 on the remote machine (~2-3 days),
# da_ma_sb and da_ma_varsol here on two nights.
#
# Tracking uses every core (omp_num_threads="auto"), so the cases of a step run
# one after another. Logs go to data/logs_paper_campaign/. A failing case is
# reported and skipped.
#
# Test mode: CAMPAIGN_TEST=1 ./run_paper_campaign.sh <step> [N_PART] runs the
# same commands with 10 turns (DA/MA) or 100 turns (emittance + spin, fit from
# turn 50), logs to data/logs_paper_campaign_test/, puts the figures in a
# temporary folder and removes the test data (*_10t_* / *_100t_* in data/)
# afterwards. Use a small N_PART for emitt, e.g. 10.

set -u
cd "$(dirname "$0")"
LOG_DIR=data/logs_paper_campaign
DA_MA_EXTRA=""
EMITT_TURNS=10000
EMITT_EXTRA=""
if [ "${CAMPAIGN_TEST:-0}" = 1 ]; then
    LOG_DIR=data/logs_paper_campaign_test
    DA_MA_EXTRA="--n-turns 10"
    EMITT_TURNS=100
    EMITT_EXTRA="--emitt-pol-fit-start-turn 50"
    FCC_SOLENOID_PAPER_PLOT_DIR=$(mktemp -d)
    export FCC_SOLENOID_PAPER_PLOT_DIR MPLBACKEND=Agg
    # Only the test's own turn counts: real runs use 5000 / 10000 turns.
    trap 'rm -f data/DA_*_10t_* data/MA_*_10t_* data/EMIT_*_100t_* data/POL_*_100t_*; rm -rf "$FCC_SOLENOID_PAPER_PLOT_DIR"' EXIT
    echo "TEST MODE: short runs, figures in $FCC_SOLENOID_PAPER_PLOT_DIR (removed at the end)"
fi
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

# DA + MA with quantum radiation for one model case (sb_on / varsol_on) at
# 3T and 2T, with and without mid-bend trims.
da_ma_model() {
    local case_name=$1 model=$2
    for b0 in 3.0 2.0; do
        for trims in "" notrims; do
            local tag=${b0%.0}T${trims:+_$trims}
            run "da_ma_quantum_${model}_${tag}" 013_run_da_and_ma.py --b0 "$b0" \
                ${trims:+--output-tag "$trims"} \
                --da-cases "$case_name" --ma-cases "$case_name" \
                --no-emitt --radiation quantum $DA_MA_EXTRA
        done
    done
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
da_ma_sb)
    da_ma_model sb_on sb
    # Solenoids off: same design ring at every field and with or without
    # trims (all correction knobs are gated off), so it runs once.
    run da_ma_quantum_off 013_run_da_and_ma.py --b0 2.0 \
        --da-cases sb_off --ma-cases sb_off --no-emitt --radiation quantum \
        $DA_MA_EXTRA
    ;;
da_ma_varsol)
    da_ma_model varsol_on varsol
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
    n_part=${2:?usage: $0 emitt N_PART [all]}
    specs=("3.0 " "2.0 ")
    if [ "${3:-}" = all ]; then
        specs=("3.0 " "3.0 notrims" "2.0 " "2.0 notrims")
    fi
    for spec in "${specs[@]}"; do
        set -- $spec
        tag=${1%.0}T${2:+_$2}
        run "emitt_${tag}_${n_part}p" 013_run_da_and_ma.py --b0 "$1" \
            ${2:+--output-tag "$2"} --emitt-only --emitt-cases sb_on \
            --n-part "$n_part" --emitt-n-turns "$EMITT_TURNS" --emitt-seed 1 \
            $EMITT_EXTRA
    done
    ;;
*)
    echo "usage: $0 {lattices|calibrate|emitt N_PART [all]|da_ma_sb|da_ma_varsol}" >&2
    exit 2
    ;;
esac
