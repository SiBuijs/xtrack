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
#   ./run_paper_campaign.sh emitt_spin CASE B0 N_PART
#                                            # emittance + spin for one case
#                                            #   (e.g. varsol_on) and field, then
#                                            #   a spin-only run from a bunch at the
#                                            #   equilibrium that run just fitted
#                                            #   (018 --bunch-emitt-source tracked
#                                            #   reads its EMIT npz); 10000 turns,
#                                            #   seed 1, with trims
#   ./run_paper_campaign.sh pol_eq CASE B0 N_PART
#                                            # only the spin-only second half of
#                                            #   emitt_spin, from the case's existing
#                                            #   EMIT npz (same N_PART, 10000 turns)
#   ./run_paper_campaign.sh da_ma_bfe        # DA + MA, quantum radiation, 5000 turns:
#                                            #   BFieldExpansion RK4 (bfe_on) and Boris
#                                            #   (bfeboris_on), 3T and 2T, with trims
#   ./run_paper_campaign.sh bfe_campaign N_PART
#                                            # the 2026-11 remote campaign, in order:
#                                            #   da_ma_bfe; emitt_spin for bfe_on and
#                                            #   bfeboris_on at 3T and 2T; pol_eq for
#                                            #   varsol_on 3T and 2T (spin rerun with
#                                            #   the e5f282c35 spin fix). ~11 days on
#                                            #   the VM at N_PART=5000
#   ./run_paper_campaign.sh da_ma_sb         # DA + MA, quantum radiation, 5000 turns:
#                                            #   SB 3T/2T with and without trims, then
#                                            #   solenoids off (~14 h)
#   ./run_paper_campaign.sh da_ma_varsol     # the same for VarSol (~6.5 h)
#   ./run_paper_campaign.sh da_quantum       # DA only, quantum, on the zoomed
#                                            #   20 x 80 sigma grid (010, 2026-10-21):
#                                            #   SB and VarSol 3T/2T with and without
#                                            #   trims, solenoids off, the 2T field
#                                            #   map; then the field-map MA (quantum)
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
POL_EXTRA=""
# Bunch source of the spin-only runs. Test mode has no EMIT npz at its own turn
# count unless emitt_spin made one just before, so pol_eq uses Twiss there.
POL_EQ_SOURCE=tracked
if [ "${CAMPAIGN_TEST:-0}" = 1 ]; then
    LOG_DIR=data/logs_paper_campaign_test
    DA_MA_EXTRA="--n-turns 10"
    EMITT_TURNS=100
    EMITT_EXTRA="--emitt-pol-fit-start-turn 50"
    POL_EXTRA="--pol-fit-start-turn 50"
    POL_EQ_SOURCE=twiss
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
        return 1
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

# DA only, quantum radiation, for one model case at 3T and 2T, with and
# without trims.
da_model() {
    local case_name=$1 model=$2
    for b0 in 3.0 2.0; do
        for trims in "" notrims; do
            local tag=${b0%.0}T${trims:+_$trims}
            run "da_quantum_${model}_${tag}" 013_run_da_and_ma.py --b0 "$b0" \
                ${trims:+--output-tag "$trims"} \
                --da-only --da-cases "$case_name" --radiation quantum $DA_MA_EXTRA
        done
    done
}

# Spin-only run from a bunch at the tracked equilibrium of this case's EMIT
# npz (the second half of emitt_spin).
pol_eq_case() {
    local case_name=$1 b0=$2 n_part=$3 source=${4:-tracked}
    local name=${case_name%_on}_${b0%.0}T_${n_part}p
    run "pol_eq_${name}" 018_emittance_and_polarization.py \
        --cases "$case_name" --b0 "$b0" --n-turns "$EMITT_TURNS" \
        --n-part "$n_part" --no-emitt --bunch-emitt-divisor 1.0 \
        --bunch-emitt-source "$source" --seed 1 --no-show $POL_EXTRA
}

# Emittance + spin for one case and field, then the spin-only run from the
# equilibrium that run just fitted.
emitt_spin_case() {
    local case_name=$1 b0=$2 n_part=$3
    local name=${case_name%_on}_${b0%.0}T_${n_part}p
    # The spin run needs the EMIT npz of this emittance run, so it only starts
    # if the emittance run succeeded.
    run "emitt_${name}" 013_run_da_and_ma.py --b0 "$b0" --emitt-only \
        --emitt-cases "$case_name" --n-part "$n_part" \
        --emitt-n-turns "$EMITT_TURNS" --emitt-seed 1 $EMITT_EXTRA \
    && pol_eq_case "$case_name" "$b0" "$n_part" tracked
}

# DA + MA with quantum radiation for the two BFieldExpansion cases (RK4 and
# Boris on the same 004c --model bfe lattice), 3T and 2T, with trims (no
# notrims BFE lattice exists).
da_ma_bfe() {
    for case_name in bfe_on bfeboris_on; do
        for b0 in 3.0 2.0; do
            run "da_ma_quantum_${case_name%_on}_${b0%.0}T" 013_run_da_and_ma.py \
                --b0 "$b0" --da-cases "$case_name" --ma-cases "$case_name" \
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
da_quantum)
    da_model sb_on sb
    da_model varsol_on varsol
    run da_quantum_off 013_run_da_and_ma.py --b0 2.0 \
        --da-only --da-cases sb_off --radiation quantum $DA_MA_EXTRA
    # 2 T BNL field map (only 2T, with trims, exists): DA, and its first MA.
    run da_quantum_partialmap_2T 013_run_da_and_ma.py --b0 2.0 \
        --da-only --da-cases partialmap_on --radiation quantum $DA_MA_EXTRA
    run ma_quantum_partialmap_2T 013_run_da_and_ma.py --b0 2.0 \
        --ma-only --ma-cases partialmap_on --radiation quantum $DA_MA_EXTRA
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
emitt_spin)
    usage="usage: $0 emitt_spin CASE B0 N_PART (e.g. varsol_on 3.0 5000)"
    emitt_spin_case "${2:?$usage}" "${3:?$usage}" "${4:?$usage}"
    ;;
pol_eq)
    usage="usage: $0 pol_eq CASE B0 N_PART (e.g. varsol_on 3.0 5000)"
    pol_eq_case "${2:?$usage}" "${3:?$usage}" "${4:?$usage}" "$POL_EQ_SOURCE"
    ;;
da_ma_bfe)
    da_ma_bfe
    ;;
bfe_campaign)
    n_part=${2:?usage: $0 bfe_campaign N_PART}
    da_ma_bfe
    for spec in "bfe_on 3.0" "bfe_on 2.0" "bfeboris_on 3.0" "bfeboris_on 2.0"; do
        set -- $spec
        emitt_spin_case "$1" "$2" "$n_part"
    done
    # VarSol spin with the spin fix (e5f282c35): its emittance runs are
    # unaffected (magnet_spin never touches the orbit), so only the spin-only
    # half is redone, from the existing tracked equilibrium.
    for b0 in 3.0 2.0; do
        pol_eq_case varsol_on "$b0" "$n_part" "$POL_EQ_SOURCE"
    done
    ;;
*)
    echo "usage: $0 {lattices|calibrate|emitt N_PART [all]|emitt_spin CASE B0 N_PART|pol_eq CASE B0 N_PART|da_ma_bfe|bfe_campaign N_PART|da_ma_sb|da_ma_varsol|da_quantum}" >&2
    exit 2
    ;;
esac
