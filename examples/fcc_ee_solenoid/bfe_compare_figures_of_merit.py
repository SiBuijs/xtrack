"""Side-by-side SplineBoris vs BFieldExpansion figures of merit (004n JSONs).

Reads data/figures_of_merit/figures_of_merit_{case}.json for each SB/BFE
pair and prints the corrected-ring ('on') figures and the match residuals.
Exploratory (bfe_setup refactor), not a paper table.

Usage: python bfe_compare_figures_of_merit.py [--sb-tag bfecmp]
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--sb-tag', default='bfecmp',
                    help='004c/004n --output-tag of the SB reference runs.')
args = parser.parse_args()

DIR = Path(__file__).parent / 'data' / 'figures_of_merit'
SB_SUFFIX = f'_{args.sb_tag}' if args.sb_tag else ''
PAIRS = {
    '2T analytic': (f'figures_of_merit_2T{SB_SUFFIX}', 'figures_of_merit_bfe_2T'),
    '3T analytic': (f'figures_of_merit_3T{SB_SUFFIX}', 'figures_of_merit_bfe_3T'),
    '2T BNL map': (f'figures_of_merit_partialmap_2T{SB_SUFFIX}',
                   'figures_of_merit_partialmap_bfe_2T'),
}
# (label, getter, format); Q'' is the true d2Q/ddelta2 (004n q2x/q2y).
ROWS = [
    ('Qx', lambda d: d['on']['qx'], '.8f'),
    ('Qy', lambda d: d['on']['qy'], '.8f'),
    ("Q'x", lambda d: d['on']['dqx'], '.5f'),
    ("Q'y", lambda d: d['on']['dqy'], '.5f'),
    ("Q''x", lambda d: d['on']['q2x'], '.3f'),
    ("Q''y", lambda d: d['on']['q2y'], '.3f'),
    ('|C-|', lambda d: d['on']['c_minus'], '.3e'),
    ('|C-| solenoids on, no corr', lambda d: d['nocorr']['c_minus'], '.4e'),
    ('edge max beta beat', lambda d: d['match']['edge']['beta_beat'], '.2e'),
    ('edge max coupled beta ratio',
     lambda d: d['match']['edge']['coupled_beta_ratio'], '.2e'),
    ('IP max |dy| [m]', lambda d: d['match']['ip']['dy'], '.2e'),
    ('edge max |orbit| [m]', lambda d: d['match']['edge']['orbit'], '.2e'),
    ('ET edge max', lambda d: d['match']['edwards_teng_edge_max'], '.2e'),
    ('arc max coupled beta ratio',
     lambda d: d['match']['arc_coupled_beta_ratio_max'], '.2e'),
    ('max straight leak',
     lambda d: max(d['match']['straight_leak'].values()), '.2e'),
]

for case, (sb_stem, bfe_stem) in PAIRS.items():
    try:
        sb = json.loads((DIR / f'{sb_stem}.json').read_text())
        bfe = json.loads((DIR / f'{bfe_stem}.json').read_text())
    except FileNotFoundError as exc:
        print(f'\n{case}: missing {exc.filename}')
        continue
    print(f'\n=== {case}: {sb["lattice"]}  vs  {bfe["lattice"]}')
    print(f'  {"figure":30s} {"SB":>20s} {"BFE":>20s} {"BFE - SB":>12s}')
    for label, get, fmt in ROWS:
        try:
            a, b = get(sb), get(bfe)
        except KeyError:
            print(f'  {label:30s} (missing)')
            continue
        print(f'  {label:30s} {a:>20{fmt}} {b:>20{fmt}} {b - a:>+12.2e}')
