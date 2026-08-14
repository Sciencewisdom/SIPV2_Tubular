#!/usr/bin/env python3
"""B1 analysis: does the loss--architecture interaction generalize to a second
topology loss (skeleton recall)?

Collects summary.json from:
  - lambda=0 baselines: outputs/E1_size512_bs2_seed42, outputs/E4_size512_bs2_seed42
  - skelrec sweep:      outputs/b1_{E1,E4}_skelrec{0.1,0.3}_seed42

Prints a table (best Dice, clDice, SkelRec, break count) with deltas vs the
matched lambda=0 baseline, saves outputs/b1_skelrec_table.json and a figure
paper_figures/b1_skelrec_trajectory.png.

Note: skel_break_count in summary.json uses the validation-split protocol of
train.py; the paper's headline break counts come from the separate 4-case
recount script (scripts/recompute_drive_breaks.py) -- rerun that on these
checkpoints before quoting break numbers in the paper.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS = {
    ('E1', 0.0): 'outputs/E1_size512_bs2_seed42',
    ('E4', 0.0): 'outputs/E4_size512_bs2_seed42',
    ('E1', 0.1): 'outputs/b1_E1_skelrec0.1_seed42/E1_size512_bs2_seed42',
    ('E1', 0.3): 'outputs/b1_E1_skelrec0.3_seed42/E1_size512_bs2_seed42',
    ('E4', 0.1): 'outputs/b1_E4_skelrec0.1_seed42/E4_size512_bs2_seed42',
    ('E4', 0.3): 'outputs/b1_E4_skelrec0.3_seed42/E4_size512_bs2_seed42',
}
METRICS = [('best_dice', 'Best Dice'), ('skel_cldice', 'clDice'),
           ('skel_skeleton_recall', 'SkelRec'), ('skel_break_count', 'Breaks')]


def load(path):
    s = json.load(open(os.path.join(path, 'summary.json')))
    fm = s['final_metrics']
    return {k: fm.get(k, s.get(k)) for k, _ in METRICS}


def main():
    table = {}
    for key, path in RUNS.items():
        if os.path.exists(os.path.join(path, 'summary.json')):
            table[key] = load(path)
        else:
            print(f'MISSING {key}: {path}')

    print(f'\n{"exp":<5}{"lambda":<8}{"Dice":<9}{"clDice":<9}{"SkelRec":<9}{"Breaks":<8}')
    for (exp, lam), m in sorted(table.items()):
        print(f'{exp:<5}{lam:<8}{m["best_dice"]:<9.4f}{m["skel_cldice"]:<9.4f}'
              f'{m["skel_skeleton_recall"]:<9.4f}{m["skel_break_count"]:<8.0f}')

    print('\nDeltas vs lambda=0 (same architecture):')
    for exp in ('E1', 'E4'):
        if (exp, 0.0) not in table:
            continue
        base = table[(exp, 0.0)]
        for lam in (0.1, 0.3):
            if (exp, lam) not in table:
                continue
            m = table[(exp, lam)]
            d = {k: m[k] - base[k] for k, _ in METRICS}
            print(f'{exp} lambda={lam}: dDice={d["best_dice"]:+.4f} dclDice={d["skel_cldice"]:+.4f} '
                  f'dSkelRec={d["skel_skeleton_recall"]:+.4f} dBreaks={d["skel_break_count"]:+.0f}')

    out = {f'{e}_l{l}': m for (e, l), m in table.items()}
    with open('outputs/b1_skelrec_table.json', 'w') as f:
        json.dump(out, f, indent=2)

    # figure: per-metric trajectory vs lambda, one line per architecture
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (k, label) in zip(axes, METRICS):
        for exp, color in (('E1', 'tab:blue'), ('E4', 'tab:red')):
            xs = [l for (e, l) in sorted(table) if e == exp]
            ys = [table[(exp, l)][k] for l in xs]
            if xs:
                ax.plot(xs, ys, 'o-', color=color,
                        label='DW-CNN (E1)' if exp == 'E1' else 'SIP-v2 Min (E4)')
        ax.set_xlabel(r'skel-recall weight $\lambda$')
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('B1: skeleton-recall loss sweep on DRIVE (seed 42)', fontsize=11)
    plt.tight_layout()
    os.makedirs('paper_figures', exist_ok=True)
    plt.savefig('paper_figures/b1_skelrec_trajectory.png', dpi=300)
    print('saved outputs/b1_skelrec_table.json and paper_figures/b1_skelrec_trajectory.png')


if __name__ == '__main__':
    main()
