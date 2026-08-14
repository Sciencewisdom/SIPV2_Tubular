#!/usr/bin/env python3
"""B2/B3 result analysis.

B2 (DCN control): tabulates outputs/b2_E1D_cldice{0.0,0.1,0.3}_seed42 summaries
against E1/E4 references (clDice sweep numbers already published).

B3 (road component ablation): parses outputs/b3_ablation.log for the final
validation metrics of each ablation run and compares against the R1 seed-42
reference (full SIP-v2 Road) from outputs/road_percase_deterministic.json /
the original R1 training log if available.

Run after the chained sweeps finish. Prints tables; writes
outputs/b2_b3_analysis.json.
"""
import json, os, re
import numpy as np

B2_RUNS = {
    0.0: 'outputs/b2_E1D_cldice0.0_seed42/E1D_size512_bs2_seed42',
    0.1: 'outputs/b2_E1D_cldice0.1_seed42/E1D_size512_bs2_seed42',
    0.3: 'outputs/b2_E1D_cldice0.3_seed42/E1D_size512_bs2_seed42',
}
METRICS = ['best_dice', 'skel_cldice', 'skel_skeleton_recall', 'skel_break_count']


def b2():
    print('=== B2: E1D (DCNv2) clDice sweep ===')
    table = {}
    for lam, path in B2_RUNS.items():
        sp = os.path.join(path, 'summary.json')
        if not os.path.exists(sp):
            print(f'MISSING lambda={lam}: {path}')
            continue
        fm = json.load(open(sp))['final_metrics']
        table[lam] = {k: fm.get(k) for k in METRICS}
        print(f"lambda={lam}: " + '  '.join(f'{k}={fm.get(k):.4f}' for k in METRICS))
    if 0.0 in table:
        b = table[0.0]
        for lam in (0.1, 0.3):
            if lam in table:
                d = {k: table[lam][k] - b[k] for k in METRICS}
                print(f'delta lambda={lam} vs 0: ' + '  '.join(f'{k}={d[k]:+.4f}' for k in METRICS))
    return table


def b3():
    print('\n=== B3: road component ablation (from b3_ablation.log) ===')
    # R1 (full SIP-v2 Road, seed 42) reference from the deterministic per-case eval
    ref_path = 'outputs/road_percase_deterministic.json'
    if os.path.exists(ref_path):
        cases = json.load(open(ref_path))['R1_SIPV2_final']['cases']
        print('R1 reference (seed 42, n=%d):' % len(cases))
        for k in ['dice', 'cldice', 'skel_recall', 'apls', 'gap_recovery']:
            v = np.array([c[k] for c in cases if c.get(k) is not None], dtype=float)
            print(f'  {k}: {v.mean():.4f} +- {v.std():.4f}')
    log = 'outputs/b3_ablation.log'
    if not os.path.exists(log):
        print(f'MISSING {log}')
        return {}
    text = open(log, encoding='utf-8', errors='replace').read()
    sections = re.split(r'=== (B3[abc]: [^=]+) ===', text)
    out = {}
    for i in range(1, len(sections), 2):
        name, body = sections[i].strip(), sections[i + 1]
        dice = re.findall(r'Best Dice: ([\d.]+)', body)
        metric_lines = re.findall(
            r'clDice=([\d.]+), SkelRec=([\d.]+).*?APLS=([\d.]+), Conn=([\d.]+), '
            r'GapRec=([\d.]+).*?JPR=([\d.]+)', body, re.S)
        last = metric_lines[-1] if metric_lines else None
        out[name] = {
            'best_dice': float(dice[-1]) if dice else None,
            'final': dict(zip(['clDice', 'SkelRec', 'APLS', 'Conn', 'GapRec', 'JPR'],
                              map(float, last))) if last else None,
        }
        print(name, out[name])
    return out


def main():
    res = {'B2_E1D': b2(), 'B3_ablation': b3()}
    with open('outputs/b2_b3_analysis.json', 'w') as f:
        json.dump(res, f, indent=2, default=str)
    print('\nsaved outputs/b2_b3_analysis.json')


if __name__ == '__main__':
    main()
