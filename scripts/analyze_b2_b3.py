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
    print('\n=== B3: road component ablation ===')
    KEYS = ['dice', 'cldice', 'skel_recall', 'apls', 'gap_recovery']

    def agg_from_cases(cases):
        agg = {}
        for k in KEYS:
            v = np.array([c[k] for c in cases if c.get(k) is not None], dtype=float)
            agg[k] = {'mean': float(v.mean()), 'std': float(v.std()), 'n': len(v)}
        return agg

    # Preferred: fixed-code retrain evals (uniform protocol; audit section 9).
    fixed_jsons = {a: f'outputs/fixed_eval_{a}.json'
                   for a in ['r1', 'r2', 'b3_scharr', 'b3_sobel', 'b3_stencil3', 'b3_nogate']}
    if any(os.path.exists(p) for p in fixed_jsons.values()):
        out = {}
        for name, path in fixed_jsons.items():
            if not os.path.exists(path):
                continue
            d = json.load(open(path))
            entry = {'valid': agg_from_cases(d['cases'])}
            print(name, 'valid:', {k: f"{v['mean']:.4f}+-{v['std']:.4f}" for k, v in entry['valid'].items()})
            if 'test_at_valid_threshold' in d:
                entry['test_fixed_thr'] = agg_from_cases(d['test_at_valid_threshold']['cases'])
                print(name, 'test@valid-thr:', {k: f"{v['mean']:.4f}+-{v['std']:.4f}" for k, v in entry['test_fixed_thr'].items()})
            out[name] = entry
        return out

    # DEPRECATED fallback: pre-fix evals (diffusion disabled; not comparable
    # across protocols — see audit section 9). Kept for archaeology only.
    jsons = {a: f'outputs/b3_eval_{a}.json' for a in ['sobel', 'stencil3', 'nogate', 'scharr_bs4']}
    if any(os.path.exists(p) for p in jsons.values()):
        print('WARNING: using deprecated pre-fix eval JSONs (diffusion-disabled models)')
        out = {}
        for name, path in jsons.items():
            if not os.path.exists(path):
                continue
            cases = json.load(open(path))['cases']
            out[name] = {'deterministic_deprecated': agg_from_cases(cases)}
            print(name, {k: f"{a['mean']:.4f}+-{a['std']:.4f}" for k, a in out[name]['deterministic_deprecated'].items()})
        return out
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
