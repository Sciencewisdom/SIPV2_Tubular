"""
Compare road extraction results across experiments.
"""
import json
import os

experiments = {
    'R0 (DW)': 'outputs/road_dw_50ep/road_dw_crop512_bs8_ep50_seed42/summary.json',
    'R1 (SIP-v2 Road)': 'outputs/road_sipv2_50ep/road_sipv2_road_crop512_bs8_ep50_seed42/summary.json',
    'R2 (SIP-v2 Road + clDice)': 'outputs/road_sipv2_cldice_50ep/road_sipv2_road_crop512_bs8_ep50_seed42_cldice0.3/summary.json',
}

print("=" * 80)
print("Road Extraction Results Comparison (Massachusetts Roads, Synthetic)")
print("=" * 80)

results = {}
for name, path in experiments.items():
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = json.load(f)
        metrics = data.get('final_metrics', {})
        results[name] = metrics
    else:
        results[name] = None

# Print table
headers = ['Exp', 'Dice', 'IoU', 'clDice', 'SkelRec', 'APLS', 'Conn', 'GapRec']
print(f"{headers[0]:20s} {headers[1]:>8s} {headers[2]:>8s} {headers[3]:>8s} {headers[4]:>8s} {headers[5]:>8s} {headers[6]:>8s} {headers[7]:>8s}")
print("-" * 80)

for name, metrics in results.items():
    if metrics:
        print(f"{name:20s} {metrics.get('dice', 0):8.4f} {metrics.get('iou', 0):8.4f} "
              f"{metrics.get('skel_cldice', 0):8.4f} {metrics.get('skel_skeleton_recall', 0):8.4f} "
              f"{metrics.get('road_apls', 0):8.4f} {metrics.get('road_connectivity', 0):8.4f} {metrics.get('road_gap_recovery', 0):8.4f}")
    else:
        print(f"{name:20s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s} {'N/A':>8s}")

print("=" * 80)
