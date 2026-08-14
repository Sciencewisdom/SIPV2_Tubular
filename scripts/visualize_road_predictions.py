#!/usr/bin/env python3
"""
Generate side-by-side prediction visualizations for real road experiments.
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from sipv2.models import build_model
from sipv2.datasets.mass_roads import MassachusettsRoadsDataset
from sipv2.utils import set_seed
from torch.utils.data import DataLoader


def visualize_predictions(checkpoints, cases_to_show=4, split="test", output_path="outputs/road_predictions_comparison.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    dataset = MassachusettsRoadsDataset(
        root_dir="/root/autodl-tmp/sipnet/SIPNet_code_for_local/SIPV2_Tubular/data/raw/mass_roads",
        split=split,
        crop_size=512,
        augment=False,
        normalize=True,
    )
    # Force deterministic center crop
    orig_crop = dataset._random_crop
    def _force_det_crop(img, mask, deterministic=False):
        return orig_crop(img, mask, deterministic=True)
    dataset._random_crop = _force_det_crop
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    models = {}
    for name, ckpt_path in checkpoints.items():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        block_type = ckpt.get("config", {}).get("block_type", "dw" if "dw" in name.lower() else "sipv2_road")
        
        model = build_model(
            block_type=block_type,
            in_channels=3,
            num_classes=1,
            channels=[32, 64, 128, 256],
            blocks_per_stage=[2, 2, 2, 2],
            decoder_blocks=1,
            directions=16,
            use_confidence_gate=True,
        )
        model = model.to(device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        model.eval()
        models[name] = model
    
    # Collect predictions for selected cases
    results = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= cases_to_show:
                break
            img = batch["image"].to(device)
            mask = batch["mask"].numpy()[0, 0]
            img_np = batch["image"].numpy()[0].transpose(1, 2, 0)
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            preds = {}
            for name, m in models.items():
                if name.lower().startswith("r1") or name.lower().startswith("r2") or "sipv2" in name.lower():
                    out = m(img, image=img)
                else:
                    out = m(img)
                preds[name] = torch.sigmoid(out).cpu().numpy()[0, 0]
            results.append({"image": img_np, "mask": mask, "preds": preds, "id": batch["image_id"][0]})
    
    # Plot
    n_cases = len(results)
    n_models = len(checkpoints)
    fig = plt.figure(figsize=(3 * (n_models + 2), 3 * n_cases))
    gs = GridSpec(n_cases, n_models + 2, figure=fig)
    
    for i, res in enumerate(results):
        # Image
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(res["image"])
        ax.set_title(f"Image: {res['id']}" if i == 0 else res["id"], fontsize=8)
        ax.axis("off")
        
        # GT
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(res["mask"], cmap="gray")
        ax.set_title("GT" if i == 0 else "", fontsize=8)
        ax.axis("off")
        
        # Predictions
        for j, name in enumerate(checkpoints.keys()):
            ax = fig.add_subplot(gs[i, j + 2])
            ax.imshow(res["preds"][name], cmap="jet", vmin=0, vmax=1)
            ax.set_title(name if i == 0 else "", fontsize=8)
            ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {output_path}")


def main():
    checkpoints = {
        "R0 (DW)": "outputs/road_real_r0_seed42/road_dw_crop512_bs8_ep50_seed42/checkpoints/checkpoint_final.pth",
        "R1 (SIP-v2)": "outputs/road_real_r1_seed42/road_sipv2_road_crop512_bs8_ep50_seed42/checkpoints/checkpoint_final.pth",
        "R2 (+clDice)": "outputs/road_real_r2_seed42/road_sipv2_road_crop512_bs8_ep50_seed42_cldice0.3/checkpoints/checkpoint_final.pth",
    }
    visualize_predictions(checkpoints, cases_to_show=4, split="test", output_path="outputs/paper_figures/road_predictions_comparison.png")


if __name__ == "__main__":
    main()
