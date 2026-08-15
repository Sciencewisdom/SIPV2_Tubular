"""Efficiency measurement with the diffusion branch ACTIVE (post-56c5d56 audit).

Background: the historical complexity table (E1 == E4 at 49.1 GFLOPs / 25.9 ms)
was measured via model(x) without image=, which skips the diffusion branch
entirely — so the "E4 adds <0.1 ms" claim described a model whose signature
branch never ran. This script re-measures params / conv+linear GFLOPs /
CUDA latency / peak VRAM with image passed for all sipv2-family blocks.

Usage: .venv_torch/Scripts/python scripts/measure_efficiency.py
Writes: outputs/fixed_efficiency.json
"""
import json
import torch
import torch.nn as nn


def count_flops(model, x):
    flops = {}

    def conv_hook(module, inp, out):
        kernel_ops = 1
        for k in module.kernel_size:
            kernel_ops *= k
        kernel_ops *= module.in_channels // module.groups
        n = out.shape[0] * module.out_channels
        for d in out.shape[2:]:
            n *= d
        flops[module] = kernel_ops * 2 * n

    def linear_hook(module, inp, out):
        flops[module] = module.in_features * module.out_features * 2

    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    model.eval()
    with torch.no_grad():
        model(x, image=x)
    for h in hooks:
        h.remove()
    return sum(flops.values())


def measure(exp, device, warmup=10, reps=50):
    from sipv2.models.model_factory import build_experiment_model
    model = build_experiment_model(exp).to(device).eval()
    x = torch.randn(1, 3, 512, 512, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    gflops = count_flops(model, x) / 1e9

    latency_ms = None
    peak_mb = None
    if device == 'cuda':
        with torch.no_grad():
            for _ in range(warmup):
                model(x, image=x)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(reps):
                model(x, image=x)
            end.record()
            torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / reps
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return {
        'params': n_params,
        'params_M': round(n_params / 1e6, 3),
        'gflops_conv_linear': round(gflops, 1),
        'latency_ms_cuda_b1_512': None if latency_ms is None else round(latency_ms, 2),
        'peak_vram_MB': None if peak_mb is None else round(peak_mb, 1),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    for exp in ['E0', 'E1', 'E1D', 'E4', 'E5']:
        results[exp] = measure(exp, device)
        print(exp, results[exp])
    results['_protocol'] = {
        'device': device,
        'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
        'input': '1x3x512x512, batch 1, image=x passed (diffusion ACTIVE)',
        'latency': 'CUDA events, 10 warmup + 50 reps',
        'gflops': 'hook counter over Conv2d/ConvTranspose2d/Linear only '
                  '(diffusion stencil is element-wise, reflected in latency)',
    }
    with open('outputs/fixed_efficiency.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('saved outputs/fixed_efficiency.json')


if __name__ == '__main__':
    main()
