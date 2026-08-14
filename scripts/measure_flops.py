"""One-off FLOPs measurement for E0/E1/E4/E5 at 512x512 (conv-linear hook counter)."""
import torch
import torch.nn as nn


def count_flops(model, x, extra_inputs=None):
    flops = {}

    def conv_hook(module, inp, out):
        batch = out.shape[0]
        out_dims = out.shape[2:]
        kernel_ops = 1
        for k in module.kernel_size:
            kernel_ops *= k
        kernel_ops *= module.in_channels // module.groups
        # MACs -> FLOPs (x2 for mul+add)
        flops_per_instance = kernel_ops * 2
        n = batch * module.out_channels
        for d in out_dims:
            n *= d
        flops[module] = flops_per_instance * n

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
        if extra_inputs is not None:
            model(x, *extra_inputs)
        else:
            model(x)
    for h in hooks:
        h.remove()
    return sum(flops.values())


def main():
    from sipv2.models.model_factory import build_experiment_model
    device = "cpu"
    results = {}
    for exp in ["E0", "E1", "E4", "E5"]:
        model = build_experiment_model(exp).to(device)
        x = torch.randn(1, 3, 512, 512)
        try:
            f = count_flops(model, x)
        except TypeError:
            f = count_flops(model, x, extra_inputs=[x])
        n_params = sum(p.numel() for p in model.parameters())
        results[exp] = (n_params, f / 1e9)
        print(f"{exp}: params={n_params/1e6:.2f}M  FLOPs={f/1e9:.1f} GFLOPs")
    return results


if __name__ == "__main__":
    main()
