"""SlowFast 4x16 ResNet-50 built with PyTorchVideo, configured to match PySlowFast."""
import argparse

import torch
from pytorchvideo.models.slowfast import create_slowfast

from utils import load_config, save_json


def build_model(cfg):
    m = cfg["model"]
    d = cfg["data"]
    t_fast = d["num_frames"]
    t_slow = t_fast // m["alpha"]
    s = d["train_crop_size"] // 32
    model = create_slowfast(
        model_depth=50,
        model_num_class=m["num_classes"],
        slowfast_channel_reduction_ratio=(m["beta_inv"],),
        slowfast_conv_channel_fusion_ratio=m["fusion_conv_channel_ratio"],
        slowfast_fusion_conv_kernel_size=(m["fusion_kernel_size"], 1, 1),
        slowfast_fusion_conv_stride=(m["alpha"], 1, 1),
        dropout_rate=m["dropout"],
        head_pool_kernel_sizes=((t_slow, s, s), (t_fast, s, s)),
        head_activation=None,  # logits; softmax is applied at test time
    )
    if m.get("pretrained_path"):  # hook for Assignment 03 (Kinetics init)
        sd = torch.load(m["pretrained_path"], map_location="cpu")
        sd = sd.get("model_state", sd.get("model", sd))
        sd = {k: v for k, v in sd.items() if "proj" not in k}  # drop 400-way classifier
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Loaded pretrained weights. missing={len(missing)} unexpected={len(unexpected)}")
    return model


def summarise(cfg, out_path=None):
    """Per-stage output shapes for both pathways, parameter count, GFLOPs."""
    model = build_model(cfg).eval()
    d = cfg["data"]
    c = d["train_crop_size"]
    x = [torch.zeros(1, 3, d["num_frames"] // cfg["model"]["alpha"], c, c),
         torch.zeros(1, 3, d["num_frames"], c, c)]
    rows = []

    def hook(name):
        def f(_, __, out):
            if isinstance(out, (list, tuple)):
                rows.append((name, [list(o.shape[1:]) for o in out]))
            else:
                rows.append((name, [list(out.shape[1:])]))
        return f

    names = ["stem (conv1 + pool)", "res2", "res3", "res4", "res5", "pathway pool", "head"]
    for i, blk in enumerate(model.blocks):
        blk.register_forward_hook(hook(names[i] if i < len(names) else f"block{i}"))
    in_shapes = [list(t.shape[1:]) for t in x]
    with torch.no_grad():
        model(list(x))  # PyTorchVideo fuses pathways in place, so pass a copy
    total = sum(p.numel() for p in model.parameters())
    summary = {
        "input_slow_CTHW": in_shapes[0],
        "input_fast_CTHW": in_shapes[1],
        "note": "slow-pathway channels after each stage include the channels fused in from the fast pathway",
        "stages": [{"stage": n, "output_shapes_CTHW": s} for n, s in rows],
        "params_million": round(total / 1e6, 2),
    }
    try:
        from fvcore.nn import FlopCountAnalysis
        fl = FlopCountAnalysis(model, ([t.clone() for t in x],))
        fl.unsupported_ops_warnings(False)
        fl.uncalled_modules_warnings(False)
        summary["gflops_per_view"] = round(fl.total() / 1e9, 1)
    except Exception as e:
        summary["gflops_per_view"] = f"n/a ({type(e).__name__})"
    if out_path:
        save_json(summary, out_path)
        with open(out_path.replace(".json", ".txt"), "w") as f:
            f.write(str(model))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="model_summary.json")
    args = ap.parse_args()
    import json
    print(json.dumps(summarise(load_config(args.config), args.out), indent=2))
