"""Evaluate the best checkpoint with the paper's protocol (2 temporal clips x 3 spatial
crops, softmax summed) and write the original-vs-reproduced comparison.

Usage:
    python src/evaluate.py --config configs/slowfast_4x16_r50_thetis.yaml
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader

from dataset import ThetisClips
from model import build_model
from utils import CLASSES, PRETTY, load_config, save_json

# Hovad et al. (2024), Table 2 (SlowFast 4x16, R50) and Table 3 (error sources).
PAPER = {
    "train_acc": 80.39, "val_acc": 70.20, "test_acc": 73.96,
    "err_serve_confusion": 44.4, "err_slice_volley": 20.4,
    "err_smash_serve": 16.7, "err_beginner": 9.3, "err_other": 14.8,
}
SERVES = {"flat_service", "kick_service", "slice_service"}
SLICES = {"backhand_slice", "forehand_slice"}
VOLLEYS = {"backhand_volley", "forehand_volley"}


def error_categories(y_true, y_pred):
    """Share (%) of misclassified test videos in each of the paper's error groups.
    Groups can overlap in the paper; ours are disjoint by construction except 'other'."""
    errs = [(CLASSES[t], CLASSES[p]) for t, p in zip(y_true, y_pred) if t != p]
    n = max(len(errs), 1)
    serve = sum(t in SERVES and p in SERVES for t, p in errs)
    sv = sum((t in SLICES and p in VOLLEYS) or (t in VOLLEYS and p in SLICES) for t, p in errs)
    ss = sum((t == "smash" and p in SERVES) or (p == "smash" and t in SERVES) for t, p in errs)
    other = len(errs) - serve - sv - ss
    return {"n_errors": len(errs),
            "err_serve_confusion": 100 * serve / n, "err_slice_volley": 100 * sv / n,
            "err_smash_serve": 100 * ss / n, "err_other": 100 * other / n}


@torch.no_grad()
def multiview_predict(model, ds, device, bs, nw, amp):
    model.eval()
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    n_vid = len(ds.df)
    probs = torch.zeros(n_vid, len(CLASSES))
    single = torch.zeros(n_vid, len(CLASSES))  # first clip, centre crop only
    for i0, (inputs, _, vid) in zip(range(0, len(ds), bs), loader):
        inputs = [x.to(device, non_blocking=True) for x in inputs]
        with torch.autocast(device.type, enabled=amp):
            p = model(inputs).float().softmax(1).cpu()
        probs.index_add_(0, vid, p)
        for j in range(len(vid)):
            _, clip, crop = ds.items[i0 + j]
            if clip == 0 and crop == (1 if ds.num_crops == 3 else 0):
                single[vid[j]] = p[j]
    y = torch.tensor(ds.df.label_id.values)
    return probs, single, y


def precise_bn(model, cfg, device, amp):
    from fvcore.nn.precise_bn import update_bn_stats
    ds = ThetisClips(cfg["data"]["split_csv"], "train", cfg, seed=cfg["seed"] + 7)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                        num_workers=cfg["data"]["num_workers"], drop_last=True)
    n = min(cfg["train"]["precise_bn_batches"], len(loader))

    def gen():
        for inputs, _, _ in loader:
            yield [x.to(device) for x in inputs]
    model.train()
    update_bn_stats(model, gen(), num_iters=n)
    print(f"Precise BN: recomputed BatchNorm statistics over {n} training batches")


def plot_confusion(cm, path, title):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    labels = [PRETTY[c] for c in CLASSES]
    ax.set_xticks(range(12)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(12)); ax.set_yticklabels(labels)
    for i in range(12):
        for j in range(12):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def plot_per_class(acc, path):
    plt.figure(figsize=(9, 4))
    order = np.argsort(acc)
    plt.barh([PRETTY[CLASSES[i]] for i in order], 100 * acc[order], color="#4C72B0")
    plt.axvline(100 * acc.mean(), color="red", ls="--", label="mean")
    plt.xlabel("Test accuracy (%)"); plt.xlim(0, 100); plt.legend()
    plt.title("Per-class test accuracy (2 clips x 3 crops)")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.opts)
    out = cfg["train"]["output_dir"]
    fig_dir = os.path.join(out, "figures"); os.makedirs(fig_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg["train"]["amp"] and device.type == "cuda"

    ck = torch.load(args.ckpt or os.path.join(out, "best.pt"), map_location="cpu", weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    print(f"Loaded checkpoint from epoch {ck['epoch']} (val acc {100*ck['val_acc']:.2f}%)")
    if cfg["train"].get("precise_bn"):
        precise_bn(model, cfg, device, amp)

    t = cfg["test"]
    results = {"checkpoint_epoch": ck["epoch"]}
    for split in ("test", "val"):
        ds = ThetisClips(cfg["data"]["split_csv"], split, cfg, mode="test",
                         num_clips=t["num_clips"], num_crops=t["num_crops"])
        probs, single, y = multiview_predict(model, ds, device, t["batch_size"],
                                             cfg["data"]["num_workers"], amp)
        pred = probs.argmax(1).numpy(); y = y.numpy()
        results[f"{split}_acc_multiview"] = 100 * accuracy_score(y, pred)
        results[f"{split}_acc_single_view"] = 100 * accuracy_score(y, single.argmax(1).numpy())
        if split == "test":
            p, r, f1, _ = precision_recall_fscore_support(
                y, pred, labels=range(12), average="macro", zero_division=0)
            results.update({"test_macro_precision": 100 * p, "test_macro_recall": 100 * r,
                            "test_macro_f1": 100 * f1})
            cm = confusion_matrix(y, pred, labels=range(12))
            per_class = cm.diagonal() / cm.sum(1)
            results["test_per_class_acc"] = {c: round(100 * a, 2) for c, a in zip(CLASSES, per_class)}
            results["test_per_class_acc_range"] = [round(100 * per_class.min(), 1),
                                                   round(100 * per_class.max(), 1)]
            results.update(error_categories(y, pred))
            pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(os.path.join(out, "confusion_matrix.csv"))
            with open(os.path.join(out, "classification_report.txt"), "w") as f:
                f.write(classification_report(y, pred, labels=range(12), target_names=CLASSES,
                                              digits=4, zero_division=0))
            plot_confusion(cm, os.path.join(fig_dir, "confusion_matrix.png"),
                           f"Test confusion matrix (acc {results['test_acc_multiview']:.2f}%)")
            plot_per_class(per_class, os.path.join(fig_dir, "per_class_accuracy.png"))
            pd.DataFrame({"video_id": ds.df.video_id, "true": [CLASSES[i] for i in y],
                          "pred": [CLASSES[i] for i in pred],
                          "confidence": probs.max(1).values.numpy() / (t["num_clips"] * t["num_crops"])}
                         ).to_csv(os.path.join(out, "test_predictions.csv"), index=False)

    hist = pd.read_csv(os.path.join(out, "history.csv"))
    best_row = hist[hist.epoch == ck["epoch"]].iloc[0]
    results["train_acc_at_best_epoch"] = 100 * best_row.train_acc
    results["val_acc_at_best_epoch"] = 100 * best_row.val_acc
    save_json(results, os.path.join(out, "results.json"))

    rows = [
        ("Train accuracy (%)", PAPER["train_acc"], results["train_acc_at_best_epoch"]),
        ("Validation accuracy (%)", PAPER["val_acc"], results["val_acc_at_best_epoch"]),
        ("Test accuracy, 2x3 views (%)", PAPER["test_acc"], results["test_acc_multiview"]),
        ("Test macro F1 (%)", None, results["test_macro_f1"]),
        ("Errors: serve vs serve (%)", PAPER["err_serve_confusion"], results["err_serve_confusion"]),
        ("Errors: slice vs volley (%)", PAPER["err_slice_volley"], results["err_slice_volley"]),
        ("Errors: smash vs serve (%)", PAPER["err_smash_serve"], results["err_smash_serve"]),
    ]
    lines = ["| Evaluation metric | Original paper | Reproduced model | Difference |",
             "|---|---|---|---|"]
    for name, o, r in rows:
        lines.append(f"| {name} | {'-' if o is None else f'{o:.2f}'} | {r:.2f} | "
                     f"{'-' if o is None else f'{r - o:+.2f}'} |")
    table = "\n".join(lines)
    with open(os.path.join(out, "comparison_table.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
