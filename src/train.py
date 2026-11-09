"""Train SlowFast 4x16 R50 from scratch on THETIS with PySlowFast's recipe.

Usage:
    python src/train.py --config configs/slowfast_4x16_r50_thetis.yaml
    python src/train.py --config ... --opts train.batch_size=8 solver.max_epoch=50

Resumes automatically from <output_dir>/last.pt. --time-limit-hours stops cleanly
before a Kaggle/Colab session is killed, so the next session can continue.
"""
import argparse
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ThetisClips
from model import build_model, summarise
from utils import environment_info, load_config, save_json, seed_everything


def lr_at_epoch(cur_epoch, s, scale):
    """PySlowFast solver/lr_policy.py: linear warmup into half-cosine (end lr 0)."""
    base = s["base_lr"] * scale
    def cosine(e):
        return base * (math.cos(math.pi * e / s["max_epoch"]) + 1.0) * 0.5
    lr = cosine(cur_epoch)
    if cur_epoch < s["warmup_epochs"]:
        start = s["warmup_start_lr"] * scale
        end = cosine(s["warmup_epochs"])
        lr = start + (end - start) * cur_epoch / s["warmup_epochs"]
    return lr


def build_optimizer(model, s, scale):
    """SGD; BatchNorm params get no weight decay (PySlowFast BN.WEIGHT_DECAY = 0)."""
    bn, rest = [], []
    for m in model.modules():
        for p in m.parameters(recurse=False):
            if not p.requires_grad:
                continue
            (bn if isinstance(m, nn.modules.batchnorm._NormBase) else rest).append(p)
    groups = [{"params": rest, "weight_decay": s["weight_decay"]},
              {"params": bn, "weight_decay": 0.0}]
    return torch.optim.SGD(groups, lr=s["base_lr"] * scale, momentum=s["momentum"],
                           nesterov=s["nesterov"], dampening=0.0)


@torch.no_grad()
def evaluate_loss_acc(model, loader, device, amp):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    tot_loss, correct, n = 0.0, 0, 0
    for inputs, y, _ in loader:
        inputs = [x.to(device, non_blocking=True) for x in inputs]
        y = y.to(device)
        with torch.autocast(device.type, enabled=amp and device.type == "cuda"):
            out = model(inputs)
        tot_loss += crit(out.float(), y).item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()
    return tot_loss / n, correct / n


def plot_curves(hist, out_dir):
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(h.epoch, h.train_loss, label="train")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set_title("Cross-entropy loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h.epoch, 100 * h.train_acc, label="train (augmented)")
    ax[1].plot(h.epoch, 100 * h.val_acc, label="val")
    best = h.val_acc.idxmax()
    ax[1].scatter(h.epoch[best], 100 * h.val_acc[best], color="red", zorder=5,
                  label=f"best val {100*h.val_acc[best]:.2f}% @ {int(h.epoch[best])}")
    ax[1].set_title("Top-1 accuracy (%)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    ax[2].plot(h.epoch, h.lr)
    ax[2].set_title("Learning rate"); ax[2].set_xlabel("epoch")
    plt.tight_layout()
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    plt.savefig(os.path.join(out_dir, "figures", "training_curves.png"), dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--time-limit-hours", type=float, default=11.0)
    args = ap.parse_args()
    cfg = load_config(args.config, args.opts)
    out = cfg["train"]["output_dir"]
    os.makedirs(out, exist_ok=True)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    t_start = time.time()

    save_json(cfg, os.path.join(out, "config_used.json"))
    save_json(environment_info(), os.path.join(out, "environment.json"))
    if not os.path.exists(os.path.join(out, "model_summary.json")):
        summarise(cfg, os.path.join(out, "model_summary.json"))

    tr_cfg, s = cfg["train"], cfg["solver"]
    bs = tr_cfg["batch_size"]
    scale = bs / s["reference_batch_size"]  # linear LR scaling rule
    amp = tr_cfg["amp"] and device.type == "cuda"
    nw = cfg["data"]["num_workers"]

    train_ds = ThetisClips(cfg["data"]["split_csv"], "train", cfg, seed=cfg["seed"])
    val_ds = ThetisClips(cfg["data"]["split_csv"], "val", cfg, mode="train", seed=cfg["seed"] + 1)
    print(f"train clips {len(train_ds)}, val clips {len(val_ds)}, batch {bs}, lr scale {scale:.4f}")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    model = build_model(cfg).to(device)
    opt = build_optimizer(model, s, scale)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = nn.CrossEntropyLoss()

    start_epoch, hist, best_acc, best_epoch, bad_epochs = 0, [], -1.0, -1, 0
    last_path = os.path.join(out, "last.pt")
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, hist = ck["epoch"] + 1, ck["hist"]
        best_acc, best_epoch, bad_epochs = ck["best_acc"], ck["best_epoch"], ck["bad_epochs"]
        print(f"Resumed from epoch {start_epoch}, best val {best_acc:.4f} @ {best_epoch}")

    for epoch in range(start_epoch, s["max_epoch"]):
        ep_t = time.time()
        model.train()
        train_ds.set_epoch(epoch)
        g = torch.Generator(); g.manual_seed(cfg["seed"] * 1000 + epoch)
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g, num_workers=nw,
                            pin_memory=True, drop_last=True)
        n_it = len(loader)
        run_loss, run_correct, run_n = 0.0, 0, 0
        for it, (inputs, y, _) in enumerate(loader):
            lr = lr_at_epoch(epoch + it / n_it, s, scale)
            for gr in opt.param_groups:
                gr["lr"] = lr
            inputs = [x.to(device, non_blocking=True) for x in inputs]
            y = y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=amp):
                out_ = model(inputs)
                loss = crit(out_, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} iter {it}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * y.numel()
            run_correct += (out_.argmax(1) == y).sum().item()
            run_n += y.numel()

        row = {"epoch": epoch + 1, "lr": lr, "train_loss": run_loss / run_n,
               "train_acc": run_correct / run_n}
        if (epoch + 1) % tr_cfg["eval_period"] == 0 or epoch + 1 == s["max_epoch"]:
            val_ds.set_epoch(0)  # same random val views every epoch -> comparable curve
            row["val_loss"], row["val_acc"] = evaluate_loss_acc(model, val_loader, device, amp)
        row["epoch_minutes"] = (time.time() - ep_t) / 60
        hist.append(row)

        improved = row.get("val_acc", -1) > best_acc
        if improved:
            best_acc, best_epoch, bad_epochs = row["val_acc"], epoch + 1, 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_acc": best_acc,
                        "cfg": cfg}, os.path.join(out, "best.pt"))
        elif "val_acc" in row:
            bad_epochs += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "hist": hist,
                    "best_acc": best_acc, "best_epoch": best_epoch, "bad_epochs": bad_epochs},
                   last_path)
        pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
        plot_curves(hist, out)
        print(f"ep {epoch+1:3d}/{s['max_epoch']} lr {lr:.5f} | train loss {row['train_loss']:.3f} "
              f"acc {100*row['train_acc']:.2f} | val loss {row.get('val_loss', float('nan')):.3f} "
              f"acc {100*row.get('val_acc', float('nan')):.2f} | best {100*best_acc:.2f}@{best_epoch} "
              f"| {row['epoch_minutes']:.1f} min", flush=True)

        pat = tr_cfg.get("early_stop_patience", 0)
        if pat and bad_epochs >= pat:
            print(f"Early stopping: no val improvement for {pat} epochs.")
            break
        elapsed_h = (time.time() - t_start) / 3600
        if elapsed_h + 1.5 * row["epoch_minutes"] / 60 > args.time_limit_hours:
            print(f"Time limit reached after {elapsed_h:.2f} h. Re-run the same command to resume.")
            break

    total_h = sum(r["epoch_minutes"] for r in hist) / 60
    save_json({"best_val_acc": best_acc, "best_epoch": best_epoch,
               "epochs_run": len(hist), "train_hours": round(total_h, 2),
               "batch_size": bs, "scaled_base_lr": s["base_lr"] * scale},
              os.path.join(out, "train_summary.json"))
    print(f"Done. best val acc {100*best_acc:.2f}% at epoch {best_epoch}; {total_h:.2f} GPU-hours")


if __name__ == "__main__":
    main()
