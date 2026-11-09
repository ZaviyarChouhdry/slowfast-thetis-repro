"""Index THETIS RGB videos, cache decoded frames as JPEG, build the 70/20/10 split,
and produce the dataset figures for the report.

Usage:
    python src/prepare_thetis.py --config configs/slowfast_4x16_r50_thetis.yaml
"""
import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from utils import CLASSES, PRETTY, load_config, normalise_class_name, save_json

VIDEO_EXT = (".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg")
NON_RGB = ("depth", "mask", "silhouette", "skelet")


def find_videos(root):
    rows = []
    for dirpath, _, files in os.walk(root):
        low = dirpath.lower()
        # THETIS ships depth / mask / skeleton videos next to RGB; keep RGB only.
        parts = re.split(r"[\\/]", low)
        if any(tag in p for p in parts for tag in NON_RGB if "rgb" not in p):
            continue
        for fn in files:
            if not fn.lower().endswith(VIDEO_EXT):
                continue
            path = os.path.join(dirpath, fn)
            label = None
            for p in reversed(re.split(r"[\\/]", dirpath)):
                label = normalise_class_name(p)
                if label:
                    break
            if label is None:
                continue
            m = re.search(r"p(\d+)", os.path.splitext(fn)[0].lower())
            rows.append({
                "video_path": path,
                "label": label,
                "label_id": CLASSES.index(label),
                "player": int(m.group(1)) if m else -1,
            })
    df = pd.DataFrame(rows).drop_duplicates("video_path").sort_values("video_path").reset_index(drop=True)
    df["video_id"] = [f"{r.label}__{os.path.splitext(os.path.basename(r.video_path))[0]}" for r in df.itertuples()]
    return df


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path],
        capture_output=True, text=True)
    try:
        s = json.loads(out.stdout)["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return int(s["width"]), int(s["height"]), float(num) / float(den)
    except Exception:
        return -1, -1, -1.0


def extract(row, cache_dir, short_side):
    out_dir = os.path.join(cache_dir, row["video_id"])
    done_flag = os.path.join(out_dir, ".done")
    if not os.path.exists(done_flag):
        os.makedirs(out_dir, exist_ok=True)
        scale = (f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'")
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", row["video_path"],
               "-vf", scale, "-q:v", "2", "-start_number", "0",
               os.path.join(out_dir, "%05d.jpg")]
        subprocess.run(cmd, check=True)
        open(done_flag, "w").close()
    n = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    w, h, fps = probe(row["video_path"])
    return out_dir, n, w, h, fps


def make_split(df, mode, ratios, seed):
    tr, va, te = ratios
    if mode == "stratified":
        idx = np.arange(len(df))
        i_tr, i_rest = train_test_split(idx, train_size=tr, stratify=df.label_id, random_state=seed)
        i_va, i_te = train_test_split(i_rest, train_size=va / (va + te),
                                      stratify=df.label_id.iloc[i_rest], random_state=seed)
        split = np.empty(len(df), dtype=object)
        split[i_tr], split[i_va], split[i_te] = "train", "val", "test"
        return split
    if mode == "player":
        players = np.array(sorted(df.player.unique()))
        rng = np.random.RandomState(seed)
        rng.shuffle(players)
        n = len(players)
        n_tr, n_va = round(tr * n), round(va * n)
        assign = {p: "train" for p in players[:n_tr]}
        assign.update({p: "val" for p in players[n_tr:n_tr + n_va]})
        assign.update({p: "test" for p in players[n_tr + n_va:]})
        return df.player.map(assign).values
    raise ValueError(mode)


def figures(df, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)

    # 1. class distribution per split
    ct = pd.crosstab(df.label, df.split).reindex(CLASSES)[["train", "val", "test"]]
    ax = ct.rename(index=PRETTY).plot(kind="bar", stacked=True, figsize=(9, 4),
                                       color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylabel("Clips")
    ax.set_title("THETIS RGB: clips per class and split")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "class_distribution.png"), dpi=200)
    plt.close()

    # 2. video length distribution
    plt.figure(figsize=(6, 3.5))
    plt.hist(df.n_frames, bins=40, color="#4C72B0")
    plt.axvline(64, color="red", ls="--", label="64-frame clip window")
    plt.xlabel("Frames per video")
    plt.ylabel("Videos")
    plt.legend()
    plt.title("THETIS RGB: video length")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "video_lengths.png"), dpi=200)
    plt.close()

    # 3. one mid-stroke frame per class
    fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
    for ax, c in zip(axes.flat, CLASSES):
        ax.axis("off")
        if not (df.label == c).any():
            continue
        r = df[df.label == c].iloc[0]
        img = Image.open(os.path.join(r.frame_dir, f"{r.n_frames // 2:05d}.jpg"))
        ax.imshow(img)
        ax.set_title(PRETTY[c], fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "sample_frames.png"), dpi=150)
    plt.close()

    # 4. clip sampling illustration on one video
    r = df.iloc[len(df) // 2]
    n = r.n_frames
    start = max(n - 64, 0) // 2
    fast = np.clip(np.linspace(start, start + 63, 32).round().astype(int), 0, n - 1)
    slow = fast[np.linspace(0, 31, 4).astype(int)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, i in zip(axes, slow):
        ax.imshow(Image.open(os.path.join(r.frame_dir, f"{i:05d}.jpg")))
        ax.set_title(f"slow frame t={i}")
        ax.axis("off")
    plt.suptitle(f"Slow pathway input (4 of 32 fast frames), {PRETTY[r.label]}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "slow_pathway_frames.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config, args.opts)
    d = cfg["data"]

    df = find_videos(d["root"])
    print(f"Found {len(df)} RGB videos in {df.label.nunique()} classes, {df.player.nunique()} player ids")
    if len(df) == 0:
        raise SystemExit("No videos found. Check data.root.")

    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda r: extract(r, d["cache_dir"], d["cache_short_side"]),
                          df.to_dict("records")))
    df["frame_dir"], df["n_frames"], df["width"], df["height"], df["fps"] = zip(*res)
    bad = df[df.n_frames == 0]
    if len(bad):
        print(f"WARNING: {len(bad)} videos decoded to 0 frames and are dropped:\n{bad.video_path.tolist()}")
        df = df[df.n_frames > 0].reset_index(drop=True)

    df["split"] = make_split(df, d["split_mode"], d["split_ratios"], cfg["seed"])
    os.makedirs(os.path.dirname(d["split_csv"]), exist_ok=True)
    df.to_csv(d["split_csv"], index=False)

    stats = {
        "num_videos": int(len(df)),
        "num_classes": int(df.label.nunique()),
        "num_players": int(df.player.nunique()),
        "split_mode": d["split_mode"],
        "split_counts": df.split.value_counts().to_dict(),
        "per_class": df.label.value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict(),
        "per_class_per_split": pd.crosstab(df.label, df.split).reindex(CLASSES).to_dict(),
        "frames": {k: float(v) for k, v in df.n_frames.describe().items()},
        "short_videos_lt_64_frames": int((df.n_frames < 64).sum()),
        "fps_values": df.fps.round(2).value_counts().to_dict(),
        "resolutions": df.apply(lambda r: f"{r.width}x{r.height}", axis=1).value_counts().to_dict(),
    }
    out = cfg["train"]["output_dir"]
    save_json(stats, os.path.join(out, "dataset_stats.json"))
    figures(df, os.path.join(out, "figures"))
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
