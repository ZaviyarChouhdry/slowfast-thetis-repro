"""THETIS clip dataset that mirrors PySlowFast's Kinetics sampling.

train / val : one random 64-frame window, 32 frames at stride 2, random short-side
              scale in [256, 320], random 224 crop, random horizontal flip.
              (PySlowFast uses the same random sampling for validation; the paper
              says validation used "one randomly sampled clip".)
test        : num_clips uniformly spaced windows x num_crops spatial crops
              (left / centre / right along the long side) at short side 256.
"""
import math
import os
import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg, read_file


def temporal_indices(n_frames, window, num_frames, clip_idx, num_clips, rng):
    """PySlowFast get_start_end_idx + temporal_sampling (indices clamped to video)."""
    delta = max(n_frames - window, 0)
    if clip_idx == -1:
        start = rng.uniform(0, delta)
    else:
        start = delta * clip_idx / num_clips
    end = start + window - 1
    idx = torch.linspace(start, end, num_frames).round().long()
    return idx.clamp(0, n_frames - 1)


def load_frames(frame_dir, idx):
    frames = [decode_jpeg(read_file(os.path.join(frame_dir, f"{int(i):05d}.jpg"))) for i in idx]
    return torch.stack(frames)  # T, C, H, W uint8


def short_side_resize(x, size):
    # x: T, C, H, W float
    t, c, h, w = x.shape
    if min(h, w) == size:
        return x
    if h < w:
        nh, nw = size, int(math.floor(w * size / h))
    else:
        nh, nw = int(math.floor(h * size / w)), size
    return F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)


def crop(x, size, mode, rng):
    _, _, h, w = x.shape
    if mode == "random":
        y0 = rng.randint(0, h - size)
        x0 = rng.randint(0, w - size)
    else:  # 0, 1, 2 = left/top, centre, right/bottom (PySlowFast uniform_crop)
        y0, x0 = (h - size) // 2, (w - size) // 2
        if mode != 1:
            if w > h:
                x0 = 0 if mode == 0 else w - size
            else:
                y0 = 0 if mode == 0 else h - size
    return x[:, :, y0:y0 + size, x0:x0 + size]


class ThetisClips(Dataset):
    def __init__(self, split_csv, split, cfg, mode=None, num_clips=1, num_crops=1, seed=0):
        df = pd.read_csv(split_csv)
        self.df = df[df.split == split].reset_index(drop=True)
        self.d = cfg["data"]
        self.alpha = cfg["model"]["alpha"]
        self.mode = mode or ("train" if split in ("train", "val") else "test")
        self.num_clips, self.num_crops = num_clips, num_crops
        self.mean = torch.tensor(self.d["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(self.d["std"]).view(1, 3, 1, 1)
        self.seed = seed
        self.epoch = 0
        # test: every (video, clip, crop) is its own item
        self.items = [(v, c, s) for v in range(len(self.df))
                      for c in range(num_clips) for s in range(num_crops)] \
            if self.mode == "test" else [(v, -1, "random") for v in range(len(self.df))]

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def _rng(self, i):
        # deterministic per (seed, epoch, item) so runs and val curves are repeatable
        return random.Random(hash((self.seed, self.epoch, i)) & 0xFFFFFFFF)

    def __getitem__(self, i):
        v, clip_idx, crop_mode = self.items[i]
        r = self.df.iloc[v]
        rng = self._rng(i)
        idx = temporal_indices(int(r.n_frames), self.d["clip_window"], self.d["num_frames"],
                               clip_idx, self.num_clips, rng)
        x = load_frames(r.frame_dir, idx).float() / 255.0  # T C H W

        if self.mode == "train":
            lo, hi = self.d["train_jitter_scales"]
            x = short_side_resize(x, rng.randint(lo, hi))
            x = crop(x, self.d["train_crop_size"], "random", rng)
            if self.d.get("random_flip", True) and rng.random() < 0.5:
                x = x.flip(-1)
        else:
            x = short_side_resize(x, self.d["cache_short_side"])
            x = crop(x, self.d["test_crop_size"], crop_mode if self.num_crops == 3 else 1, rng)

        x = (x - self.mean) / self.std
        fast = x.permute(1, 0, 2, 3).contiguous()  # C T H W
        # PySlowFast pack_pathway_output: 32 / alpha = 4 frames spread over the clip
        t = fast.shape[1]
        slow = fast.index_select(1, torch.linspace(0, t - 1, t // self.alpha).long())
        return [slow, fast], int(r.label_id), v
