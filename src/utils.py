"""Shared helpers: config loading, seeding, environment logging, class list."""
import json
import os
import platform
import random
import re
import subprocess

import numpy as np
import torch
import yaml

# Canonical THETIS RGB class order (Table 1 of Hovad et al.).
CLASSES = [
    "backhand",
    "backhand2hands",
    "backhand_slice",
    "backhand_volley",
    "flat_service",
    "forehand_flat",
    "forehand_openstands",
    "forehand_slice",
    "forehand_volley",
    "kick_service",
    "slice_service",
    "smash",
]

# Pretty names for plots.
PRETTY = {
    "backhand": "Backhand",
    "backhand2hands": "Backhand 2H",
    "backhand_slice": "BH slice",
    "backhand_volley": "BH volley",
    "flat_service": "Flat serve",
    "forehand_flat": "Forehand flat",
    "forehand_openstands": "FH open stance",
    "forehand_slice": "FH slice",
    "forehand_volley": "FH volley",
    "kick_service": "Kick serve",
    "slice_service": "Slice serve",
    "smash": "Smash",
}


def normalise_class_name(name: str):
    """Map a folder name from any THETIS mirror to a canonical class, or None."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    for c in CLASSES:
        if re.sub(r"[^a-z0-9]", "", c) == key:
            return c
    aliases = {
        "backhand2hand": "backhand2hands",
        "backhandtwohands": "backhand2hands",
        "forehandopenstand": "forehand_openstands",
        "forehandopenstance": "forehand_openstands",
        "flatserve": "flat_service",
        "kickserve": "kick_service",
        "sliceserve": "slice_service",
    }
    return aliases.get(key)


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        # "section.key=value" overrides from the command line
        k, v = ov.split("=", 1)
        sec, key = k.split(".", 1)
        cfg[sec][key] = yaml.safe_load(v)
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def environment_info():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count(),
    }
    try:
        import torchvision
        import pytorchvideo
        info["torchvision"] = torchvision.__version__
        info["pytorchvideo"] = getattr(pytorchvideo, "__version__", "0.1.5")
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    try:
        info["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    except Exception:
        pass
    return info


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
