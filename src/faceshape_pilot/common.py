# Shared helpers: JSON output, environment capture, the analysis set and
# plot styling.

import datetime as _dt
import json
import math
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

from faceshape_pilot import config

PACKAGES = (
    "numpy", "scipy", "scikit-learn", "matplotlib", "pillow", "opencv-python-headless",
    "mediapipe", "insightface", "onnxruntime-gpu", "torch", "torchvision", "pyarrow",
)


def _clean(obj):
    # Converts numpy types to Python types. NaN and inf become None (JSON
    # null): a value that was not measured stays empty.
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, indent=2, allow_nan=False)
        f.write("\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def environment_info() -> dict:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "utc_time": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


LFW_DATASET = "LFW (funneled), identities with >= 2 images, via sklearn.datasets.fetch_lfw_people"


def run_header(measurement: str, dataset: str = LFW_DATASET) -> dict:
    # Common block written at the top of every result file.
    from faceshape_pilot.router import router_spec

    return {
        "measurement": measurement,
        "seed": config.SEED,
        "dataset": dataset,
        "dataset_limitations": config.DATASET_LIMITATIONS,
        "router": router_spec(),
        "embedding": f"InsightFace {config.INSIGHTFACE_PACK} ArcFace R50 (WebFace600K), L2-normalised, 512-d",
        "environment": environment_info(),
    }


class AnalysisSet:
    # Images for which both the router and the embedder succeeded, restricted
    # to identities that still have at least two such images. The same set is
    # used by all three measurements so their numbers refer to the same data.

    def __init__(self):
        from faceshape_pilot.extract import EMBED_CACHE, ROUTER_CACHE, main as extract_main

        if not (ROUTER_CACHE.exists() and EMBED_CACHE.exists()):
            extract_main()
        r = np.load(ROUTER_CACHE, allow_pickle=False)
        e = np.load(EMBED_CACHE, allow_pickle=False)
        if not np.array_equal(r["paths"], e["paths"]):
            raise RuntimeError("Router and embedding caches refer to different image lists; re-run extract --force")

        n_all = len(r["paths"])
        ok = r["landmark_ok"] & e["embed_ok"]
        ident_all = r["identity"]
        counts = np.bincount(ident_all[ok], minlength=len(r["names"]))
        keep = ok & (counts[ident_all] >= config.MIN_FACES_PER_PERSON)

        self.keep = keep
        self.paths = r["paths"][keep]
        old_ids = ident_all[keep]
        uniq, self.identity = np.unique(old_ids, return_inverse=True)
        self.names = r["names"][uniq]
        self.probs = r["probs"][keep]
        self.z_features = r["z_features"][keep]
        self.ratio_features = r["ratio_features"][keep]
        self.embeddings = e["embeddings"][keep].astype(np.float64)
        self.top1 = self.probs.argmax(axis=1)
        self.n_images = int(keep.sum())
        self.n_identities = len(uniq)
        self.accounting = {
            "lfw_images_ge2": n_all,
            "lfw_identities_ge2": int(len(r["names"])),
            "landmark_failed": int((~r["landmark_ok"]).sum()),
            "embedding_failed": int((~e["embed_ok"]).sum()),
            "either_failed": int((~ok).sum()),
            "dropped_identity_below_2_after_failures_images": int((ok & ~keep).sum()),
            "analysis_images": self.n_images,
            "analysis_identities": self.n_identities,
        }

    def images_of(self):
        # List of index arrays, one per identity, sorted.
        order = np.argsort(self.identity, kind="stable")
        bounds = np.flatnonzero(np.diff(self.identity[order])) + 1
        return np.split(order, bounds)


def percentile_ci(samples: np.ndarray, level: float = 0.95):
    a = (1 - level) / 2
    lo, hi = np.nanpercentile(samples, [100 * a, 100 * (1 - a)], axis=0)
    return lo, hi


# Plot styling: reference palette (dataviz skill), slots 1 and 2.
COLOR_SHAPE = "#2a78d6"
COLOR_KMEANS = "#eb6834"
COLOR_SURFACE = "#fcfcfb"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_2 = "#52514e"
COLOR_GRID = "#e4e3df"


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": COLOR_SURFACE,
        "axes.facecolor": COLOR_SURFACE,
        "savefig.facecolor": COLOR_SURFACE,
        "axes.edgecolor": COLOR_TEXT_2,
        "axes.labelcolor": COLOR_TEXT,
        "text.color": COLOR_TEXT,
        "xtick.color": COLOR_TEXT_2,
        "ytick.color": COLOR_TEXT_2,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLOR_GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "lines.linewidth": 2,
        "lines.markersize": 7,
    })
    return plt
