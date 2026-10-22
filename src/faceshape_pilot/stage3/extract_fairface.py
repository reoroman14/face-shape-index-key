# Runs the stage 1 geometric router, unchanged, over every FairFace image and
# caches the outputs in results/stage3/cache/fairface_router.npz.
#
# The router is the stage 1 code path: same MediaPipe Face Landmarker model
# file (sha256 recorded), same landmarks, features, prototypes, width and
# temperature. Its only data-dependent component is the robust standardiser
# (median / IQR of the four ratio features). Two variants are cached:
#
#   frozen : the standardiser fitted on LFW in stage 1, read from
#            results/pilot/cache/router.npz. This is exactly the stage 1
#            index key applied to new faces. Primary analysis.
#   refit  : the standardiser re-fitted on all FairFace images with
#            landmarks (the stage 1 procedure applied to a new image set).
#            Sensitivity analysis.
#
# Images that fail to decode or in which no face is found are recorded as
# failures (flag false, values NaN); nothing is imputed.
#
# Run:  python -m faceshape_pilot.stage3.extract_fairface [--force]

import argparse
import time

import numpy as np

from faceshape_pilot import config, router
from faceshape_pilot.common import environment_info, write_json
from faceshape_pilot.models import ensure_mediapipe_model, sha256_file
from faceshape_pilot.stage3 import fairface

ROUTER_CACHE = config.STAGE3_CACHE / "fairface_router.npz"
META = config.STAGE3_CACHE / "fairface_router_meta.json"
STAGE1_ROUTER_CACHE = config.CACHE_DIR / "router.npz"


def lfw_standardiser() -> dict:
    r = np.load(STAGE1_ROUTER_CACHE)
    return {"median": r["standardiser_median"].tolist(), "robust_scale": r["standardiser_scale"].tolist()}


def probs_with(ratios: np.ndarray, ok: np.ndarray, stats: dict):
    z = np.full(ratios.shape, np.nan)
    z[ok] = router.standardise(ratios[ok], stats)
    p = np.full((len(ratios), len(config.CLASSES)), np.nan)
    p[ok] = router.shape_probabilities(z[ok])
    return z, p


def run(force: bool = False):
    if ROUTER_CACHE.exists() and not force:
        print(f"[extract] cache exists: {ROUTER_CACHE}")
        return
    shards = fairface.shard_list()
    idx = fairface.load_index(shards)
    n = len(idx)
    model = ensure_mediapipe_model()
    detector = router.LandmarkDetector(model)
    raw = np.full((n, len(router.RAW_FEATURE_NAMES)), np.nan)
    ok = np.zeros(n, dtype=bool)
    decoded = np.zeros(n, dtype=bool)
    height = np.zeros(n, dtype=np.int64)
    width = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for i, rgb in fairface.iter_images(shards):
        if rgb is not None:
            decoded[i] = True
            height[i], width[i] = rgb.shape[:2]
            pts = detector.detect(rgb)
            if pts is not None:
                raw[i] = router.raw_features(pts)
                ok[i] = True
        if (i + 1) % 5000 == 0:
            print(f"[router] {i + 1}/{n}  ok={ok[: i + 1].sum()}  {time.time() - t0:.0f}s", flush=True)
    detector.close()
    seconds = time.time() - t0

    ratios = np.full((n, 4), np.nan)
    ratios[ok] = router.ratio_features(raw[ok])
    std_frozen = lfw_standardiser()
    std_refit = router.fit_standardiser(ratios[ok])
    z_frozen, p_frozen = probs_with(ratios, ok, std_frozen)
    z_refit, p_refit = probs_with(ratios, ok, std_refit)

    config.STAGE3_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ROUTER_CACHE, split=idx.split, shard=idx.shard, row=idx.row, race=idx.race, gender=idx.gender, age=idx.age,
        decoded=decoded, landmark_ok=ok, height=height, width=width, raw_features=raw, ratio_features=ratios,
        z_frozen=z_frozen, probs_frozen=p_frozen, z_refit=z_refit, probs_refit=p_refit,
    )
    write_json(META, {
        "n_images": n,
        "n_decode_failed": int((~decoded).sum()),
        "n_landmark_ok": int(ok.sum()),
        "n_landmark_failed_after_decode": int((decoded & ~ok).sum()),
        "image_sizes": sorted({f"{h}x{w}" for h, w in zip(height[decoded], width[decoded])}),
        "standardiser_frozen_lfw": std_frozen,
        "standardiser_refit_fairface": std_refit,
        "mediapipe_model_sha256": sha256_file(model),
        "router": router.router_spec(),
        "seconds": round(seconds, 1),
        "environment": environment_info(),
    })
    print(f"[extract] {ok.sum()} of {n} with landmarks, {seconds:.0f}s -> {ROUTER_CACHE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    run(ap.parse_args().force)
