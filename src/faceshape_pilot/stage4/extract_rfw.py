# Runs the stage 1 landmark step, unchanged, over every RFW image and caches
# the landmark flags, the five raw measurements and the four ratio features
# in results/stage4/cache/rfw_router.npz. The embedder is not run: Part B
# uses measurement 1 only.
#
# No standardiser is fitted here. part_b.py applies the three standardiser
# conditions (pooled RFW, per group, LFW transferred) to the cached ratios.
#
# Step 0 (--step0) runs the same code on the first 200 images of each group,
# in enumeration order, and prints the detection rate per group next to the
# LFW rate from results/pilot/cache/router_meta.json. It writes
# results/stage4/step0.json and nothing else.
#
# Images that fail to decode or in which no face is found are recorded as
# failures (flag false, values NaN); nothing is imputed.
#
# Run:  python -m faceshape_pilot.stage4.extract_rfw [--force | --step0]

import argparse
import time
from pathlib import Path

import numpy as np

from faceshape_pilot import config, router
from faceshape_pilot.common import environment_info, read_json, write_json
from faceshape_pilot.data import read_rgb
from faceshape_pilot.models import ensure_mediapipe_model, sha256_file
from faceshape_pilot.stage3.part_a import wilson
from faceshape_pilot.stage4 import rfw

ROUTER_CACHE = config.STAGE4_CACHE / "rfw_router.npz"
META = config.STAGE4_CACHE / "rfw_router_meta.json"
STEP0_JSON = config.STAGE4_DIR / "step0.json"
LFW_ROUTER_META = config.CACHE_DIR / "router_meta.json"
STEP0_PER_GROUP = 200


def landmarks(paths, label: str = "router"):
    detector = router.LandmarkDetector(ensure_mediapipe_model())
    n = len(paths)
    raw = np.full((n, len(router.RAW_FEATURE_NAMES)), np.nan)
    ok = np.zeros(n, dtype=bool)
    decoded = np.zeros(n, dtype=bool)
    height = np.zeros(n, dtype=np.int64)
    width = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for i, path in enumerate(paths):
        try:
            rgb = read_rgb(path)
        except Exception:
            rgb = None
        if rgb is not None:
            decoded[i] = True
            height[i], width[i] = rgb.shape[:2]
            pts = detector.detect(rgb)
            if pts is not None:
                raw[i] = router.raw_features(pts)
                ok[i] = True
        if (i + 1) % 2000 == 0:
            print(f"[{label}] {i + 1}/{n}  ok={ok[: i + 1].sum()}  {time.time() - t0:.0f}s", flush=True)
    detector.close()
    ratios = np.full((n, 4), np.nan)
    ratios[ok] = router.ratio_features(raw[ok])
    return {"decoded": decoded, "landmark_ok": ok, "height": height, "width": width,
            "raw_features": raw, "ratio_features": ratios, "seconds": time.time() - t0}


def lfw_reference() -> dict:
    m = read_json(LFW_ROUTER_META)
    n = m["n_landmark_ok"] + m["n_landmark_failed"]
    return {"n_landmark_ok": m["n_landmark_ok"], "n_images": n, "rate": m["n_landmark_ok"] / n,
            "source": LFW_ROUTER_META.relative_to(config.REPO_ROOT).as_posix()}


def step0():
    index = rfw.load_rfw_index()
    pick = np.concatenate([np.flatnonzero(index.group == gi)[:STEP0_PER_GROUP]
                           for gi in range(len(config.RFW_GROUPS))])
    r = landmarks([index.paths[i] for i in pick], label="step0")
    lfw = lfw_reference()
    groups = {}
    for gi, g in enumerate(config.RFW_GROUPS):
        m = index.group[pick] == gi
        k, n = int(r["landmark_ok"][m].sum()), int(m.sum())
        lo, hi = wilson(k, n)
        groups[g] = {"n_images": n, "n_identities_covered": int(len(np.unique(index.identity[pick][m]))),
                     "n_decoded": int(r["decoded"][m].sum()), "n_landmark_ok": k,
                     "detection_rate": k / n, "detection_rate_ci95_wilson": [lo, hi],
                     "first_path": Path(index.paths[pick[m][0]]).relative_to(config.DATA_DIR).as_posix(),
                     "last_path": Path(index.paths[pick[m][-1]]).relative_to(config.DATA_DIR).as_posix()}
    k_all = int(r["landmark_ok"].sum())
    out = {
        "step": "0",
        "seed": config.SEED,
        "description": f"MediaPipe landmark detection on the first {STEP0_PER_GROUP} images of each RFW group "
                       "(enumeration order: identity folder, then file name), stage 1 detector unchanged",
        "provenance": rfw.provenance(),
        "rfw_counts_verified": index.counts(),
        "image_sizes": sorted({f"{h}x{w}" for h, w, d in zip(r["height"], r["width"], r["decoded"]) if d}),
        "groups": groups,
        "pooled": {"n_images": int(len(pick)), "n_landmark_ok": k_all, "detection_rate": k_all / len(pick)},
        "lfw_reference": lfw,
        "mediapipe_model_sha256": sha256_file(ensure_mediapipe_model()),
        "seconds": round(r["seconds"], 1),
        "environment": environment_info(),
    }
    write_json(STEP0_JSON, out)
    print(f"[step0] LFW reference: {lfw['n_landmark_ok']} of {lfw['n_images']} = {lfw['rate']:.4f}")
    for g, d in groups.items():
        lo, hi = d["detection_rate_ci95_wilson"]
        print(f"[step0] {g:10s} {d['n_landmark_ok']:3d} of {d['n_images']} = {d['detection_rate']:.4f} "
              f"(95% Wilson {lo:.4f} to {hi:.4f}), {d['n_identities_covered']} identities")
    print(f"[step0] pooled RFW {k_all} of {len(pick)} = {k_all / len(pick):.4f}  -> {STEP0_JSON}")
    return out


def run(force: bool = False):
    if ROUTER_CACHE.exists() and not force:
        print(f"[extract] cache exists: {ROUTER_CACHE}")
        return
    index = rfw.load_rfw_index()
    model = ensure_mediapipe_model()
    r = landmarks(index.paths)
    rel = np.array([Path(p).relative_to(config.DATA_DIR).as_posix() for p in index.paths])
    config.STAGE4_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ROUTER_CACHE, paths=rel, group=index.group, identity=index.identity, names=np.array(index.names),
        identity_group=index.identity_group, **{k: v for k, v in r.items() if isinstance(v, np.ndarray)},
    )
    ok, decoded = r["landmark_ok"], r["decoded"]
    write_json(META, {
        "provenance": rfw.provenance(),
        "rfw_counts_verified": index.counts(),
        "n_images": len(index.paths),
        "n_decode_failed": int((~decoded).sum()),
        "n_landmark_ok": int(ok.sum()),
        "n_landmark_failed_after_decode": int((decoded & ~ok).sum()),
        "image_sizes": sorted({f"{h}x{w}" for h, w in zip(r["height"][decoded], r["width"][decoded])}),
        "mediapipe_model_sha256": sha256_file(model),
        "router": router.router_spec(),
        "embedder": "not run (Part B uses measurement 1 only)",
        "seconds": round(r["seconds"], 1),
        "environment": environment_info(),
    })
    print(f"[extract] {ok.sum()} of {len(ok)} with landmarks, {r['seconds']:.0f}s -> {ROUTER_CACHE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    ap.add_argument("--step0", action="store_true", help=f"first {STEP0_PER_GROUP} images per group only")
    args = ap.parse_args()
    if args.step0:
        step0()
    else:
        run(args.force)
