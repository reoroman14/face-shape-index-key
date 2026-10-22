# Stage 0: run the router and the embedder once over LFW and cache the
# outputs in results/pilot/cache/. Failures are recorded as failures (flag
# false, values NaN); nothing is imputed.
#
# Run:  python -m faceshape_pilot.extract [--force]

import argparse
import time
from pathlib import Path

import numpy as np

from faceshape_pilot import config, embed, router
from faceshape_pilot.common import write_json, environment_info
from faceshape_pilot.data import load_lfw_index, read_rgb
from faceshape_pilot.models import ensure_mediapipe_model

ROUTER_CACHE = config.CACHE_DIR / "router.npz"
EMBED_CACHE = config.CACHE_DIR / "embeddings.npz"


def run_router(index) -> dict:
    detector = router.LandmarkDetector(ensure_mediapipe_model())
    n = len(index.paths)
    raw = np.full((n, len(router.RAW_FEATURE_NAMES)), np.nan)
    ok = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, path in enumerate(index.paths):
        pts = detector.detect(read_rgb(path))
        if pts is not None:
            raw[i] = router.raw_features(pts)
            ok[i] = True
        if (i + 1) % 1000 == 0:
            print(f"[router] {i + 1}/{n}  ok={ok[: i + 1].sum()}  {time.time() - t0:.0f}s", flush=True)
    detector.close()

    ratios = np.full((n, 4), np.nan)
    ratios[ok] = router.ratio_features(raw[ok])
    stats = router.fit_standardiser(ratios[ok])
    z = np.full((n, 4), np.nan)
    z[ok] = router.standardise(ratios[ok], stats)
    probs = np.full((n, len(config.CLASSES)), np.nan)
    probs[ok] = router.shape_probabilities(z[ok])
    return {
        "landmark_ok": ok,
        "raw_features": raw,
        "ratio_features": ratios,
        "z_features": z,
        "probs": probs,
        "standardiser": stats,
        "seconds": time.time() - t0,
    }


def run_embedder(index) -> dict:
    embedder = embed.ArcFaceEmbedder()
    providers = embedder.providers_in_use()
    print(f"[embed] providers in use: {providers}", flush=True)
    n = len(index.paths)
    emb = np.full((n, 512), np.nan, dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    score = np.full(n, np.nan)
    det_size = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for i, path in enumerate(index.paths):
        e, s, d = embedder.embed(read_rgb(path))
        if e is not None:
            emb[i], score[i], det_size[i], ok[i] = e, s, d, True
        if (i + 1) % 500 == 0:
            print(f"[embed] {i + 1}/{n}  ok={ok[: i + 1].sum()}  {time.time() - t0:.0f}s", flush=True)
    return {"embed_ok": ok, "embeddings": emb, "det_score": score, "det_size": det_size,
            "seconds": time.time() - t0, "providers_in_use": providers}


def main(force: bool = False):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index = load_lfw_index()
    rel = np.array([Path(p).relative_to(config.DATA_DIR).as_posix() for p in index.paths])
    common = {"paths": rel, "identity": index.identity, "names": np.array(index.names)}
    meta = {"n_images": len(index.paths), "n_identities": len(index.names)}

    if force or not ROUTER_CACHE.exists():
        r = run_router(index)
        np.savez_compressed(
            ROUTER_CACHE,
            **common,
            **{k: v for k, v in r.items() if isinstance(v, np.ndarray)},
            standardiser_median=np.asarray(r["standardiser"]["median"]),
            standardiser_scale=np.asarray(r["standardiser"]["robust_scale"]),
        )
        meta["router"] = {
            "n_landmark_ok": int(r["landmark_ok"].sum()),
            "n_landmark_failed": int((~r["landmark_ok"]).sum()),
            "standardiser": r["standardiser"],
            "seconds": round(r["seconds"], 1),
        }
        write_json(config.CACHE_DIR / "router_meta.json", meta["router"])
    if force or not EMBED_CACHE.exists():
        e = run_embedder(index)
        np.savez_compressed(EMBED_CACHE, **common, **{k: v for k, v in e.items() if isinstance(v, np.ndarray)})
        meta["embedder"] = {
            "n_embed_ok": int(e["embed_ok"].sum()),
            "n_embed_failed": int((~e["embed_ok"]).sum()),
            "seconds": round(e["seconds"], 1),
            "providers_in_use": e["providers_in_use"],
            "det_size": embed.DET_SIZE[0],
        }
        write_json(config.CACHE_DIR / "embed_meta.json", meta["embedder"])
    write_json(config.CACHE_DIR / "extract_environment.json", environment_info())
    print("[extract] done", meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if caches exist")
    main(ap.parse_args().force)
