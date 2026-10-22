# Step 0 check (PILOT2.md): after moving ArcFace to the GPU and removing the
# 640x640 detector fallback, confirm stage 1 is unchanged, and explain any
# value that moved.
#
# Compares the re-run in results/pilot/ against the CPU run archived in
# results/stage2/step0_cpu_reference/ (JSON results plus both caches):
#   1. router cache: must be bit-identical (MediaPipe runs on CPU both times)
#   2. measurements 1 and 2: must be identical (they use only the router)
#   3. embeddings: size of the GPU-versus-CPU difference
#   4. measurement 3: the current code re-run on the archived CPU embeddings
#      must reproduce the archived CPU numbers exactly; that isolates any
#      change in measurement 3 to the embeddings
#   5. k-means seed sensitivity on CPU and GPU embeddings, to show how large
#      a change in the k-means curve is relative to its optimisation variance
#
# Run:  python -m faceshape_pilot.stage2.step0_check

import numpy as np

from faceshape_pilot import config, m3_partition
from faceshape_pilot.common import AnalysisSet, read_json, write_json, environment_info

OUT_JSON = config.STAGE2_DIR / "step0_gpu_check.json"
N_KMEANS_SEEDS = 20
VOLATILE_KEYS = {"environment", "utc_time"}


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in VOLATILE_KEYS}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


def _diff(a, b, path=""):
    # Lists every leaf where two JSON trees differ: (path, old, new).
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k), b.get(k), f"{path}.{k}" if path else k)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append({"path": path, "cpu_run": a, "gpu_run": b})
    return out


def compare_router():
    ref = np.load(config.STEP0_CPU_REFERENCE / "router_cpu.npz")
    cur = np.load(config.CACHE_DIR / "router.npz")
    keys = sorted(set(ref.files) | set(cur.files))
    mismatched = []
    for k in keys:
        if k not in ref.files or k not in cur.files:
            mismatched.append(k)
            continue
        a, b = ref[k], cur[k]
        same = a.shape == b.shape and (
            np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)
        )
        if not same:
            mismatched.append(k)
    return {"arrays_compared": keys, "arrays_not_bit_identical": mismatched, "bit_identical": not mismatched}


def compare_embeddings():
    ref = np.load(config.STEP0_CPU_REFERENCE / "embeddings_cpu.npz")
    cur = np.load(config.CACHE_DIR / "embeddings.npz")
    ok = ref["embed_ok"] & cur["embed_ok"]
    a, b = ref["embeddings"][ok].astype(np.float64), cur["embeddings"][ok].astype(np.float64)
    cos = (a * b).sum(axis=1)
    return {
        "embed_ok_cpu": int(ref["embed_ok"].sum()),
        "embed_ok_gpu": int(cur["embed_ok"].sum()),
        "embed_ok_identical": bool(np.array_equal(ref["embed_ok"], cur["embed_ok"])),
        "cpu_run_det_size_counts": {int(k): int(v) for k, v in zip(*np.unique(ref["det_size"], return_counts=True))},
        "max_abs_component_diff": float(np.abs(a - b).max()),
        "mean_abs_component_diff": float(np.abs(a - b).mean()),
        "cosine_cpu_vs_gpu_min": float(cos.min()),
        "cosine_cpu_vs_gpu_median": float(np.median(cos)),
        "det_score_max_abs_diff": float(np.nanmax(np.abs(ref["det_score"][ok] - cur["det_score"][ok]))),
    }


def compare_results():
    out = {}
    for name in ("consistency.json", "distribution.json", "partition_comparison.json"):
        a = _strip(read_json(config.STEP0_CPU_REFERENCE / name))
        b = _strip(read_json(config.RESULTS_DIR / name))
        d = _diff(a, b)
        out[name] = {"identical": not d, "n_differing_values": len(d), "differences": d[:40]}
    return out


def _with_embeddings(ds: AnalysisSet, cache_path):
    e = np.load(cache_path)
    ds.embeddings = e["embeddings"][ds.keep].astype(np.float64)
    return ds


def reproduce_cpu_m3():
    # Current measurement-3 code on the archived CPU embeddings.
    ds = _with_embeddings(AnalysisSet(), config.STEP0_CPU_REFERENCE / "embeddings_cpu.npz")
    split, kmeans_info, parts, comparison = m3_partition.partition_comparison(ds, write_split=False)
    probe = m3_partition.linear_probe(ds, write_split=False)
    ref = read_json(config.STEP0_CPU_REFERENCE / "partition_comparison.json")
    fresh = _strip(read_json_like({"partitions": parts, "comparison": comparison, "linear_probe": probe,
                                   "kmeans": kmeans_info}))
    old = {k: _strip(ref[k]) for k in ("partitions", "comparison", "linear_probe", "kmeans")}
    d = _diff(old, fresh)
    return {"reproduces_archived_cpu_numbers_exactly": not d, "n_differing_values": len(d), "differences": d[:20]}


def read_json_like(obj):
    # Round-trips through the same JSON cleaning used for result files so
    # the comparison is like-for-like.
    import json

    from faceshape_pilot.common import _clean

    return json.loads(json.dumps(_clean(obj)))


def kmeans_seed_sensitivity():
    rows = {}
    for label, path in (("cpu", config.STEP0_CPU_REFERENCE / "embeddings_cpu.npz"),
                        ("gpu", config.CACHE_DIR / "embeddings.npz")):
        ds = _with_embeddings(AnalysisSet(), path)
        per_seed = []
        for i in range(N_KMEANS_SEEDS):
            seed = config.SEED + i
            _, info, parts, _ = m3_partition.partition_comparison(ds, kmeans_seed=seed, write_split=False)
            km = parts["kmeans"]
            per_seed.append({
                "kmeans_random_state": seed,
                "auc": km["auc_recall_vs_penetration"],
                "recall_1_bucket": km["curve"][0]["routing_recall"],
                "penetration_1_bucket": km["curve"][0]["penetration"],
                "largest_bucket_share": km["gallery_largest_bucket_share"],
                "inertia": info["inertia"],
            })
        aucs = np.array([r["auc"] for r in per_seed])
        rows[label] = {
            "per_seed": per_seed,
            "auc_min": aucs.min(), "auc_max": aucs.max(), "auc_mean": aucs.mean(), "auc_sd": aucs.std(ddof=1),
        }
    return {"n_seeds": N_KMEANS_SEEDS, "seeds": f"{config.SEED} .. {config.SEED + N_KMEANS_SEEDS - 1}", **rows}


def main():
    manifest = read_json(config.RESULTS_DIR / "run_manifest.json")
    cpu_meta = read_json(config.STEP0_CPU_REFERENCE / "embed_meta.json")
    result = {
        "seed": config.SEED,
        "change": "ArcFace/SCRFD moved from CPUExecutionProvider (ctx_id=-1) to CUDAExecutionProvider "
                  "with CPU fallback (ctx_id=0); 640x640 fallback detector removed",
        "providers_in_use_gpu_run": manifest["embedder_runtime"]["providers_in_use"],
        "embedding_seconds": {"cpu_run": cpu_meta["seconds"], "gpu_run": manifest["embedder_runtime"]["seconds"]},
        "fallback_detector_usage_in_stage1": compare_embeddings()["cpu_run_det_size_counts"],
        "router_cache": compare_router(),
        "embeddings": compare_embeddings(),
        "results": compare_results(),
        "measurement3_code_on_cpu_embeddings": reproduce_cpu_m3(),
        "kmeans_seed_sensitivity": kmeans_seed_sensitivity(),
        "environment": environment_info(),
    }
    write_json(OUT_JSON, result)
    r = result["results"]
    print("[step0] router bit-identical:", result["router_cache"]["bit_identical"])
    print("[step0] M1 identical:", r["consistency.json"]["identical"], " M2 identical:", r["distribution.json"]["identical"],
          " M3 differing values:", r["partition_comparison.json"]["n_differing_values"])
    print("[step0] M3 code reproduces CPU numbers on CPU embeddings:",
          result["measurement3_code_on_cpu_embeddings"]["reproduces_archived_cpu_numbers_exactly"])
    ks = result["kmeans_seed_sensitivity"]
    print(f"[step0] k-means AUC over {N_KMEANS_SEEDS} seeds: cpu {ks['cpu']['auc_min']:.4f}-{ks['cpu']['auc_max']:.4f}, "
          f"gpu {ks['gpu']['auc_min']:.4f}-{ks['gpu']['auc_max']:.4f}  -> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
