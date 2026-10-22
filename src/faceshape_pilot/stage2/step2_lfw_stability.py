# Step 2 (PILOT2.md): identity consistency of the CNN router on LFW.
#
# The trained CNN runs over the identical LFW analysis set used in stage 1
# (same images, same identities). Consistency is computed by the stage 1
# code (m1_consistency.compute) with the same seed, so the within-identity
# pairs and the bootstrap over identities are identical; only the
# probability vectors differ.
#
# The routers have different numbers of classes, so they are compared on
# Cohen's kappa (agreement corrected for chance from the class marginals),
# not on raw p_top1.
#
# Run:  python -m faceshape_pilot.stage2.step2_lfw_stability

import copy
import csv

import numpy as np

from faceshape_pilot import config, m1_consistency
from faceshape_pilot.common import AnalysisSet, environment_info, read_json, setup_matplotlib, write_json, \
    COLOR_KMEANS, COLOR_SHAPE
from faceshape_pilot.data import read_rgb
from faceshape_pilot.stage2 import cnn, faces
from faceshape_pilot.stage2.step1_train_cnn import OUT_JSON as STEP1_JSON, model_path

OUT_JSON = config.STAGE2_DIR / "consistency_cnn_lfw.json"
OUT_PROBS_CSV = config.STAGE2_DIR / "lfw_cnn_probs.csv"
OUT_PLOT = config.STAGE2_DIR / "consistency_routers_topk.png"
PROBS_CACHE = config.STAGE2_CACHE / "lfw_cnn_probs.npz"


def lfw_probs(ds: AnalysisSet, selected: str):
    if PROBS_CACHE.exists():
        c = np.load(PROBS_CACHE)
        if np.array_equal(c["paths"], ds.paths) and str(c["backbone"]) == selected:
            return c["probs"], c["ok"], c["providers"].item() if c["providers"].shape == () else None
    detector = faces.FaceDetector()
    providers = detector.providers_in_use()
    model = cnn.load_trained(selected, len(config.KAGGLE_CLASSES), model_path(selected))
    n = ds.n_images
    ok = np.zeros(n, dtype=bool)
    crops = np.zeros((n, faces.CROP_SIZE, faces.CROP_SIZE, 3), dtype=np.uint8)
    for i, rel in enumerate(ds.paths):
        rgb = read_rgb(str(config.DATA_DIR / rel))
        b, _ = detector.detect(rgb)
        if b is not None:
            crops[i] = faces.extract_crop(rgb, faces.crop_square(b))
            ok[i] = True
    probs = np.full((n, len(config.KAGGLE_CLASSES)), np.nan)
    probs[ok] = cnn.predict_proba(model, crops[ok])
    np.savez_compressed(PROBS_CACHE, paths=ds.paths, probs=probs, ok=ok, backbone=selected,
                        providers=np.array(str(providers)))
    return probs, ok, str(providers)


def main():
    step1 = read_json(STEP1_JSON)
    selected = step1["selected_backbone"]
    ds = AnalysisSet()
    probs, ok, providers = lfw_probs(ds, selected)
    if not ok.all():
        # The pairing protocol requires the identical image set; a failure
        # here is recorded and stops the step rather than silently changing
        # the pairs.
        write_json(OUT_JSON, {"error": "face detection failed on LFW analysis images",
                              "failed_paths": ds.paths[~ok].tolist()})
        raise RuntimeError(f"{(~ok).sum()} LFW analysis images had no face; see {OUT_JSON}")

    with open(OUT_PROBS_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "identity"] + [f"p_{c}" for c in config.KAGGLE_CLASSES])
        for i in range(ds.n_images):
            wr.writerow([ds.paths[i], ds.names[ds.identity[i]]] + [f"{v:.6f}" for v in probs[i]])

    ds_cnn = copy.copy(ds)
    ds_cnn.probs = probs
    ds_cnn.top1 = probs.argmax(axis=1)
    cnn_res = m1_consistency.compute(ds_cnn, classes=config.KAGGLE_CLASSES)
    geo = read_json(m1_consistency.OUT_JSON)
    if cnn_res["n_unordered_pairs"] != geo["n_unordered_pairs"] or cnn_res["n_identities"] != geo["n_identities"]:
        raise RuntimeError("Pairing differs from stage 1")

    share = np.bincount(ds_cnn.top1, minlength=len(config.KAGGLE_CLASSES)) / ds.n_images
    table = [
        {"router": "geometric (stage 1)", "classes": len(config.CLASSES), "own_test_accuracy": None,
         "own_test_accuracy_note": "not applicable (rule-based, no labelled test set)",
         "p_top1": geo["p_top1"], "chance": geo["chance_agreement_from_marginals"],
         "kappa": geo["cohen_kappa_top1"], "kappa_ci95": geo["cohen_kappa_top1_ci95"],
         "source": "results/pilot/consistency.json"},
        {"router": f"CNN ({selected}, Kaggle-trained)", "classes": len(config.KAGGLE_CLASSES),
         "own_test_accuracy": step1["test"]["accuracy_on_detected"],
         "own_test_accuracy_counting_detection_failures_as_wrong":
             step1["test"]["accuracy_counting_detection_failures_as_wrong"],
         "p_top1": cnn_res["p_top1"], "chance": cnn_res["chance_agreement_from_marginals"],
         "kappa": cnn_res["cohen_kappa_top1"], "kappa_ci95": cnn_res["cohen_kappa_top1_ci95"],
         "source": f"results/stage2/{OUT_JSON.name}"},
    ]
    result = {
        "seed": config.SEED,
        "router": {"backbone": selected, "classes": config.KAGGLE_CLASSES,
                   "model_file": model_path(selected).relative_to(config.REPO_ROOT).as_posix(),
                   "detector_providers_in_use": providers,
                   "preprocessing": "same SCRFD crop rule as Kaggle training (faces.py)"},
        "data_accounting": ds.accounting,
        "pairing": "identical to stage 1: m1_consistency.sample_pairs with the same seed over the same analysis set",
        "lfw_top1_distribution": {c: share[k] for k, c in enumerate(config.KAGGLE_CLASSES)},
        "comparison_table": table,
        **cnn_res,
        "probabilities_file": OUT_PROBS_CSV.name,
        "environment": environment_info(),
    }
    write_json(OUT_JSON, result)
    plot(geo, cnn_res, selected, OUT_PLOT)
    print(f"[step2] CNN p_top1={cnn_res['p_top1']:.4f} kappa={cnn_res['cohen_kappa_top1']:.4f} "
          f"(geometric kappa {geo['cohen_kappa_top1']:.4f})  -> {OUT_JSON}")
    return result


def plot(geo, cnn_res, selected, path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.plot([0, 1], [0, 1], ":", color="#8a8984", linewidth=1.2, label="Random partition")
    for res, color, marker, label in ((geo, COLOR_SHAPE, "o", "Geometric rules, 7 classes"),
                                      (cnn_res, COLOR_KMEANS, "s", f"CNN ({selected}), 5 classes")):
        mp = res["multi_probe"]
        x = np.r_[0.0, [m["mean_gallery_fraction_searched"] for m in mp]]
        y = np.r_[0.0, [m["p_topk"] for m in mp]]
        ax.plot(x, y, "-", marker=marker, color=color, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                label=f"{label}, kappa {res['cohen_kappa_top1']:.3f}")
        ax.annotate("k=1", (x[1], y[1]), textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean fraction of images in the probed classes")
    ax.set_ylabel("p_topk (gallery top-1 within probe top-k)")
    ax.set_title("Within-identity label consistency on LFW, two routers", fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
