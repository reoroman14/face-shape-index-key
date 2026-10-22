# Measurement 2: class distribution of the router's top-1 shape.
#
# Over images: top-1 shape per image.
# Over identities: each identity gets the argmax of its mean probability
# vector across its images.
#
#   max_reduction = 1 / (share of the largest class)
#   entropy       = Shannon entropy of the distribution, in bits
#
# Run:  python -m faceshape_pilot.m2_distribution

import numpy as np

from faceshape_pilot import config
from faceshape_pilot.common import AnalysisSet, run_header, setup_matplotlib, write_json, COLOR_SHAPE, COLOR_TEXT_2

OUT_JSON = config.RESULTS_DIR / "distribution.json"
OUT_PLOT = config.RESULTS_DIR / "distribution.png"


def summarise(labels: np.ndarray) -> dict:
    n_cls = len(config.CLASSES)
    counts = np.bincount(labels, minlength=n_cls)
    share = counts / counts.sum()
    nz = share[share > 0]
    entropy = float(-(nz * np.log2(nz)).sum())
    largest = int(share.argmax())
    return {
        "n": int(counts.sum()),
        "counts": {c: int(counts[i]) for i, c in enumerate(config.CLASSES)},
        "shares": {c: share[i] for i, c in enumerate(config.CLASSES)},
        "largest_class": config.CLASSES[largest],
        "largest_class_share": share[largest],
        "max_reduction": 1.0 / share[largest],
        "entropy_bits": entropy,
        "entropy_max_bits": float(np.log2(n_cls)),
        "entropy_normalised": entropy / float(np.log2(n_cls)),
        "effective_number_of_classes": 2.0**entropy,
    }


def compute(ds: AnalysisSet) -> dict:
    groups = ds.images_of()
    id_labels = np.array([ds.probs[g].mean(axis=0).argmax() for g in groups])
    mean_prob = ds.probs.mean(axis=0)
    return {
        "over_images": summarise(ds.top1),
        "over_identities": summarise(id_labels),
        "mean_probability_mass_per_class": {c: mean_prob[i] for i, c in enumerate(config.CLASSES)},
        "notes": [
            "Identity label = argmax of the mean probability vector over that identity's images.",
            "The router standardises its features over this image set (no identity labels used). "
            "Class balance is therefore partly a consequence of where the prototypes sit in "
            "standard-score space, not only of the faces. A different, equally defensible "
            "prototype placement would give a different distribution.",
        ],
    }


def plot(result, path):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, key, title in (
        (axes[0], "over_images", "Over images"),
        (axes[1], "over_identities", "Over identities"),
    ):
        s = result[key]
        shares = np.array([s["shares"][c] for c in config.CLASSES])
        x = np.arange(len(config.CLASSES))
        ax.bar(x, shares, width=0.7, color=COLOR_SHAPE, edgecolor="none")
        for xi, v in zip(x, shares):
            ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=COLOR_TEXT_2)
        ax.set_xticks(x, config.CLASSES, rotation=30, ha="right")
        ax.set_title(
            f"{title} (n={s['n']}): max_reduction {s['max_reduction']:.2f}x, H {s['entropy_bits']:.2f} bits",
            fontsize=10,
        )
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of top-1 shape")
    fig.suptitle("Face shape class distribution on LFW (landmark router, top-1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(ds: AnalysisSet = None):
    ds = ds or AnalysisSet()
    result = {**run_header("2_class_distribution"), "data_accounting": ds.accounting, **compute(ds)}
    write_json(OUT_JSON, result)
    plot(result, OUT_PLOT)
    s = result["over_images"]
    print(f"[m2] largest={s['largest_class']} share={s['largest_class_share']:.4f} "
          f"max_reduction={s['max_reduction']:.3f} H={s['entropy_bits']:.3f}  -> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
