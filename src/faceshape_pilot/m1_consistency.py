# Measurement 1: identity consistency of the face shape router.
#
# For every identity, all within-identity image pairs are formed; identities
# with more than MAX_PAIRS_PER_IDENTITY pairs are subsampled (without
# replacement, fixed seed) so a few large identities do not dominate.
#
#   p_top1 = fraction of pairs whose two images receive the same top-1 shape
#   p_topk = fraction of ordered pairs (probe, gallery) where the gallery
#            image's top-1 shape is within the probe's top-k shapes
#
# p_topk is asymmetric, so every pair is scored in both directions. Cost of
# the multi-probe policy is reported as classes searched (exactly k) and as
# the mean fraction of the gallery that falls in those k classes.
# Confidence intervals are percentile bootstraps over identities.
#
# Run:  python -m faceshape_pilot.m1_consistency

from itertools import combinations

import numpy as np

from faceshape_pilot import config
from faceshape_pilot.common import AnalysisSet, percentile_ci, run_header, setup_matplotlib, write_json, COLOR_SHAPE

OUT_JSON = config.RESULTS_DIR / "consistency.json"
OUT_PLOT = config.RESULTS_DIR / "consistency_topk.png"


def sample_pairs(groups, rng):
    # Returns a list of (n_i, 2) arrays of image indices, one per identity.
    out = []
    for idx in groups:
        pairs = np.array(list(combinations(idx.tolist(), 2)), dtype=np.int64)
        if len(pairs) > config.MAX_PAIRS_PER_IDENTITY:
            pick = rng.choice(len(pairs), size=config.MAX_PAIRS_PER_IDENTITY, replace=False)
            pairs = pairs[np.sort(pick)]
        out.append(pairs)
    return out


def kappa_from_confusion(cm):
    n = cm.sum(axis=(-1, -2))
    po = np.trace(cm, axis1=-2, axis2=-1) / n
    pe = (cm.sum(axis=-1) * cm.sum(axis=-2)).sum(axis=-1) / n**2
    return (po - pe) / (1 - pe)


def compute(ds: AnalysisSet, classes=config.CLASSES):
    rng = np.random.default_rng(config.SEED)
    groups = ds.images_of()
    pairs_per_id = sample_pairs(groups, rng)
    n_cls = len(classes)
    ks = np.arange(1, n_cls + 1)

    rank = np.argsort(-ds.probs, axis=1, kind="stable")
    # position[i, c] = rank of class c in image i's distribution (0 = top-1)
    position = np.empty_like(rank)
    position[np.arange(len(rank))[:, None], rank] = np.arange(n_cls)[None, :]
    image_share = np.bincount(ds.top1, minlength=n_cls) / ds.n_images
    # gallery fraction searched when probing image i with its top-k classes
    gal_frac = np.cumsum(image_share[rank], axis=1)

    n_id = len(pairs_per_id)
    unordered = np.zeros(n_id)
    same = np.zeros(n_id)
    hits = np.zeros((n_id, n_cls))
    frac = np.zeros((n_id, n_cls))
    ordered = np.zeros(n_id)
    cls_n = np.zeros((n_id, n_cls))
    cls_same = np.zeros((n_id, n_cls))
    conf = np.zeros((n_id, n_cls, n_cls))
    for i, pr in enumerate(pairs_per_id):
        a, b = pr[:, 0], pr[:, 1]
        unordered[i] = len(pr)
        same[i] = (ds.top1[a] == ds.top1[b]).sum()
        probe = np.concatenate([a, b])
        gal = np.concatenate([b, a])
        ordered[i] = len(probe)
        pos = position[probe, ds.top1[gal]]
        hits[i] = (pos[:, None] < ks[None, :]).sum(axis=0)
        frac[i] = gal_frac[probe].sum(axis=0)
        pc, gc = ds.top1[probe], ds.top1[gal]
        cls_n[i] = np.bincount(pc, minlength=n_cls)
        cls_same[i] = np.bincount(pc[pc == gc], minlength=n_cls)
        np.add.at(conf[i], (pc, gc), 1)

    def stats(w):
        # w: (n_id,) or (B, n_id) identity weights
        p1 = (w @ same) / (w @ unordered)
        pk = (w @ hits) / (w @ ordered)[..., None]
        fr = (w @ frac) / (w @ ordered)[..., None]
        with np.errstate(invalid="ignore", divide="ignore"):
            pc = (w @ cls_same) / (w @ cls_n)
        cm = np.tensordot(w, conf, axes=(-1, 0))
        return p1, pk, fr, pc, kappa_from_confusion(cm)

    p1, pk, fr, pc, kap = stats(np.ones(n_id))
    W = rng.multinomial(n_id, np.full(n_id, 1.0 / n_id), size=config.N_BOOTSTRAP).astype(np.float64)
    bp1, bpk, bfr, bpc, bkap = stats(W)

    def ci(x):
        lo, hi = percentile_ci(x)
        return {"low": lo, "high": hi}

    conf_total = conf.sum(axis=0)
    per_class = {}
    for c, name in enumerate(classes):
        lo, hi = percentile_ci(bpc[:, c])
        per_class[name] = {
            "n_ordered_pairs_with_probe_in_class": int(cls_n[:, c].sum()),
            "p_top1_given_probe_class": pc[c],
            "ci95": {"low": lo, "high": hi},
            "share_of_images": image_share[c],
        }

    chance_agreement = float((image_share**2).sum())
    return {
        "n_identities": n_id,
        "n_unordered_pairs": int(unordered.sum()),
        "n_ordered_pairs": int(ordered.sum()),
        "max_pairs_per_identity": config.MAX_PAIRS_PER_IDENTITY,
        "p_top1": p1,
        "p_top1_ci95": ci(bp1),
        "cohen_kappa_top1": kap,
        "cohen_kappa_top1_ci95": ci(bkap),
        "chance_agreement_from_marginals": chance_agreement,
        "multi_probe": [
            {
                "k": int(k),
                "p_topk": pk[j],
                "p_topk_ci95": {"low": percentile_ci(bpk[:, j])[0], "high": percentile_ci(bpk[:, j])[1]},
                "mean_classes_searched": float(k),
                "mean_gallery_fraction_searched": fr[j],
                "mean_gallery_fraction_searched_ci95": {
                    "low": percentile_ci(bfr[:, j])[0],
                    "high": percentile_ci(bfr[:, j])[1],
                },
            }
            for j, k in enumerate(ks)
        ],
        "per_class": per_class,
        "pair_confusion_ordered": {
            "classes": classes,
            "rows_probe_cols_gallery": conf_total.astype(int),
        },
        "notes": [
            "p_top1 is computed over unordered pairs; p_topk, per-class values and kappa over ordered pairs (both directions).",
            "p_top1_given_probe_class equals the specific (positive) agreement for that class because both directions are counted.",
            "mean_classes_searched is exactly k under fixed top-k multi-probe; mean_gallery_fraction_searched is the cost in images.",
            "CIs: 95% percentile bootstrap over identities, pair sample held fixed.",
        ],
    }


def plot(result, path):
    plt = setup_matplotlib()
    mp = result["multi_probe"]
    k = np.array([m["k"] for m in mp])
    p = np.array([m["p_topk"] for m in mp])
    lo = np.array([m["p_topk_ci95"]["low"] for m in mp])
    hi = np.array([m["p_topk_ci95"]["high"] for m in mp])
    fr = np.array([m["mean_gallery_fraction_searched"] for m in mp])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, x, xlabel in (
        (axes[0], k, "Mean classes searched (k)"),
        (axes[1], fr, "Mean fraction of gallery searched"),
    ):
        ax.fill_between(x, lo, hi, color=COLOR_SHAPE, alpha=0.15, linewidth=0)
        ax.plot(x, p, "-o", color=COLOR_SHAPE, label="Face shape router, top-k multi-probe")
        for xi, pi, ki in zip(x, p, k):
            if ki <= 3:
                ax.annotate(f"k={ki}: {pi:.3f}", (xi, pi), textcoords="offset points", xytext=(6, -12), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.02)
    axes[1].plot([0, 1], [0, 1], ":", color="#8a8984", linewidth=1.2, label="Random partition (recall = fraction searched)")
    axes[1].set_xlim(0, 1.02)
    axes[0].set_xticks(k)
    axes[0].set_ylabel("p_topk (gallery top-1 within probe top-k)")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle(
        f"Identity consistency on LFW: p_top1 = {result['p_top1']:.3f}, "
        f"{result['n_unordered_pairs']} pairs, {result['n_identities']} identities",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(ds: AnalysisSet = None):
    ds = ds or AnalysisSet()
    result = {**run_header("1_identity_consistency"), "data_accounting": ds.accounting, **compute(ds)}
    write_json(OUT_JSON, result)
    plot(result, OUT_PLOT)
    print(f"[m1] p_top1={result['p_top1']:.4f}  kappa={result['cohen_kappa_top1']:.4f}  -> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
