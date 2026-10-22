# Measurement 3: face shape partition versus a learned partition.
#
# Split: for each identity one image is drawn at random (fixed seed) as the
# held-out probe; every other image forms the gallery. The gallery is
# partitioned into 7 buckets two ways:
#
#   shape  : the router's top-1 shape of each gallery image
#   k-means: k-means (k = 7) fitted on gallery ArcFace embeddings only,
#            which is what an IVF coarse quantiser does
#
# A probe searches its n best buckets, n = 1..7: shape buckets ranked by the
# probe's shape probabilities, k-means buckets ranked by distance from the
# probe embedding to each centroid.
#
#   routing_recall = fraction of probes with at least one gallery mate in a
#                    searched bucket (per-mate recall is also reported)
#   penetration    = mean fraction of the gallery searched
#
# Curves are compared by the area under recall-vs-penetration, with (0, 0)
# and (1, 1) as end points, and by recall at matched penetration. CIs are
# paired percentile bootstraps over probes (one probe per identity).
#
# Linear probe: logistic regression from ArcFace embedding to the router's
# top-1 shape, train/test split grouped by identity so no person appears on
# both sides. Reports balanced accuracy, chance level and a shuffled-label
# control.
#
# Run:  python -m faceshape_pilot.m3_partition

import csv

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from faceshape_pilot import config
from faceshape_pilot.common import (
    AnalysisSet, percentile_ci, run_header, setup_matplotlib, write_json,
    COLOR_KMEANS, COLOR_SHAPE,
)

OUT_JSON = config.RESULTS_DIR / "partition_comparison.json"
OUT_PLOT = config.RESULTS_DIR / "partition_recall_penetration.png"
OUT_SPLIT = config.RESULTS_DIR / "partition_split.csv"
OUT_PROBE_SPLIT = config.RESULTS_DIR / "linear_probe_split.csv"

# Independent random streams for the two splits, both derived from SEED.
STREAM_PROBE_GALLERY = 3
STREAM_PERMUTATION = 4


def make_split(ds: AnalysisSet):
    rng = np.random.default_rng([config.SEED, STREAM_PROBE_GALLERY])
    probes = np.array([rng.choice(g) for g in ds.images_of()])
    is_probe = np.zeros(ds.n_images, dtype=bool)
    is_probe[probes] = True
    return probes, np.flatnonzero(~is_probe)


def routing_tables(order, gallery_bucket, probes, gallery, identity):
    # order: (n_probe, B) bucket ranking per probe.
    # Returns found_any (n_probe, B), mate_frac (n_probe, B), penetration (n_probe, B).
    n_b = order.shape[1]
    sizes = np.bincount(gallery_bucket, minlength=n_b) / len(gallery)
    pen = np.cumsum(sizes[order], axis=1)
    position = np.empty_like(order)
    position[np.arange(len(order))[:, None], order] = np.arange(n_b)[None, :]
    gal_id = identity[gallery]
    found = np.zeros((len(probes), n_b), dtype=bool)
    mate_frac = np.zeros((len(probes), n_b))
    ns = np.arange(1, n_b + 1)
    for i, p in enumerate(probes):
        mates = np.flatnonzero(gal_id == identity[p])
        pos = position[i, gallery_bucket[mates]]
        found[i] = pos.min() < ns
        mate_frac[i] = (pos[:, None] < ns[None, :]).mean(axis=0)
    return found, mate_frac, pen, sizes


def auc(pen, rec):
    # pen, rec: (..., B). Trapezoid area with (0, 0) prepended; the last
    # point is (1, 1) by construction.
    zeros = np.zeros(pen.shape[:-1] + (1,))
    x = np.concatenate([zeros, pen], axis=-1)
    y = np.concatenate([zeros, rec], axis=-1)
    return ((x[..., 1:] - x[..., :-1]) * (y[..., 1:] + y[..., :-1]) / 2).sum(axis=-1)


def curve_summary(found, mate_frac, pen, W):
    tot = W.sum(axis=-1, keepdims=True)
    rec = (W @ found) / tot
    rmate = (W @ mate_frac) / tot
    p = (W @ pen) / tot
    return rec, rmate, p, auc(p, rec)


def partition_comparison(ds: AnalysisSet, kmeans_seed: int = config.SEED, write_split: bool = True):
    probes, gallery = make_split(ds)
    n_b = config.N_BUCKETS

    shape_bucket = ds.top1[gallery]
    shape_order = np.argsort(-ds.probs[probes], axis=1, kind="stable")

    km = KMeans(n_clusters=n_b, n_init=10, random_state=kmeans_seed)
    km_bucket = km.fit_predict(ds.embeddings[gallery])
    d = ((ds.embeddings[probes][:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
    km_order = np.argsort(d, axis=1, kind="stable")

    tabs = {
        "shape": routing_tables(shape_order, shape_bucket, probes, gallery, ds.identity),
        "kmeans": routing_tables(km_order, km_bucket, probes, gallery, ds.identity),
    }

    one = np.ones(len(probes))
    rng = np.random.default_rng([config.SEED, STREAM_PROBE_GALLERY, 1])
    W = rng.multinomial(len(probes), one / len(probes), size=config.N_BOOTSTRAP).astype(np.float64)

    out, boot = {}, {}
    for name, (found, mate_frac, pen, sizes) in tabs.items():
        rec, rmate, p, a = curve_summary(found, mate_frac, pen, one)
        brec, brmate, bp, ba = curve_summary(found, mate_frac, pen, W)
        boot[name] = {"rec": brec, "pen": bp, "auc": ba}
        lo_r, hi_r = percentile_ci(brec)
        lo_p, hi_p = percentile_ci(bp)
        lo_a, hi_a = percentile_ci(ba)
        out[name] = {
            "gallery_bucket_shares": sizes,
            "gallery_largest_bucket_share": sizes.max(),
            "curve": [
                {
                    "buckets_probed": n + 1,
                    "routing_recall": rec[n],
                    "routing_recall_ci95": {"low": lo_r[n], "high": hi_r[n]},
                    "routing_recall_per_mate": rmate[n],
                    "penetration": p[n],
                    "penetration_ci95": {"low": lo_p[n], "high": hi_p[n]},
                }
                for n in range(n_b)
            ],
            "auc_recall_vs_penetration": a,
            "auc_ci95": {"low": lo_a, "high": hi_a},
        }

    diff = boot["shape"]["auc"] - boot["kmeans"]["auc"]
    lo_d, hi_d = percentile_ci(diff)
    s_rec, s_pen = np.array([c["routing_recall"] for c in out["shape"]["curve"]]), np.array([c["penetration"] for c in out["shape"]["curve"]])
    k_rec, k_pen = np.array([c["routing_recall"] for c in out["kmeans"]["curve"]]), np.array([c["penetration"] for c in out["kmeans"]["curve"]])
    kmeans_at_shape_pen = np.interp(s_pen, np.r_[0.0, k_pen], np.r_[0.0, k_rec])
    shape_at_kmeans_pen = np.interp(k_pen, np.r_[0.0, s_pen], np.r_[0.0, s_rec])

    comparison = {
        "auc_shape_minus_kmeans": out["shape"]["auc_recall_vs_penetration"] - out["kmeans"]["auc_recall_vs_penetration"],
        "auc_shape_minus_kmeans_ci95": {"low": lo_d, "high": hi_d},
        "matched_penetration": [
            {
                "buckets_probed_shape": n + 1,
                "penetration": s_pen[n],
                "shape_recall": s_rec[n],
                "kmeans_recall_interpolated": kmeans_at_shape_pen[n],
                "shape_minus_kmeans": s_rec[n] - kmeans_at_shape_pen[n],
            }
            for n in range(n_b)
        ],
        "matched_penetration_at_kmeans_points": [
            {
                "buckets_probed_kmeans": n + 1,
                "penetration": k_pen[n],
                "kmeans_recall": k_rec[n],
                "shape_recall_interpolated": shape_at_kmeans_pen[n],
                "shape_minus_kmeans": shape_at_kmeans_pen[n] - k_rec[n],
            }
            for n in range(n_b)
        ],
        "random_partition_auc": 0.5,
        "notes": [
            "The AUC joins (0, 0) to each partition's first operating point by a straight line. "
            "If one partition's first point lies at high penetration (unbalanced buckets), the "
            "AUC credits it only with that interpolated segment, so the AUC difference should be "
            "read together with both matched-penetration tables, which compare the curves at "
            "each partition's real operating points.",
        ],
    }

    if write_split:
        with open(OUT_SPLIT, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["path", "identity", "role", "shape_top1", "kmeans_bucket"])
            km_of = dict(zip(gallery.tolist(), km_bucket.tolist()))
            for i in range(ds.n_images):
                role = "gallery" if i in km_of else "probe"
                wr.writerow([ds.paths[i], ds.names[ds.identity[i]], role, config.CLASSES[ds.top1[i]], km_of.get(i, "")])

    split = {
        "protocol": "one random probe per identity, all remaining images form the gallery",
        "seed_stream": [config.SEED, STREAM_PROBE_GALLERY],
        "n_probes": len(probes),
        "n_gallery": len(gallery),
        "file": OUT_SPLIT.name,
    }
    kmeans_info = {"k": n_b, "n_init": 10, "random_state": kmeans_seed, "fit_on": "gallery embeddings only",
                   "inertia": float(km.inertia_)}
    return split, kmeans_info, out, comparison


def linear_probe(ds: AnalysisSet, write_split: bool = True):
    X, y, groups = ds.embeddings, ds.top1, ds.identity
    gss = GroupShuffleSplit(n_splits=1, test_size=config.PROBE_TEST_FRACTION, random_state=config.SEED)
    tr, te = next(gss.split(X, y, groups))

    def model():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced", random_state=config.SEED),
        )

    clf = model().fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    labels = np.arange(len(config.CLASSES))
    bal = balanced_accuracy_score(y[te], pred)
    per_class_recall = recall_score(y[te], pred, labels=labels, average=None, zero_division=np.nan)

    rng = np.random.default_rng([config.SEED, STREAM_PERMUTATION])
    y_perm = rng.permutation(y[tr])
    bal_perm = balanced_accuracy_score(y[te], model().fit(X[tr], y_perm).predict(X[te]))

    present = np.unique(y[te])
    if write_split:
        _write_probe_split(ds, te)

    return _probe_result(ds, tr, te, groups, bal, bal_perm, present, per_class_recall, y, pred, labels)


def _write_probe_split(ds, te):
    with open(OUT_PROBE_SPLIT, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["path", "identity", "role"])
        role = np.full(ds.n_images, "train", dtype=object)
        role[te] = "test"
        for i in range(ds.n_images):
            wr.writerow([ds.paths[i], ds.names[ds.identity[i]], role[i]])


def _probe_result(ds, tr, te, groups, bal, bal_perm, present, per_class_recall, y, pred, labels):
    return {
        "target": "router top-1 shape (not a human label)",
        "model": "StandardScaler + multinomial LogisticRegression(C=1.0, class_weight='balanced')",
        "split": {
            "method": "GroupShuffleSplit by identity",
            "test_fraction_of_identities": config.PROBE_TEST_FRACTION,
            "random_state": config.SEED,
            "n_train_images": len(tr),
            "n_test_images": len(te),
            "n_train_identities": int(len(np.unique(groups[tr]))),
            "n_test_identities": int(len(np.unique(groups[te]))),
            "file": OUT_PROBE_SPLIT.name,
        },
        "balanced_accuracy": bal,
        "chance_balanced_accuracy": 1.0 / len(present),
        "shuffled_label_control_balanced_accuracy": bal_perm,
        "per_class_recall": {c: per_class_recall[i] for i, c in enumerate(config.CLASSES)},
        "confusion_test_rows_true_cols_pred": confusion_matrix(y[te], pred, labels=labels),
        "notes": [
            "The target is the router's own output. If the router is inconsistent across "
            "images of one person, an identity-centred embedding cannot predict it well "
            "even if shape is encoded, so read this together with p_top1.",
        ],
    }


def plot(parts, path):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.plot([0, 1], [0, 1], ":", color="#8a8984", linewidth=1.2, label="Random partition")
    for name, color, marker, label in (
        ("shape", COLOR_SHAPE, "o", "Face shape (7 classes)"),
        ("kmeans", COLOR_KMEANS, "s", "k-means on ArcFace (k = 7)"),
    ):
        c = parts[name]["curve"]
        x = np.r_[0.0, [e["penetration"] for e in c]]
        y = np.r_[0.0, [e["routing_recall"] for e in c]]
        lo = np.r_[0.0, [e["routing_recall_ci95"]["low"] for e in c]]
        hi = np.r_[0.0, [e["routing_recall_ci95"]["high"] for e in c]]
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, "-", marker=marker, color=color,
                label=f"{label}, AUC {parts[name]['auc_recall_vs_penetration']:.3f}",
                markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        ax.annotate("1 bucket", (x[1], y[1]), textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Penetration (mean fraction of gallery searched)")
    ax.set_ylabel("Routing recall (probe's mate in a searched bucket)")
    ax.set_title("Routing recall vs penetration on LFW, held-out probes", fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(ds: AnalysisSet = None):
    ds = ds or AnalysisSet()
    split, kmeans_info, parts, comparison = partition_comparison(ds)
    probe = linear_probe(ds)
    result = {
        **run_header("3_shape_vs_learned_partition"),
        "data_accounting": ds.accounting,
        "split": split,
        "kmeans": kmeans_info,
        "partitions": parts,
        "comparison": comparison,
        "linear_probe": probe,
    }
    write_json(OUT_JSON, result)
    plot(parts, OUT_PLOT)
    print(f"[m3] AUC shape={parts['shape']['auc_recall_vs_penetration']:.4f} "
          f"kmeans={parts['kmeans']['auc_recall_vs_penetration']:.4f} "
          f"probe_bal_acc={probe['balanced_accuracy']:.4f}  -> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
