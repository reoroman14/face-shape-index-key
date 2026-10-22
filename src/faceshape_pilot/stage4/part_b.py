# Stage 4, Part B (PILOT4.md): within-identity stability of the stage 1
# router, separately for the four RFW groups.
#
# Landmarks and ratio features are extracted once (extract_rfw.py). Three
# standardiser conditions are applied to the cached ratios, each followed by
# the unchanged softmax:
#
#   B1 pooled    : one robust standardiser fitted on all four groups pooled.
#                  One router, one calibration, mixed population. Primary.
#   B2 per_group : a standardiser fitted inside each group. Sensitivity.
#   B3 lfw       : the stage 1 LFW standardiser, read from
#                  results/pilot/cache/router.npz, applied unchanged.
#
# Per group and condition, m1_consistency.compute (unmodified) gives p_top1,
# Cohen's kappa, chance agreement, the multi-probe table and the per-class
# block, with 95% percentile bootstrap CIs over identities. The class
# distribution and the effective number of classes 1 / sum(share^2) are
# added.
#
# Between groups: per-identity sufficient statistics are rebuilt with the
# same pair sample m1_consistency uses (same SEED, same sample_pairs), and
# identities are resampled independently within each group
# (N_BOOTSTRAP resamples, one independent seed stream per group derived from
# SEED). For each of the six group pairs the difference, its 95% percentile
# CI and a two-sided bootstrap p-value are reported; Holm's correction is
# applied across the six pairs, separately for p_top1 and kappa, in each
# condition. The spread (max group minus min group by point estimate) is
# reported with the CI of the replicate spread max_g - min_g.
#
# Detector failure per group: images without landmarks, identities that fell
# below two images as a result, Wilson CIs, chi-square of failure against
# group, and a bootstrap CI (identities resampled within group) on the
# difference between the highest and the lowest group.
#
# Run:  python -m faceshape_pilot.stage4.part_b

from itertools import combinations

import numpy as np

from faceshape_pilot import config, m1_consistency, router
from faceshape_pilot.common import (
    percentile_ci, read_json, run_header, setup_matplotlib, write_json, COLOR_SHAPE, COLOR_TEXT_2,
)
from faceshape_pilot.stage3.part_a import failure_by_group
from faceshape_pilot.stage4 import extract_rfw, rfw

OUT_JSON = config.STAGE4_DIR / "part_b.json"
OUT_PLOT_STABILITY = config.STAGE4_DIR / "stability_by_group.png"
OUT_PLOT_FAIL = config.STAGE4_DIR / "landmark_failure_by_group.png"
LFW_ROUTER_NPZ = config.CACHE_DIR / "router.npz"

STREAM_PAIRWISE = 41
STREAM_FAILURE = 42
ALPHA = 0.05
CONDITIONS = {
    "B1_pooled": "one standardiser fitted on the ratio features of all four RFW groups pooled (primary)",
    "B2_per_group": "a separate standardiser fitted inside each RFW group (sensitivity)",
    "B3_lfw": "the stage 1 LFW standardiser applied unchanged (transfer)",
}
METRICS = ("p_top1", "cohen_kappa_top1")


# ------------------------------------------------------------- standardisers

def lfw_standardiser() -> dict:
    # router.npz and router_meta.json both record it; they must agree.
    r = np.load(LFW_ROUTER_NPZ)
    stats = {"median": r["standardiser_median"].tolist(), "robust_scale": r["standardiser_scale"].tolist()}
    meta = read_json(extract_rfw.LFW_ROUTER_META)["standardiser"]
    if not (np.allclose(meta["median"], stats["median"], rtol=0, atol=1e-12)
            and np.allclose(meta["robust_scale"], stats["robust_scale"], rtol=0, atol=1e-12)):
        raise RuntimeError("LFW standardiser in router.npz and router_meta.json disagree")
    return stats


def condition_probs(c: dict) -> tuple:
    # Returns ({condition: (n, 7) probs}, {condition: standardiser(s)}).
    ok, ratios, group = c["landmark_ok"], c["ratio_features"], c["group"]
    n = len(ok)
    probs, stds = {}, {}

    def apply(mask, stats, out):
        out[mask] = router.shape_probabilities(router.standardise(ratios[mask], stats))

    p = np.full((n, len(config.CLASSES)), np.nan)
    s = router.fit_standardiser(ratios[ok])
    apply(ok, s, p)
    probs["B1_pooled"], stds["B1_pooled"] = p, s

    p = np.full((n, len(config.CLASSES)), np.nan)
    stds["B2_per_group"] = {}
    for gi, g in enumerate(config.RFW_GROUPS):
        m = ok & (group == gi)
        s = router.fit_standardiser(ratios[m])
        apply(m, s, p)
        stds["B2_per_group"][g] = s
    probs["B2_per_group"] = p

    p = np.full((n, len(config.CLASSES)), np.nan)
    s = lfw_standardiser()
    apply(ok, s, p)
    probs["B3_lfw"], stds["B3_lfw"] = p, s
    return probs, stds


# ----------------------------------------------------------------- statistics

def identity_stats(ds) -> dict:
    # Per-identity sufficient statistics on exactly the pair sample that
    # m1_consistency.compute draws (same generator, same first call).
    rng = np.random.default_rng(config.SEED)
    pairs = m1_consistency.sample_pairs(ds.images_of(), rng)
    n_id, n_cls = len(pairs), len(config.CLASSES)
    same = np.zeros(n_id)
    unordered = np.zeros(n_id)
    conf = np.zeros((n_id, n_cls, n_cls))
    for i, pr in enumerate(pairs):
        a, b = pr[:, 0], pr[:, 1]
        unordered[i] = len(pr)
        same[i] = (ds.top1[a] == ds.top1[b]).sum()
        np.add.at(conf[i], (ds.top1[np.concatenate([a, b])], ds.top1[np.concatenate([b, a])]), 1)
    class_counts = np.zeros((n_id, n_cls))
    np.add.at(class_counts, (ds.identity, ds.top1), 1)
    return {"same": same, "unordered": unordered, "conf": conf, "class_counts": class_counts}


def metrics(st: dict, w: np.ndarray) -> dict:
    # w: (n_id,) or (B, n_id) identity weights.
    cc = w @ st["class_counts"]
    share = cc / cc.sum(axis=-1, keepdims=True)
    return {
        "p_top1": (w @ st["same"]) / (w @ st["unordered"]),
        "cohen_kappa_top1": m1_consistency.kappa_from_confusion(np.tensordot(w, st["conf"], axes=(-1, 0))),
        "effective_n_classes": 1.0 / (share ** 2).sum(axis=-1),
    }


def bootstrap_weights(n_id: int, stream: int, gi: int) -> np.ndarray:
    rng = np.random.default_rng([config.SEED, stream, gi])
    return rng.multinomial(n_id, np.full(n_id, 1.0 / n_id), size=config.N_BOOTSTRAP).astype(np.float64)


def boot_p_value(d: np.ndarray) -> float:
    # Two-sided percentile-bootstrap p-value for H0: difference = 0.
    b = len(d)
    lo = (1 + (d <= 0).sum()) / (b + 1)
    hi = (1 + (d >= 0).sum()) / (b + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(p: np.ndarray, alpha: float = ALPHA):
    # Holm step-down adjusted p-values and rejections.
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj, adj <= alpha


def ci(x) -> dict:
    lo, hi = percentile_ci(np.asarray(x))
    return {"low": float(lo), "high": float(hi)}


def compare_groups(point: dict, reps: dict) -> dict:
    # point[g][metric] scalar; reps[g][metric] (B,) replicates.
    groups = list(point)
    out = {}
    for metric in METRICS:
        pairs = []
        for a, b in combinations(groups, 2):
            d = reps[a][metric] - reps[b][metric]
            c = ci(d)
            pairs.append({
                "group_a": a, "group_b": b,
                "difference_a_minus_b": point[a][metric] - point[b][metric],
                "ci95": c,
                "ci_excludes_zero": bool(c["low"] > 0 or c["high"] < 0),
                "p_value_bootstrap": boot_p_value(d),
            })
        adj, rej = holm(np.array([q["p_value_bootstrap"] for q in pairs]))
        for q, pa, r in zip(pairs, adj, rej):
            q["p_value_holm"] = float(pa)
            q["significant_uncorrected"] = q["p_value_bootstrap"] <= ALPHA
            q["significant_holm"] = bool(r)
        vals = {g: point[g][metric] for g in groups}
        g_max, g_min = max(vals, key=vals.get), min(vals, key=vals.get)
        rep_mat = np.stack([reps[g][metric] for g in groups])
        out[metric] = {
            "pairwise": pairs,
            "n_significant_uncorrected": int(sum(q["significant_uncorrected"] for q in pairs)),
            "n_significant_holm": int(sum(q["significant_holm"] for q in pairs)),
            "spread": {
                "max_group": g_max, "min_group": g_min,
                "max_minus_min": vals[g_max] - vals[g_min],
                "ci95_replicate_max_minus_min": ci(rep_mat.max(0) - rep_mat.min(0)),
                "ci95_fixed_pair_difference": ci(reps[g_max][metric] - reps[g_min][metric]),
            },
        }
    out["notes"] = [
        "Groups are resampled independently: identities with replacement within each group, "
        f"N_BOOTSTRAP={config.N_BOOTSTRAP}, seed stream [SEED, {STREAM_PAIRWISE}, group index]; the pair "
        "sample is the one m1_consistency draws and is held fixed.",
        "p_value_bootstrap = 2 * min(P*(d <= 0), P*(d >= 0)) with the +1 correction; it agrees with "
        "'ci_excludes_zero' up to percentile discreteness.",
        f"Holm step-down across the six pairwise comparisons, alpha = {ALPHA}, separately per metric and "
        "per condition. Only significant_holm supports a claim of a difference.",
        "Spread: point = max group minus min group by point estimate. ci95_replicate_max_minus_min is the "
        "percentile CI of max_g - min_g over replicates (biased upward when groups are equal, so it never "
        "covers zero); ci95_fixed_pair_difference is the CI of the difference for the two groups named.",
    ]
    return out


def detector_failure(c: dict) -> dict:
    group, ident = c["group"], c["identity"]
    decoded, ok = c["decoded"], c["landmark_ok"]
    failed = ~ok  # decode failures counted as failures too; reported separately below
    n_names = len(c["names"])
    res = failure_by_group(group, failed, list(config.RFW_GROUPS))
    per_group = {}
    reps_img, reps_id = {}, {}
    for gi, g in enumerate(config.RFW_GROUPS):
        ids = np.flatnonzero(c["identity_group"] == gi)
        n_img = np.bincount(ident, minlength=n_names)[ids].astype(np.float64)
        n_fail = np.bincount(ident, weights=failed.astype(float), minlength=n_names)[ids]
        n_ok = n_img - n_fail
        dropped = ((n_img >= config.MIN_FACES_PER_PERSON) & (n_ok < config.MIN_FACES_PER_PERSON)).astype(float)
        w = bootstrap_weights(len(ids), STREAM_FAILURE, gi)
        reps_img[g] = (w @ n_fail) / (w @ n_img)
        reps_id[g] = (w @ dropped) / w.sum(axis=1)
        m = group == gi
        per_group[g] = {
            "n_images": int(m.sum()),
            "n_decode_failed": int((m & ~decoded).sum()),
            "n_landmark_failed_after_decode": int((m & decoded & ~ok).sum()),
            "n_failed": int(n_fail.sum()),
            "image_failure_rate": float(n_fail.sum() / n_img.sum()),
            "image_failure_rate_ci95_identity_bootstrap": ci(reps_img[g]),
            "n_identities": int(len(ids)),
            "n_identities_dropped_below_2": int(dropped.sum()),
            "identity_drop_rate": float(dropped.mean()),
            "identity_drop_rate_ci95_identity_bootstrap": ci(reps_id[g]),
        }

    def extreme(key, reps):
        vals = {g: per_group[g][key] for g in per_group}
        hi_g, lo_g = max(vals, key=vals.get), min(vals, key=vals.get)
        d = reps[hi_g] - reps[lo_g]
        c95 = ci(d)
        return {"highest_group": hi_g, "lowest_group": lo_g, "difference": vals[hi_g] - vals[lo_g],
                "ci95": c95, "ci_excludes_zero": bool(c95["low"] > 0 or c95["high"] < 0),
                "p_value_bootstrap": boot_p_value(d)}

    res["per_group"] = per_group
    res["highest_minus_lowest"] = {
        "image_failure_rate": extreme("image_failure_rate", reps_img),
        "identity_drop_rate": extreme("identity_drop_rate", reps_id),
    }
    res["notes"] = [
        "A failure is an image with no landmarks (decode failures included; none expected, counted separately).",
        "groups.*: Wilson CIs and chi-square over images (from stage3.part_a.failure_by_group); these treat "
        "images as independent.",
        "per_group.*_ci95_identity_bootstrap and highest_minus_lowest: identities resampled with replacement "
        f"within each group, N_BOOTSTRAP={config.N_BOOTSTRAP}, seed stream [SEED, {STREAM_FAILURE}, group "
        "index]; this respects clustering of failures within identity.",
        "highest/lowest are chosen by point estimate before bootstrapping, so the CI on their difference is "
        "conditional on that selection.",
    ]
    return res


# ---------------------------------------------------------------------- plots

def plot_stability(res: dict):
    # The figure-level legend is built from the top-left axes and therefore
    # covers both rows, so the marker label stays metric-neutral: the upper
    # row is p_top1 and the lower row is Cohen's kappa. Chance agreement is
    # drawn on the upper row only, because kappa is already chance-corrected.
    plt = setup_matplotlib()
    conds = list(CONDITIONS)
    groups = list(config.RFW_GROUPS)
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2), sharey=True, sharex="row")
    y = np.arange(len(groups))
    for col, cond in enumerate(conds):
        per = res["conditions"][cond]["groups"]
        for row, (metric, label) in enumerate((("p_top1", "p_top1"), ("cohen_kappa_top1", "Cohen's kappa"))):
            ax = axes[row, col]
            v = np.array([per[g][metric] for g in groups])
            lo = np.array([per[g][metric + "_ci95"]["low"] for g in groups])
            hi = np.array([per[g][metric + "_ci95"]["high"] for g in groups])
            ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="o", color=COLOR_SHAPE, ecolor=COLOR_SHAPE,
                        elinewidth=2, capsize=0, markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=2,
                        label="point estimate (95% CI)")
            if metric == "p_top1":
                ch = np.array([per[g]["chance_agreement_from_marginals"] for g in groups])
                ax.plot(ch, y, "|", color=COLOR_TEXT_2, markersize=12, markeredgewidth=2,
                        label="chance agreement, own marginals (upper row only)")
            for yi, vi in zip(y, v):
                ax.annotate(f"{vi:.3f}", (vi, yi), textcoords="offset points", xytext=(0, 8), ha="center",
                            fontsize=8, color=COLOR_TEXT_2)
            ax.set_yticks(y, groups)
            ax.set_ylim(len(groups) - 0.4, -0.6)
            ax.grid(axis="y", visible=False)
            if row == 0:
                ax.set_title(cond.replace("_", " "), loc="left", fontsize=10)
            ax.set_xlabel(label)
    fig.suptitle("Within-identity stability of the stage 1 router per RFW group, by standardiser condition",
                 x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               frameon=False, fontsize=8)
    fig.savefig(OUT_PLOT_STABILITY, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_failures(fail: dict):
    plt = setup_matplotlib()
    groups = list(config.RFW_GROUPS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), sharey=True)
    y = np.arange(len(groups))
    for ax, key, label in ((axes[0], "image_failure_rate", "Images without landmarks"),
                           (axes[1], "identity_drop_rate", "Identities dropped below 2 images")):
        v = np.array([fail["per_group"][g][key] for g in groups])
        lo = np.array([fail["per_group"][g][key + "_ci95_identity_bootstrap"]["low"] for g in groups])
        hi = np.array([fail["per_group"][g][key + "_ci95_identity_bootstrap"]["high"] for g in groups])
        ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="o", color=COLOR_SHAPE, ecolor=COLOR_SHAPE, elinewidth=2,
                    capsize=0, markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=2)
        ax.set_yticks(y, groups)
        ax.set_ylim(len(groups) - 0.4, -0.6)
        ax.set_xlim(left=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"{label}, rate (95% CI, identity bootstrap)")
    fig.suptitle("MediaPipe Face Landmarker failures on RFW", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PLOT_FAIL, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    if not extract_rfw.ROUTER_CACHE.exists():
        raise FileNotFoundError(f"{extract_rfw.ROUTER_CACHE} missing; run faceshape_pilot.stage4.extract_rfw")
    c = dict(np.load(extract_rfw.ROUTER_CACHE))
    meta = read_json(extract_rfw.META)
    probs, stds = condition_probs(c)

    header = run_header("stage4_part_b_rfw_per_group_stability", dataset=rfw.DATASET)
    header["dataset_limitations"] = rfw.DATASET_LIMITATIONS
    header["embedding"] = "not used (Part B runs measurement 1 only; no embeddings computed)"
    result = {
        **header,
        "part": "B",
        "provenance": rfw.provenance(),
        "n_bootstrap": config.N_BOOTSTRAP,
        "seed_streams": {"m1_pairs_and_per_group_ci": config.SEED,
                         "pairwise_comparison": [config.SEED, STREAM_PAIRWISE, "group index"],
                         "detector_failure": [config.SEED, STREAM_FAILURE, "group index"]},
        "router_run": {k: meta[k] for k in ("rfw_counts_verified", "n_images", "n_decode_failed", "n_landmark_ok",
                                            "n_landmark_failed_after_decode", "image_sizes",
                                            "mediapipe_model_sha256", "seconds")},
        "keep_rule": "landmark detection succeeded, and the identity still has >= 2 such images",
        "standardisers": stds,
        "detector_failure": detector_failure(c),
        "top1_agreement_between_conditions": {},
        "conditions": {},
    }

    ok = c["landmark_ok"]
    top1 = {k: p[ok].argmax(1) for k, p in probs.items()}
    for a, b in combinations(CONDITIONS, 2):
        result["top1_agreement_between_conditions"][f"{a}_vs_{b}"] = float((top1[a] == top1[b]).mean())

    for cond, desc in CONDITIONS.items():
        groups, point, reps = {}, {}, {}
        for gi, g in enumerate(config.RFW_GROUPS):
            ds = rfw.RFWSet(c, probs[cond], gi)
            print(f"[part_b] {cond} {g}: {ds.n_images} images, {ds.n_identities} identities", flush=True)
            m1 = m1_consistency.compute(ds)
            st = identity_stats(ds)
            pt = metrics(st, np.ones(ds.n_identities))
            for k in METRICS:
                if not np.isclose(pt[k], m1[k], rtol=0, atol=1e-12):
                    raise RuntimeError(f"{cond} {g}: rebuilt {k} {pt[k]} != m1_consistency {m1[k]}")
            r = metrics(st, bootstrap_weights(ds.n_identities, STREAM_PAIRWISE, gi))
            point[g] = {k: float(pt[k]) for k in METRICS}
            reps[g] = r
            shares = np.bincount(ds.top1, minlength=len(config.CLASSES)) / ds.n_images
            groups[g] = {
                "data_accounting": ds.accounting,
                **m1,
                "class_distribution": {
                    "counts": {k: int(n) for k, n in zip(config.CLASSES, np.bincount(ds.top1, minlength=len(config.CLASSES)))},
                    "shares": dict(zip(config.CLASSES, shares.tolist())),
                    "largest_class": config.CLASSES[int(np.argmax(shares))],
                    "max_share": float(shares.max()),
                },
                "effective_n_classes": float(1.0 / (shares ** 2).sum()),
                "effective_n_classes_ci95": ci(r["effective_n_classes"]),
            }
        result["conditions"][cond] = {
            "description": desc,
            "groups": groups,
            "between_groups": compare_groups(point, reps),
        }
        cmp_ = result["conditions"][cond]["between_groups"]
        print(f"[part_b] {cond}: " + "  ".join(f"{g} p1={point[g]['p_top1']:.3f} k={point[g]['cohen_kappa_top1']:.3f}"
                                               for g in config.RFW_GROUPS)
              + f"  | Holm-significant pairs p1={cmp_['p_top1']['n_significant_holm']} "
                f"kappa={cmp_['cohen_kappa_top1']['n_significant_holm']}", flush=True)

    result["effective_n_classes_ci_note"] = ("effective_n_classes_ci95 uses the pairwise-comparison bootstrap "
                                             "weights (identities resampled within group).")
    write_json(OUT_JSON, result)
    plot_stability(result)
    plot_failures(result["detector_failure"])
    print(f"[part_b] -> {OUT_JSON}")
    return result


if __name__ == "__main__":
    main()
