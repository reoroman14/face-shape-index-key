# Stage 3, Part A (PILOT3.md): the geometric router across demographic
# groups on FairFace. No identities are needed or used.
#
# For each attribute (race, gender, age) and each standardiser variant
# (frozen = stage 1 LFW statistics, primary; refit = FairFace statistics,
# sensitivity):
#
#   - the seven-class top-1 distribution within each group, with bootstrap
#     95% CIs (images resampled with replacement, B = config.N_BOOTSTRAP);
#   - chi-square test of independence, Cramer's V and bias-corrected V
#     (Bergsma 2013), with bootstrap CIs, and a permutation null for V
#     (group labels shuffled) showing the value V takes at this sample size
#     when there is no association;
#   - adjusted standardised residuals per cell and the lift
#     pi(shape | group) / pi(shape);
#   - the per-group ceiling on candidate-set reduction 1 / max_k pi_k;
#   - landmark detection failure rate per group (Wilson 95% CI) and a
#     chi-square test of failure against the attribute.
#
# Race and gender are also analysed on adults only (age 20 and over): the
# age mix differs across race groups and the rules were written for adult
# faces.
#
# Reverse direction: how well the router output predicts the group. Models
# are fitted on the FairFace train split and evaluated once on the
# validation split (the official split), images with landmarks only:
#
#   shape_label         : top-1 label alone; the classifier is the train-set
#                         table P(group | label) with add-one smoothing.
#                         Accuracy uses argmax P(group | label); balanced
#                         accuracy uses argmax P(group | label) / P(group),
#                         the rule that is optimal when groups are weighted
#                         equally.
#   shape_probabilities : the seven router probabilities, multinomial
#                         logistic regression.
#   ratio_features      : the four ratio features the router is built on,
#                         gradient-boosted trees; shows what the underlying
#                         geometry carries beyond the label.
#
# Metrics on the test split, with bootstrap 95% CIs over test images:
# accuracy, balanced accuracy, macro one-vs-rest ROC AUC, and out-of-sample
# information gain (cross-entropy of the train prior minus cross-entropy of
# the model, in bits, also as a fraction of the prior cross-entropy).
#
# Run:  python -m faceshape_pilot.stage3.part_a

import csv

import numpy as np
from scipy import stats

from faceshape_pilot import config
from faceshape_pilot.common import environment_info, percentile_ci, read_json, setup_matplotlib, write_json
from faceshape_pilot.common import COLOR_GRID, COLOR_SHAPE, COLOR_TEXT, COLOR_TEXT_2
from faceshape_pilot.router import router_spec
from faceshape_pilot.stage3 import extract_fairface, fairface

OUT_JSON = config.STAGE3_DIR / "part_a.json"
OUT_IMAGES_CSV = config.STAGE3_DIR / "fairface_router_outputs.csv"
OUT_PLOT_SHARES = config.STAGE3_DIR / "shape_by_group.png"
OUT_PLOT_FAIL = config.STAGE3_DIR / "landmark_failure_by_group.png"

STREAM_BOOT = 31
STREAM_PERM = 32
STREAM_REVERSE = 33
N_PERMUTATIONS = 1000
ADULT_MIN_AGE_INDEX = 3  # "20-29" and older
SMOOTHING = 1.0
VARIANTS = ("frozen", "refit")

ATTRIBUTES = {
    "race": config.FAIRFACE_RACES,
    "gender": config.FAIRFACE_GENDERS,
    "age": config.FAIRFACE_AGES,
}
# (name, attribute, adults only)
ANALYSES = (
    ("race", "race", False),
    ("race_adults", "race", True),
    ("gender", "gender", False),
    ("gender_adults", "gender", True),
    ("age", "age", False),
)


# ---------------------------------------------------------------- statistics

def chi2_from_table(t: np.ndarray):
    n = t.sum()
    row = t.sum(1, keepdims=True)
    col = t.sum(0, keepdims=True)
    keep_r = row[:, 0] > 0
    keep_c = col[0] > 0
    t, row, col = t[keep_r][:, keep_c], row[keep_r], col[:, keep_c]
    e = row * col / n
    chi2 = float(((t - e) ** 2 / e).sum())
    return chi2, t.shape


def cramers_v(t: np.ndarray) -> float:
    chi2, (r, c) = chi2_from_table(t)
    return float(np.sqrt(chi2 / (t.sum() * (min(r, c) - 1))))


def cramers_v_corrected(t: np.ndarray) -> float:
    # Bergsma (2013) bias correction.
    chi2, (r, c) = chi2_from_table(t)
    n = t.sum()
    phi2 = max(0.0, chi2 / n - (c - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    cc = c - (c - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / min(rc - 1, cc - 1)))


def cohen_band(v: float, df_star: int) -> str:
    # Cohen (1988) conventions for w, divided by sqrt(min(r, c) - 1).
    s = np.sqrt(df_star)
    if v < 0.1 / s:
        return "negligible"
    if v < 0.3 / s:
        return "small"
    if v < 0.5 / s:
        return "medium"
    return "large"


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def table_of(g: np.ndarray, k: np.ndarray, n_g: int, n_k: int, w=None) -> np.ndarray:
    return np.bincount(g * n_k + k, weights=w, minlength=n_g * n_k).reshape(n_g, n_k)


# ------------------------------------------------------------- cross-tabs

def crosstab(group: np.ndarray, top1: np.ndarray, names, rng_boot, rng_perm) -> dict:
    n_g, n_k = len(names), len(config.CLASSES)
    present = np.unique(group)
    t = table_of(group, top1, n_g, n_k).astype(np.float64)
    n = int(t.sum())
    chi2, p, dof, expected = stats.chi2_contingency(t[present], correction=False)
    row = t.sum(1)
    col = t.sum(0)
    shares = np.divide(t, row[:, None], out=np.full_like(t, np.nan), where=row[:, None] > 0)
    pooled = col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        e = row[:, None] * col[None, :] / n
        adj_resid = (t - e) / np.sqrt(e * (1 - row[:, None] / n) * (1 - col[None, :] / n))
        lift = shares / pooled[None, :]
    df_star = min(len(present), n_k) - 1

    # Bootstrap over images.
    m = len(group)
    b_v, b_vc, b_shares, b_ceiling, b_pooled_ceiling = [], [], [], [], []
    for _ in range(config.N_BOOTSTRAP):
        w = np.bincount(rng_boot.integers(0, m, m), minlength=m).astype(np.float64)
        tb = table_of(group, top1, n_g, n_k, w)
        b_v.append(cramers_v(tb))
        b_vc.append(cramers_v_corrected(tb))
        rb = tb.sum(1, keepdims=True)
        sb = np.divide(tb, rb, out=np.full_like(tb, np.nan), where=rb > 0)
        b_shares.append(sb)
        b_ceiling.append(1.0 / sb.max(1))
        b_pooled_ceiling.append(1.0 / (tb.sum(0) / tb.sum()).max())
    b_shares = np.array(b_shares)
    s_lo, s_hi = percentile_ci(b_shares)
    c_lo, c_hi = percentile_ci(np.array(b_ceiling))

    # Permutation null for V.
    v_obs = cramers_v(t)
    null = np.array([cramers_v(table_of(rng_perm.permutation(group), top1, n_g, n_k).astype(np.float64))
                     for _ in range(N_PERMUTATIONS)])

    groups = {}
    for gi, name in enumerate(names):
        if row[gi] == 0:
            continue
        kmax = int(np.argmax(shares[gi]))
        groups[name] = {
            "n_with_landmarks": int(row[gi]),
            "counts": {c: int(t[gi, k]) for k, c in enumerate(config.CLASSES)},
            "shares": {c: float(shares[gi, k]) for k, c in enumerate(config.CLASSES)},
            "shares_ci95": {c: [float(s_lo[gi, k]), float(s_hi[gi, k])] for k, c in enumerate(config.CLASSES)},
            "lift_vs_pooled": {c: float(lift[gi, k]) for k, c in enumerate(config.CLASSES)},
            "adjusted_residuals": {c: float(adj_resid[gi, k]) for k, c in enumerate(config.CLASSES)},
            "largest_class": config.CLASSES[kmax],
            "max_share": float(shares[gi, kmax]),
            "ceiling_1_over_max_share": float(1.0 / shares[gi, kmax]),
            "ceiling_ci95": [float(c_lo[gi]), float(c_hi[gi])],
        }
    pc_lo, pc_hi = percentile_ci(np.array(b_pooled_ceiling))
    v_lo, v_hi = percentile_ci(np.array(b_v))
    vc_lo, vc_hi = percentile_ci(np.array(b_vc))
    return {
        "n_images": n,
        "n_groups": int(len(present)),
        "pooled": {
            "shares": {c: float(pooled[k]) for k, c in enumerate(config.CLASSES)},
            "largest_class": config.CLASSES[int(np.argmax(pooled))],
            "ceiling_1_over_max_share": float(1.0 / pooled.max()),
            "ceiling_ci95": [float(pc_lo), float(pc_hi)],
        },
        "groups": groups,
        "independence": {
            "chi2": float(chi2),
            "dof": int(dof),
            "p_value": float(p),
            "min_expected_count": float(expected.min()),
            "cramers_v": v_obs,
            "cramers_v_ci95": [float(v_lo), float(v_hi)],
            "cramers_v_bias_corrected": cramers_v_corrected(t),
            "cramers_v_bias_corrected_ci95": [float(vc_lo), float(vc_hi)],
            "min_dim_minus_1": int(df_star),
            "cohen_band": cohen_band(v_obs, df_star),
            "cohen_thresholds_small_medium_large": [0.1 / np.sqrt(df_star), 0.3 / np.sqrt(df_star),
                                                    0.5 / np.sqrt(df_star)],
            "permutation_null_v": {
                "n_permutations": N_PERMUTATIONS,
                "median": float(np.median(null)),
                "p95": float(np.percentile(null, 95)),
                "p99": float(np.percentile(null, 99)),
                "max": float(null.max()),
                "p_value": float((1 + (null >= v_obs).sum()) / (1 + N_PERMUTATIONS)),
            },
        },
    }


def failure_by_group(group: np.ndarray, failed: np.ndarray, names) -> dict:
    out = {}
    t = np.zeros((len(names), 2))
    for gi, name in enumerate(names):
        m = group == gi
        n, k = int(m.sum()), int(failed[m].sum())
        if n == 0:
            continue
        t[gi] = (n - k, k)
        lo, hi = wilson(k, n)
        out[name] = {"n_images": n, "n_failed": k, "failure_rate": k / n, "failure_rate_ci95_wilson": [lo, hi]}
    present = t.sum(1) > 0
    res = {"groups": out, "n_images": int(len(group)), "n_failed": int(failed.sum()),
           "failure_rate": float(failed.mean())}
    if t[present, 1].sum() > 0:
        chi2, p, dof, expected = stats.chi2_contingency(t[present], correction=False)
        res["independence"] = {"chi2": float(chi2), "dof": int(dof), "p_value": float(p),
                               "min_expected_count": float(expected.min()), "cramers_v": cramers_v(t[present])}
    return res


# ------------------------------------------------------- reverse direction

def _metrics(y: np.ndarray, scores: np.ndarray, pred_acc: np.ndarray, pred_bal: np.ndarray,
             prior: np.ndarray, n_cls: int) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    if n_cls == 2:
        auc = roc_auc_score(y, scores[:, 1])
    else:
        auc = roc_auc_score(y, scores, multi_class="ovr", average="macro", labels=list(range(n_cls)))
    eps = 1e-12
    ce_model = float(-np.mean(np.log2(np.clip(scores[np.arange(len(y)), y], eps, None))))
    ce_prior = float(-np.mean(np.log2(prior[y])))
    return {
        "accuracy": float((pred_acc == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred_bal)),
        "macro_ovr_auc": float(auc),
        "information_gain_bits": ce_prior - ce_model,
        "information_gain_fraction": (ce_prior - ce_model) / ce_prior,
    }


def _evaluate(y, scores, pred_acc, pred_bal, prior, n_cls, rng) -> dict:
    point = _metrics(y, scores, pred_acc, pred_bal, prior, n_cls)
    boots = {k: [] for k in point}
    n = len(y)
    for _ in range(config.N_BOOTSTRAP):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < n_cls:
            continue
        for k, v in _metrics(y[i], scores[i], pred_acc[i], pred_bal[i], prior, n_cls).items():
            boots[k].append(v)
    out = {}
    for k, v in point.items():
        lo, hi = percentile_ci(np.array(boots[k]))
        out[k] = v
        out[k + "_ci95"] = [float(lo), float(hi)]
    out["n_bootstrap_used"] = len(boots["accuracy"])
    from sklearn.metrics import recall_score
    out["recall_per_group_balanced_rule"] = recall_score(y, pred_bal, labels=list(range(n_cls)), average=None,
                                                         zero_division=0).tolist()
    return out


def reverse(y_tr, y_te, lab_tr, lab_te, prob_tr, prob_te, rat_tr, rat_te, names, rng) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_cls, n_k = len(names), len(config.CLASSES)
    prior = np.bincount(y_tr, minlength=n_cls) / len(y_tr)
    majority = int(np.argmax(prior))
    res = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_prior": dict(zip(names, prior.tolist())),
        "baselines": {
            "accuracy_majority_class": float((y_te == majority).mean()),
            "majority_class": names[majority],
            "balanced_accuracy_chance": 1.0 / n_cls,
            "macro_ovr_auc_chance": 0.5,
            "information_gain_bits_chance": 0.0,
        },
        "predictors": {},
    }

    # shape_label: P(group | label) from the train table, add-one smoothing.
    t = table_of(lab_tr, y_tr, n_k, n_cls).astype(np.float64) + SMOOTHING
    p_g_given_k = t / t.sum(1, keepdims=True)
    scores = p_g_given_k[lab_te]
    res["predictors"]["shape_label"] = {
        "model": f"train-set table P(group | top-1 label), add-{SMOOTHING:g} smoothing",
        "p_group_given_label_train": {c: dict(zip(names, p_g_given_k[k].tolist())) for k, c in enumerate(config.CLASSES)},
        "lift_p_group_given_label_over_prior": {c: dict(zip(names, (p_g_given_k[k] / prior).tolist()))
                                                for k, c in enumerate(config.CLASSES)},
        **_evaluate(y_te, scores, scores.argmax(1), (scores / prior).argmax(1), prior, n_cls, rng),
    }

    # shape_probabilities: multinomial logistic regression on the 7 probabilities.
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    lr_bal = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    lr.fit(prob_tr, y_tr)
    lr_bal.fit(prob_tr, y_tr)
    scores = lr.predict_proba(prob_te)
    res["predictors"]["shape_probabilities"] = {
        "model": "standardise + multinomial logistic regression (lbfgs, L2, C=1); balanced accuracy from a "
                 "class_weight='balanced' fit",
        **_evaluate(y_te, scores, scores.argmax(1), lr_bal.predict(prob_te), prior, n_cls, rng),
    }

    # ratio_features: gradient-boosted trees on the four ratio features.
    hgb = HistGradientBoostingClassifier(random_state=config.SEED)
    hgb_bal = HistGradientBoostingClassifier(random_state=config.SEED, class_weight="balanced")
    hgb.fit(rat_tr, y_tr)
    hgb_bal.fit(rat_tr, y_tr)
    scores = hgb.predict_proba(rat_te)
    res["predictors"]["ratio_features"] = {
        "model": f"HistGradientBoostingClassifier (sklearn defaults, random_state={config.SEED}); balanced "
                 "accuracy from a class_weight='balanced' fit",
        **_evaluate(y_te, scores, scores.argmax(1), hgb_bal.predict(rat_te), prior, n_cls, rng),
    }
    return res


# ------------------------------------------------------------------ plots

def plot_shares(res: dict):
    import matplotlib.colors as mcolors

    plt = setup_matplotlib()
    cmap = mcolors.LinearSegmentedColormap.from_list("div", [COLOR_SHAPE, "#f0efec", "#e34948"])
    panels = [("race", "Race (annotated)"), ("gender", "Gender (annotated)"), ("age", "Age band (annotated)")]
    heights = [len(res[a]["groups"]) + 1 for a, _ in panels]
    fig, axes = plt.subplots(3, 1, figsize=(8.6, 0.42 * sum(heights) + 2.6),
                             gridspec_kw={"height_ratios": heights})
    lim = 1.0
    im = None
    for ax, (a, title) in zip(axes, panels):
        g = res[a]["groups"]
        names = list(g) + ["All"]
        shares = np.array([[g[n]["shares"][c] for c in config.CLASSES] for n in g]
                          + [[res[a]["pooled"]["shares"][c] for c in config.CLASSES]])
        pooled = shares[-1]
        loglift = np.log2(shares / pooled[None, :])
        im = ax.imshow(np.clip(loglift, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
        for i in range(shares.shape[0]):
            for k in range(shares.shape[1]):
                ax.text(k, i, f"{shares[i, k]:.2f}", ha="center", va="center", fontsize=8,
                        color=COLOR_TEXT, fontweight="bold" if i == shares.shape[0] - 1 else "normal")
        ax.set_yticks(range(len(names)), names)
        ax.set_xticks(range(len(config.CLASSES)), config.CLASSES if a == "age" else [""] * len(config.CLASSES))
        ax.axhline(len(names) - 1.5, color=COLOR_TEXT_2, linewidth=1)
        ax.grid(False)
        v = res[a]["independence"]
        ax.set_title(f"{title}: Cramer's V {v['cramers_v']:.3f} "
                     f"({v['cramers_v_ci95'][0]:.3f} to {v['cramers_v_ci95'][1]:.3f})",
                     loc="left", fontsize=10, color=COLOR_TEXT)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.025, pad=0.06, aspect=40)
    cb.set_label("log2(share in group / pooled share), clipped at +/-1; cell text = share of the group's faces")
    cb.outline.set_visible(False)
    fig.suptitle("Top-1 face shape within each FairFace group (frozen stage 1 router)", x=0.02, ha="left",
                 fontsize=11, color=COLOR_TEXT)
    fig.savefig(OUT_PLOT_SHARES, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_failures(fail: dict):
    plt = setup_matplotlib()
    panels = [("race", "Race"), ("gender", "Gender"), ("age", "Age band")]
    heights = [len(fail[a]["groups"]) for a, _ in panels]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 0.34 * sum(heights) + 2.2),
                             gridspec_kw={"height_ratios": heights}, sharex=True)
    for ax, (a, title) in zip(axes, panels):
        g = fail[a]["groups"]
        names = list(g)
        r = np.array([g[n]["failure_rate"] for n in names])
        lo = np.array([g[n]["failure_rate_ci95_wilson"][0] for n in names])
        hi = np.array([g[n]["failure_rate_ci95_wilson"][1] for n in names])
        y = np.arange(len(names))
        ax.errorbar(r, y, xerr=[r - lo, hi - r], fmt="o", color=COLOR_SHAPE, ecolor=COLOR_SHAPE,
                    elinewidth=2, capsize=0, markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=2)
        ax.axvline(fail[a]["failure_rate"], color=COLOR_TEXT_2, linewidth=1, linestyle="--")
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("Landmark detection failure rate (95% Wilson CI); dashed = all images")
    fig.suptitle("MediaPipe Face Landmarker failures on FairFace", x=0.02, ha="left", fontsize=11)
    fig.savefig(OUT_PLOT_FAIL, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- main

def write_image_csv(c):
    with open(OUT_IMAGES_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "shard", "row", "race", "gender", "age", "landmark_ok", "top1_frozen", "top1_refit"]
                    + [f"p_frozen_{k}" for k in config.CLASSES] + ["L", "F", "J", "A_deg"])
        for i in range(len(c["split"])):
            ok = bool(c["landmark_ok"][i])
            wr.writerow([c["split"][i], c["shard"][i], int(c["row"][i]), config.FAIRFACE_RACES[c["race"][i]],
                         config.FAIRFACE_GENDERS[c["gender"][i]], config.FAIRFACE_AGES[c["age"][i]], ok,
                         config.CLASSES[int(np.argmax(c["probs_frozen"][i]))] if ok else "",
                         config.CLASSES[int(np.argmax(c["probs_refit"][i]))] if ok else ""]
                        + ([f"{v:.6f}" for v in c["probs_frozen"][i]] if ok else [""] * len(config.CLASSES))
                        + ([f"{v:.6f}" for v in c["ratio_features"][i]] if ok else [""] * 4))


def main():
    if not extract_fairface.ROUTER_CACHE.exists():
        extract_fairface.run()
    c = dict(np.load(extract_fairface.ROUTER_CACHE))
    meta = read_json(extract_fairface.META)
    ok = c["landmark_ok"]
    decoded = c["decoded"]
    adult = c["age"] >= ADULT_MIN_AGE_INDEX

    result = {
        "part": "A",
        "seed": config.SEED,
        "seed_streams": {"bootstrap": [config.SEED, STREAM_BOOT], "permutation": [config.SEED, STREAM_PERM],
                         "reverse_bootstrap": [config.SEED, STREAM_REVERSE]},
        "n_bootstrap": config.N_BOOTSTRAP,
        "dataset": read_json(fairface.OUT_JSON),
        "router": router_spec(),
        "router_run": {k: meta[k] for k in ("n_images", "n_decode_failed", "n_landmark_ok",
                                            "n_landmark_failed_after_decode", "image_sizes",
                                            "standardiser_frozen_lfw", "standardiser_refit_fairface",
                                            "mediapipe_model_sha256", "seconds")},
        "top1_definition": "argmax of the seven router probabilities",
        "adult_definition": f"age band index >= {ADULT_MIN_AGE_INDEX} ({config.FAIRFACE_AGES[ADULT_MIN_AGE_INDEX]} "
                            "and older)",
        "landmark_failure": {},
        "crosstabs": {},
        "reverse": {},
        "top1_agreement_frozen_vs_refit": float((c["probs_frozen"][ok].argmax(1)
                                                 == c["probs_refit"][ok].argmax(1)).mean()),
    }

    failed = decoded & ~ok
    for a, names in ATTRIBUTES.items():
        m = decoded
        result["landmark_failure"][a] = failure_by_group(c[a][m], failed[m], names)
    result["landmark_failure"]["race_adults"] = failure_by_group(c["race"][decoded & adult], failed[decoded & adult],
                                                                 config.FAIRFACE_RACES)

    for variant in VARIANTS:
        top1 = np.full(len(ok), -1)
        top1[ok] = c[f"probs_{variant}"][ok].argmax(1)
        rng_boot = np.random.default_rng([config.SEED, STREAM_BOOT])
        rng_perm = np.random.default_rng([config.SEED, STREAM_PERM])
        result["crosstabs"][variant] = {}
        for name, a, adults_only in ANALYSES:
            m = ok & (adult if adults_only else True)
            print(f"[part_a] crosstab {variant} {name}", flush=True)
            result["crosstabs"][variant][name] = crosstab(c[a][m], top1[m], ATTRIBUTES[a], rng_boot, rng_perm)

        rng_rev = np.random.default_rng([config.SEED, STREAM_REVERSE])
        result["reverse"][variant] = {}
        for name, a, adults_only in ANALYSES:
            m = ok & (adult if adults_only else True)
            tr = m & (c["split"] == "train")
            te = m & (c["split"] == "validation")
            print(f"[part_a] reverse {variant} {name}", flush=True)
            result["reverse"][variant][name] = reverse(
                c[a][tr], c[a][te], top1[tr], top1[te], c[f"probs_{variant}"][tr], c[f"probs_{variant}"][te],
                c["ratio_features"][tr], c["ratio_features"][te], list(ATTRIBUTES[a]), rng_rev)

    result["environment"] = environment_info()
    write_json(OUT_JSON, result)
    write_image_csv(c)
    plot_shares(result["crosstabs"]["frozen"])
    plot_failures(result["landmark_failure"])
    print(f"[part_a] -> {OUT_JSON}")


if __name__ == "__main__":
    main()
