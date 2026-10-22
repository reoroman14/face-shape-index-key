# Writes results/pilot/VERDICT.md from the three result files.
#
# The decision rule is PILOT.md's. PILOT.md states several thresholds as
# "about" or in words ("close to", "clearly dominates", "high accuracy").
# They are operationalised by the constants below, in one place, so they can
# be inspected and changed. The verdict is a mechanical, provisional reading
# of the numbers; the author makes the final call.
#
# Run:  python -m faceshape_pilot.verdict

from faceshape_pilot import config
from faceshape_pilot.common import read_json, environment_info
from faceshape_pilot.m1_consistency import OUT_JSON as M1_JSON
from faceshape_pilot.m2_distribution import OUT_JSON as M2_JSON
from faceshape_pilot.m3_partition import OUT_JSON as M3_JSON

OUT_MD = config.RESULTS_DIR / "VERDICT.md"

# From PILOT.md.
GREEN_P_TOP1 = 0.85
RED_P_TOP1 = 0.60
GREEN_MAX_CLASS_SHARE = 0.35

# Operationalisations chosen for this script (not stated numerically in PILOT.md).
# Shape curve "at or above" k-means: AUC(shape) - AUC(k-means) >= -AUC_TIE.
AUC_TIE = 0.01
# k-means "clearly dominates": AUC difference below -AUC_CLEAR and the 95%
# bootstrap CI of the difference lies entirely below zero.
AUC_CLEAR = 0.05
# Linear probe "high accuracy": balanced accuracy at or above this value
# (chance for 7 classes is about 0.14).
PROBE_HIGH_BAL_ACC = 0.70


def decide(m1, m2, m3):
    p1 = m1["p_top1"]
    share = m2["over_images"]["largest_class_share"]
    d = m3["comparison"]["auc_shape_minus_kmeans"]
    d_hi = m3["comparison"]["auc_shape_minus_kmeans_ci95"]["high"]
    bal = m3["linear_probe"]["balanced_accuracy"]

    shape_at_or_above = d >= -AUC_TIE
    kmeans_clearly_dominates = d < -AUC_CLEAR and d_hi < 0
    probe_high = bal >= PROBE_HIGH_BAL_ACC

    red_reasons = []
    if p1 < RED_P_TOP1:
        red_reasons.append(f"p_top1 = {p1:.3f} is below {RED_P_TOP1}")
    if kmeans_clearly_dominates:
        red_reasons.append(f"k-means clearly dominates (AUC difference {d:+.3f}, CI upper {d_hi:+.3f})")
    if probe_high:
        red_reasons.append(f"linear probe recovers shape with balanced accuracy {bal:.3f} >= {PROBE_HIGH_BAL_ACC}")

    green = p1 > GREEN_P_TOP1 and share < GREEN_MAX_CLASS_SHARE and shape_at_or_above
    if red_reasons:
        call, reasons = "Red", red_reasons
    elif green:
        call, reasons = "Green", ["all three Green conditions hold"]
    else:
        call, reasons = "Amber", []
        if p1 <= GREEN_P_TOP1:
            reasons.append(f"p_top1 = {p1:.3f} is between {RED_P_TOP1} and {GREEN_P_TOP1}")
        if share >= GREEN_MAX_CLASS_SHARE:
            reasons.append(f"largest class share {share:.3f} is not below {GREEN_MAX_CLASS_SHARE}")
        if not shape_at_or_above:
            reasons.append(f"shape curve is below k-means but not clearly dominated (AUC difference {d:+.3f})")

    checks = [
        ("p_top1 (identity consistency)", f"{p1:.3f}",
         f"[{m1['p_top1_ci95']['low']:.3f}, {m1['p_top1_ci95']['high']:.3f}]",
         f"Green > {GREEN_P_TOP1}; Red < {RED_P_TOP1}"),
        ("Largest class share (images)", f"{share:.3f}", "",
         f"Green < {GREEN_MAX_CLASS_SHARE}"),
        ("AUC shape - AUC k-means", f"{d:+.3f}",
         f"[{m3['comparison']['auc_shape_minus_kmeans_ci95']['low']:+.3f}, {d_hi:+.3f}]",
         f"Green >= -{AUC_TIE}; Red < -{AUC_CLEAR} with CI below 0"),
        ("Linear probe balanced accuracy", f"{bal:.3f}", "",
         f"Red >= {PROBE_HIGH_BAL_ACC} (chance {m3['linear_probe']['chance_balanced_accuracy']:.3f})"),
    ]
    return call, reasons, checks


def render(m1, m2, m3):
    call, reasons, checks = decide(m1, m2, m3)
    acc = m1["data_accounting"]
    lines = [
        "# Pilot verdict",
        "",
        f"**Provisional call: {call}.** Derived mechanically from the result files by "
        "`python -m faceshape_pilot.verdict`; the author makes the final call.",
        "",
        "Reasons:",
        *[f"- {r}" for r in reasons],
        "",
        "## Criteria",
        "",
        "| Quantity | Value | 95% CI | Threshold |",
        "|---|---|---|---|",
        *[f"| {a} | {b} | {c} | {t} |" for a, b, c, t in checks],
        "",
        "Thresholds marked in PILOT.md as approximate or verbal were operationalised as follows "
        "(constants at the top of `src/faceshape_pilot/verdict.py`): "
        f"\"at or above\" = AUC difference >= -{AUC_TIE}; "
        f"\"clearly dominates\" = AUC difference < -{AUC_CLEAR} with the bootstrap CI below zero; "
        f"\"high accuracy\" = balanced accuracy >= {PROBE_HIGH_BAL_ACC}.",
        "",
        "## Shape versus k-means at real operating points",
        "",
        "The AUC row above joins (0, 0) to each curve's first operating point by a straight line. "
        "The tables below compare the curves only where a partition actually has an operating "
        "point, interpolating the other curve.",
        "",
        "At k-means operating points:",
        "",
        "| k-means buckets probed | Penetration | k-means recall | Shape recall (interpolated) | Shape - k-means |",
        "|---|---|---|---|---|",
        *[f"| {e['buckets_probed_kmeans']} | {e['penetration']:.3f} | {e['kmeans_recall']:.3f} | "
          f"{e['shape_recall_interpolated']:.3f} | {e['shape_minus_kmeans']:+.3f} |"
          for e in m3["comparison"]["matched_penetration_at_kmeans_points"]],
        "",
        "At shape operating points:",
        "",
        "| Shape buckets probed | Penetration | Shape recall | k-means recall (interpolated) | Shape - k-means |",
        "|---|---|---|---|---|",
        *[f"| {e['buckets_probed_shape']} | {e['penetration']:.3f} | {e['shape_recall']:.3f} | "
          f"{e['kmeans_recall_interpolated']:.3f} | {e['shape_minus_kmeans']:+.3f} |"
          for e in m3["comparison"]["matched_penetration"]],
        "",
        "Gallery bucket shares: shape "
        + ", ".join(f"{v:.3f}" for v in m3["partitions"]["shape"]["gallery_bucket_shares"])
        + "; k-means "
        + ", ".join(f"{v:.3f}" for v in m3["partitions"]["kmeans"]["gallery_bucket_shares"])
        + ".",
        "",
        "## Per-class identity consistency",
        "",
        "| Class | Share of images | Ordered pairs (probe in class) | p_top1 given probe class | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for c in config.CLASSES:
        pc = m1["per_class"][c]
        val = pc["p_top1_given_probe_class"]
        lo, hi = pc["ci95"]["low"], pc["ci95"]["high"]
        lines.append(
            f"| {c} | {pc['share_of_images']:.3f} | {pc['n_ordered_pairs_with_probe_in_class']} | "
            f"{'' if val is None else f'{val:.3f}'} | "
            f"{'' if lo is None else f'[{lo:.3f}, {hi:.3f}]'} |"
        )
    lines += [
        "",
        f"Cohen's kappa on pair labels: {m1['cohen_kappa_top1']:.3f} "
        f"[{m1['cohen_kappa_top1_ci95']['low']:.3f}, {m1['cohen_kappa_top1_ci95']['high']:.3f}]; "
        f"chance agreement from the class marginals: {m1['chance_agreement_from_marginals']:.3f}.",
        "",
        "## Other key numbers",
        "",
        "Multi-probe (from `consistency.json`):",
        "",
        "| k | p_topk | Mean fraction of gallery searched |",
        "|---|---|---|",
        *[f"| {m['k']} | {m['p_topk']:.3f} | {m['mean_gallery_fraction_searched']:.3f} |"
          for m in m1["multi_probe"] if m["k"] <= 3],
        "",
        f"Class distribution over images: largest class {m2['over_images']['largest_class']}, "
        f"max_reduction {m2['over_images']['max_reduction']:.2f}x, entropy "
        f"{m2['over_images']['entropy_bits']:.2f} of {m2['over_images']['entropy_max_bits']:.2f} bits. "
        f"Over identities: largest class {m2['over_identities']['largest_class']}, "
        f"max_reduction {m2['over_identities']['max_reduction']:.2f}x.",
        "",
        "Routing recall at one bucket probed: "
        f"shape {m3['partitions']['shape']['curve'][0]['routing_recall']:.3f} at penetration "
        f"{m3['partitions']['shape']['curve'][0]['penetration']:.3f}; "
        f"k-means {m3['partitions']['kmeans']['curve'][0]['routing_recall']:.3f} at penetration "
        f"{m3['partitions']['kmeans']['curve'][0]['penetration']:.3f}.",
        "",
        "## Data",
        "",
        f"LFW (funneled), identities with at least two images: {acc['lfw_images_ge2']} images, "
        f"{acc['lfw_identities_ge2']} identities. Landmark failures {acc['landmark_failed']}, "
        f"embedding failures {acc['embedding_failed']}. Analysis set after removing failures and "
        f"identities left with fewer than two images: {acc['analysis_images']} images, "
        f"{acc['analysis_identities']} identities. Seed {config.SEED}. "
        f"Probe/gallery split: {m3['split']['n_probes']} probes, {m3['split']['n_gallery']} gallery images "
        f"(`{m3['split']['file']}`).",
        "",
        "## Limitations",
        "",
        f"- {config.DATASET_LIMITATIONS}",
        "- The router is one explicit rule set (prototypes in `router.py`) fixed before the run. "
        "Its identity consistency and class balance are properties of this rule set on LFW; a "
        "different rule set or a learned router (CNN, transformer) may behave differently.",
        "- Router features are standardised over the pilot images, so class balance is partly a "
        "design consequence; see the note in `distribution.json`.",
        "- The linear probe predicts the router's label, not a human label. Low router consistency "
        "caps what any identity-centred embedding can predict, so a low probe score does not by "
        "itself show that the embedding lacks shape information.",
        "- No human face shape labels are used anywhere in this pilot.",
        "",
        f"Generated {environment_info()['utc_time']}.",
        "",
    ]
    return "\n".join(lines)


def main():
    m1, m2, m3 = read_json(M1_JSON), read_json(M2_JSON), read_json(M3_JSON)
    OUT_MD.write_text(render(m1, m2, m3), encoding="utf-8")
    print(f"[verdict] {decide(m1, m2, m3)[0]}  -> {OUT_MD}")


if __name__ == "__main__":
    main()
