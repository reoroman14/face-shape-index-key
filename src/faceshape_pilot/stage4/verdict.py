# Writes results/stage4/VERDICT.md from results/stage4/part_b.json and
# results/stage4/step0.json. Every number in the file is read from those
# files; nothing is typed in. The finding sentences are chosen by fixed
# rules stated in the file, not by hand.
#
# Run:  python -m faceshape_pilot.stage4.verdict

import datetime as _dt

from faceshape_pilot import config
from faceshape_pilot.common import read_json
from faceshape_pilot.stage4 import extract_rfw, part_b

OUT_MD = config.STAGE4_DIR / "VERDICT.md"
METRIC_LABELS = {"p_top1": "p_top1", "cohen_kappa_top1": "Cohen's kappa"}


def f3(x):
    return "" if x is None else f"{x:.3f}"


def f4(x):
    return "" if x is None else f"{x:.4f}"


def ci(d, fmt=f3):
    return "" if d is None else f"({fmt(d['low'])}, {fmt(d['high'])})"


def pval(p):
    return "" if p is None else f"{p:.3g}"


def stability_table(cond: dict) -> list:
    lines = ["| Group | Images | Identities | Pairs | p_top1 (95% CI) | Kappa (95% CI) | Chance agreement "
             "| Largest class (share) | Effective classes (95% CI) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for g, d in cond["groups"].items():
        cd = d["class_distribution"]
        lines.append(f"| {g} | {d['data_accounting']['analysis_images']} | {d['n_identities']} | "
                     f"{d['n_unordered_pairs']} | {f3(d['p_top1'])} {ci(d['p_top1_ci95'])} | "
                     f"{f3(d['cohen_kappa_top1'])} {ci(d['cohen_kappa_top1_ci95'])} | "
                     f"{f3(d['chance_agreement_from_marginals'])} | {cd['largest_class']} ({f3(cd['max_share'])}) | "
                     f"{d['effective_n_classes']:.2f} ({d['effective_n_classes_ci95']['low']:.2f}, "
                     f"{d['effective_n_classes_ci95']['high']:.2f}) |")
    return lines


def shares_table(cond: dict) -> list:
    lines = ["| Group | " + " | ".join(config.CLASSES) + " |", "|" + "---|" * (len(config.CLASSES) + 1)]
    for g, d in cond["groups"].items():
        lines.append(f"| {g} | " + " | ".join(f3(d["class_distribution"]["shares"][c]) for c in config.CLASSES) + " |")
    return lines


def multiprobe_table(cond: dict) -> list:
    ks = [m["k"] for m in next(iter(cond["groups"].values()))["multi_probe"]]
    lines = ["| Group | " + " | ".join(f"k={k}: p_topk / gallery" for k in ks[:3]) + " |",
             "|" + "---|" * 4]
    for g, d in cond["groups"].items():
        lines.append(f"| {g} | " + " | ".join(f"{f3(m['p_topk'])} / {f3(m['mean_gallery_fraction_searched'])}"
                                              for m in d["multi_probe"][:3]) + " |")
    return lines


def pairwise_table(bg: dict) -> list:
    lines = ["| Metric | Pair | Difference (95% CI) | CI excludes 0 | p (bootstrap) | p (Holm) "
             "| Significant, uncorrected | Significant, Holm |",
             "|---|---|---|---|---|---|---|---|"]
    for metric in part_b.METRICS:
        for q in bg[metric]["pairwise"]:
            lines.append(f"| {METRIC_LABELS[metric]} | {q['group_a']} - {q['group_b']} | "
                         f"{q['difference_a_minus_b']:+.3f} {ci(q['ci95'])} | {'yes' if q['ci_excludes_zero'] else 'no'} | "
                         f"{pval(q['p_value_bootstrap'])} | {pval(q['p_value_holm'])} | "
                         f"{'yes' if q['significant_uncorrected'] else 'no'} | "
                         f"{'**yes**' if q['significant_holm'] else 'no'} |")
    return lines


def spread_lines(bg: dict) -> list:
    out = []
    for metric in part_b.METRICS:
        s = bg[metric]["spread"]
        out.append(f"- {METRIC_LABELS[metric]} spread: {s['max_group']} minus {s['min_group']} = "
                   f"{s['max_minus_min']:+.3f}, CI of that pair's difference {ci(s['ci95_fixed_pair_difference'])}, "
                   f"CI of replicate max-min {ci(s['ci95_replicate_max_minus_min'])}. "
                   f"Holm-significant pairs: {bg[metric]['n_significant_holm']} of 6 "
                   f"(uncorrected: {bg[metric]['n_significant_uncorrected']} of 6).")
    return out


def stability_finding(cond: dict) -> str:
    bg = cond["between_groups"]
    sig = [(m, q) for m in part_b.METRICS for q in bg[m]["pairwise"] if q["significant_holm"]]
    if not sig:
        s1 = bg["p_top1"]["spread"]
        return (f"**Flat.** No pairwise difference in p_top1 or kappa survives Holm correction. The largest "
                f"p_top1 gap is {s1['max_group']} minus {s1['min_group']} = {s1['max_minus_min']:+.3f} "
                f"{ci(s1['ci95_fixed_pair_difference'])}. Under this condition, per-group measurement does not "
                "support the claim that routing failures concentrate in one RFW group.")
    parts = [f"{METRIC_LABELS[m]} {q['group_a']} - {q['group_b']} = {q['difference_a_minus_b']:+.3f} "
             f"{ci(q['ci95'])}, Holm p = {pval(q['p_value_holm'])}" for m, q in sig]
    return ("**Uneven.** Holm-corrected pairwise differences: " + "; ".join(parts) + ". "
            "Under this condition a person in the lower-stability group is less likely to be routed to the "
            "class of their own other images than a person in the higher-stability group.")


def failure_section(f: dict) -> list:
    lines = ["| Group | Images | No landmarks | Image failure rate (95% CI) | Identities | Dropped below 2 "
             "| Identity drop rate (95% CI) |",
             "|---|---|---|---|---|---|---|"]
    for g, d in f["per_group"].items():
        lines.append(f"| {g} | {d['n_images']} | {d['n_failed']} | {f4(d['image_failure_rate'])} "
                     f"{ci(d['image_failure_rate_ci95_identity_bootstrap'], f4)} | {d['n_identities']} | "
                     f"{d['n_identities_dropped_below_2']} | {f4(d['identity_drop_rate'])} "
                     f"{ci(d['identity_drop_rate_ci95_identity_bootstrap'], f4)} |")
    lines.append("")
    for key, label in (("image_failure_rate", "Image failure rate"), ("identity_drop_rate", "Identity drop rate")):
        h = f["highest_minus_lowest"][key]
        lines.append(f"- {label}, highest minus lowest: {h['highest_group']} minus {h['lowest_group']} = "
                     f"{h['difference']:+.4f} {ci(h['ci95'], f4)}, bootstrap p = {pval(h['p_value_bootstrap'])}.")
    if "independence" in f:
        i = f["independence"]
        lines.append(f"- Chi-square of failure against group (images independent): chi2 = {i['chi2']:.2f}, "
                     f"dof = {i['dof']}, p = {pval(i['p_value'])}.")
    h = f["highest_minus_lowest"]["image_failure_rate"]
    lines.append("")
    if f["n_failed"] == 0:
        lines.append("**Flat.** MediaPipe landmarked every RFW image; there is no detector-level difference "
                     "between groups.")
    elif h["ci_excludes_zero"]:
        lines.append(f"**Uneven.** The face mesh fails more often on {h['highest_group']} than on "
                     f"{h['lowest_group']} images ({h['difference']:+.4f} {ci(h['ci95'], f4)}). This is a "
                     "channel of unequal treatment upstream of the shape label, invisible in any accuracy number.")
    else:
        lines.append(f"**Flat.** The highest-minus-lowest difference in detector failure rate "
                     f"({h['difference']:+.4f} {ci(h['ci95'], f4)}) has a CI that includes zero. The detector "
                     "does not fail measurably more often on any one RFW group.")
    return lines


def main():
    r = read_json(part_b.OUT_JSON)
    s0 = read_json(extract_rfw.STEP0_JSON) if extract_rfw.STEP0_JSON.exists() else None
    p = r["provenance"]
    conds = r["conditions"]
    lines = [
        "# Stage 4, Part B: per-group within-identity stability on RFW",
        "",
        f"Generated {_dt.date.today().isoformat()} from `results/stage4/part_b.json`. Seed {r['seed']}, "
        f"{r['n_bootstrap']} bootstrap resamples over identities.",
        "",
        "## Provenance",
        "",
        f"- Dataset: {p['source_dataset']}; derived from {p['derived_from']}.",
        f"- Access: {p['access']}.",
        f"- Obtained from {p['obtained_from']}. Downloaded: {p['downloaded']}.",
        f"- Verification: {p['verification']}.",
        f"- Maintainer readme: `{p['maintainer_readme']}` (sha256 `{p['maintainer_readme_sha256']}`).",
        "",
        "## Findings",
        "",
        f"**B1, pooled RFW standardiser (primary).** {stability_finding(conds['B1_pooled'])}",
        "",
        f"**B2, per-group standardiser (sensitivity).** {stability_finding(conds['B2_per_group'])}",
        "",
        f"**B3, LFW standardiser transferred.** {stability_finding(conds['B3_lfw'])}",
        "",
        "Decision rule: a difference is reported as a finding only if it survives Holm correction across the six "
        "group pairs (alpha 0.05), separately for p_top1 and kappa. Uncorrected decisions are listed below but "
        "are not used for any claim.",
        "",
        "## Detector failure per group",
        "",
        *failure_section(r["detector_failure"]),
        "",
    ]
    if s0:
        lines += ["Step 0 (first 200 images per group): "
                  + ", ".join(f"{g} {d['detection_rate']:.3f}" for g, d in s0["groups"].items())
                  + f"; LFW reference {s0['lfw_reference']['n_landmark_ok']} of {s0['lfw_reference']['n_images']} "
                    f"= {s0['lfw_reference']['rate']:.4f}.", ""]
    for cond, label in (("B1_pooled", "B1, pooled RFW standardiser (primary)"),
                        ("B2_per_group", "B2, per-group standardiser (sensitivity)"),
                        ("B3_lfw", "B3, LFW standardiser transferred")):
        c = conds[cond]
        lines += [f"## {label}", "", c["description"] + ".", "", *stability_table(c), "",
                  "Top-1 class shares:", "", *shares_table(c), "",
                  "Multi-probe (p_topk / mean gallery fraction searched), k = 1 to 3:", "", *multiprobe_table(c), "",
                  "Between groups:", "", *spread_lines(c["between_groups"]), "", *pairwise_table(c["between_groups"]),
                  ""]
    lines += ["## Standardiser conditions compared", "",
              "Fraction of RFW images with landmarks whose top-1 class is the same under two conditions:", ""]
    lines += [f"- {k.replace('_vs_', ' vs ')}: {v:.3f}" for k, v in r["top1_agreement_between_conditions"].items()]
    lines += ["", "## Data accounting", "",
              f"Router run: {r['router_run']['n_images']} images, {r['router_run']['n_landmark_ok']} with landmarks, "
              f"{r['router_run']['n_decode_failed']} decode failures, image sizes "
              f"{', '.join(r['router_run']['image_sizes'])}, {r['router_run']['seconds']:.0f} s. "
              f"Keep rule: {r['keep_rule']}.", "",
              "Outputs: `part_b.json`, `stability_by_group.png`, `landmark_failure_by_group.png`, "
              "`run_manifest.json`.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[verdict] -> {OUT_MD}")


if __name__ == "__main__":
    main()
