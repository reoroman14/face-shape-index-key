# Writes results/stage3/VERDICT.md from results/stage3/part_a.json.
# Every number in the file is read from part_a.json; nothing is typed in.
#
# Run:  python -m faceshape_pilot.stage3.verdict

import datetime as _dt

from faceshape_pilot import config
from faceshape_pilot.common import read_json
from faceshape_pilot.stage3 import part_a

OUT_MD = config.STAGE3_DIR / "VERDICT.md"
PRED_LABELS = {
    "shape_label": "top-1 shape label",
    "shape_probabilities": "7 shape probabilities",
    "ratio_features": "4 ratio features (L, F, J, A)",
}
ANALYSIS_LABELS = {
    "race": "race, all ages",
    "race_adults": "race, adults (20+)",
    "gender": "gender, all ages",
    "gender_adults": "gender, adults (20+)",
    "age": "age band",
}


def f3(x):
    return "" if x is None else f"{x:.3f}"


def ci(pair):
    return "" if pair is None or None in pair else f"({pair[0]:.3f}, {pair[1]:.3f})"


def pval(p):
    if p is None:
        return ""
    return "< 1e-300" if p == 0 else f"{p:.2g}"


def shares_table(ct: dict, label: str) -> list:
    head = f"| {label} | n | " + " | ".join(config.CLASSES) + " | Largest | 1/max (95% CI) |"
    lines = [head, "|" + "---|" * (len(config.CLASSES) + 4)]
    for g, d in ct["groups"].items():
        lines.append(f"| {g} | {d['n_with_landmarks']} | "
                     + " | ".join(f3(d["shares"][c]) for c in config.CLASSES)
                     + f" | {d['largest_class']} | {f3(d['ceiling_1_over_max_share'])} {ci(d['ceiling_ci95'])} |")
    p = ct["pooled"]
    lines.append(f"| **all** | {ct['n_images']} | " + " | ".join(f3(p["shares"][c]) for c in config.CLASSES)
                 + f" | {p['largest_class']} | {f3(p['ceiling_1_over_max_share'])} {ci(p['ceiling_ci95'])} |")
    return lines


def independence_table(cts: dict) -> list:
    lines = ["| Analysis | n | Chi-square | dof | p | Cramer's V (95% CI) | Bias-corrected V (95% CI) "
             "| Permutation null V, 99th pct | Cohen band |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, ct in cts.items():
        i = ct["independence"]
        lines.append(f"| {ANALYSIS_LABELS[name]} | {ct['n_images']} | {i['chi2']:.1f} | {i['dof']} | "
                     f"{pval(i['p_value'])} | {f3(i['cramers_v'])} {ci(i['cramers_v_ci95'])} | "
                     f"{f3(i['cramers_v_bias_corrected'])} {ci(i['cramers_v_bias_corrected_ci95'])} | "
                     f"{f3(i['permutation_null_v']['p99'])} | {i['cohen_band']} |")
    return lines


def reverse_table(rev: dict, analyses) -> list:
    lines = ["| Target | Predictor | n test | Accuracy (95% CI) | Majority baseline | Balanced acc. (95% CI) "
             "| Chance | Macro AUC (95% CI) | Info gain, fraction of prior entropy (95% CI) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in analyses:
        r = rev[name]
        b = r["baselines"]
        for p, d in r["predictors"].items():
            lines.append(f"| {ANALYSIS_LABELS[name]} | {PRED_LABELS[p]} | {r['n_test']} | "
                         f"{f3(d['accuracy'])} {ci(d['accuracy_ci95'])} | {f3(b['accuracy_majority_class'])} | "
                         f"{f3(d['balanced_accuracy'])} {ci(d['balanced_accuracy_ci95'])} | "
                         f"{f3(b['balanced_accuracy_chance'])} | {f3(d['macro_ovr_auc'])} {ci(d['macro_ovr_auc_ci95'])} | "
                         f"{f3(d['information_gain_fraction'])} {ci(d['information_gain_fraction_ci95'])} |")
    return lines


def lift_table(sl: dict, names) -> list:
    lines = ["| Shape label | " + " | ".join(names) + " |", "|" + "---|" * (len(names) + 1)]
    for c in config.CLASSES:
        lines.append(f"| {c} | " + " | ".join(f"{sl['lift_p_group_given_label_over_prior'][c][g]:.2f}"
                                              for g in names) + " |")
    return lines


def failure_table(fl: dict, label: str) -> list:
    lines = [f"| {label} | Images | Failed | Failure rate (95% Wilson CI) |", "|---|---|---|---|"]
    for g, d in fl["groups"].items():
        lines.append(f"| {g} | {d['n_images']} | {d['n_failed']} | {d['failure_rate']:.4f} "
                     f"({d['failure_rate_ci95_wilson'][0]:.4f}, {d['failure_rate_ci95_wilson'][1]:.4f}) |")
    lines.append(f"| **all** | {fl['n_images']} | {fl['n_failed']} | {fl['failure_rate']:.4f} |")
    if "independence" in fl:
        i = fl["independence"]
        lines += ["", f"Failure vs {label.lower()}: chi-square {i['chi2']:.1f}, dof {i['dof']}, p {pval(i['p_value'])}, "
                      f"Cramer's V {i['cramers_v']:.3f}."]
    return lines


def render(r: dict) -> str:
    ds = r["dataset"]
    run = r["router_run"]
    fz = r["crosstabs"]["frozen"]
    rf = r["crosstabs"]["refit"]
    rev = r["reverse"]["frozen"]
    race_v = fz["race"]["independence"]
    race_va = fz["race_adults"]["independence"]
    sl = rev["race"]["predictors"]["shape_label"]
    sla = rev["race_adults"]["predictors"]["shape_label"]
    rat = rev["race"]["predictors"]["ratio_features"]
    fail_race = r["landmark_failure"]["race"]["groups"]
    fr = sorted(fail_race.items(), key=lambda kv: kv[1]["failure_rate"])
    ceil = sorted(fz["race"]["groups"].items(), key=lambda kv: kv[1]["ceiling_1_over_max_share"])
    races = list(config.FAIRFACE_RACES)
    lifts = sl["lift_p_group_given_label_over_prior"]
    top_lift = max(((c, g, lifts[c][g]) for c in config.CLASSES for g in races), key=lambda t: t[2])
    low_lift = min(((c, g, lifts[c][g]) for c in config.CLASSES for g in races), key=lambda t: t[2])

    L = []
    L.append("# Stage 3 verdict: does the taxonomy behave the same across demographic groups?")
    L.append("")
    L.append("**Part A (FairFace, distribution across groups) is reported below. Part B (RFW, per-group "
             "stability) has not been run: RFW access has not been obtained. Per-group identity consistency "
             "(p_top1, kappa) therefore remains unmeasured; nothing here substitutes for it and LFW does not "
             "cover it.**")
    L.append("")
    L.append("Generated mechanically by `python -m faceshape_pilot.stage3.verdict` from `part_a.json`. "
             "The author makes the final call on wording for the paper.")
    L.append("")
    L.append("## Data and router")
    L.append("")
    L.append(f"- FairFace, Hugging Face `{config.FAIRFACE_HF_REPO}` at revision `{ds['revision']}`, padding "
             f"{ds['padding']}, {ds['total_bytes']} bytes, every shard sha256-verified (`fairface_download.json`).")
    L.append(f"- {ds['n_images']} images (train {ds['n_by_split']['train']}, validation "
             f"{ds['n_by_split']['validation']}). The 108,501 in the FairFace paper include images that were not "
             "publicly released; the released train and validation splits are all that is analysed.")
    L.append(f"- Decode failures: {run['n_decode_failed']}. Landmarks found: {run['n_landmark_ok']}; "
             f"no face found: {run['n_landmark_failed_after_decode']}. Image sizes: {', '.join(run['image_sizes'])}.")
    L.append("- Router: the stage 1 code and model file (MediaPipe sha256 "
             f"`{run['mediapipe_model_sha256'][:16]}...`), prototypes unchanged. Primary analysis uses the "
             "stage 1 standardiser fitted on LFW (**frozen**); the sensitivity analysis re-fits only the "
             "median/IQR standardiser on FairFace (**refit**). Top-1 agreement between the two variants: "
             f"{r['top1_agreement_frozen_vs_refit']:.3f}.")
    L.append("- Labels are FairFace's annotator-perceived race, gender and age, not self-identification. "
             "Race groups: " + ", ".join(races) + ".")
    L.append(f"- Seed {r['seed']}; bootstrap B = {r['n_bootstrap']} over images; permutation null with "
             f"{fz['race']['independence']['permutation_null_v']['n_permutations']} shuffles.")
    L.append("")

    L.append("## 1. Landmark detection failure per group")
    L.append("")
    L.append("A detector that fails more often on one group confounds everything downstream, so this comes first. "
             "All analyses below are conditional on landmarks being found.")
    L.append("")
    L += failure_table(r["landmark_failure"]["race"], "Race")
    L.append("")
    L += failure_table(r["landmark_failure"]["race_adults"], "Race (adults)")
    L.append("")
    L += failure_table(r["landmark_failure"]["gender"], "Gender")
    L.append("")
    L += failure_table(r["landmark_failure"]["age"], "Age")
    L.append("")
    L.append(f"Lowest race-group failure rate: {fr[0][0]} {fr[0][1]['failure_rate']:.4f}; highest: {fr[-1][0]} "
             f"{fr[-1][1]['failure_rate']:.4f}. Plot: `landmark_failure_by_group.png`.")
    L.append("")

    L.append("## 2. Seven-class distribution within each group (frozen router)")
    L.append("")
    L.append("Shares of each group's faces by top-1 class, with the per-group ceiling on candidate-set "
             "reduction 1 / max_k pi_k (bootstrap 95% CI). Per-cell CIs, adjusted residuals and lifts are in "
             "`part_a.json`. Plot: `shape_by_group.png`.")
    L.append("")
    L += shares_table(fz["race"], "Race")
    L.append("")
    L += shares_table(fz["race_adults"], "Race (adults)")
    L.append("")
    L += shares_table(fz["gender"], "Gender")
    L.append("")
    L += shares_table(fz["age"], "Age")
    L.append("")
    L.append(f"Race-group ceilings range from {ceil[0][1]['ceiling_1_over_max_share']:.2f} ({ceil[0][0]}) to "
             f"{ceil[-1][1]['ceiling_1_over_max_share']:.2f} ({ceil[-1][0]}); pooled "
             f"{fz['race']['pooled']['ceiling_1_over_max_share']:.2f}.")
    L.append("")

    L.append("## 3. Independence of predicted shape and each attribute")
    L.append("")
    L.append("At this sample size every chi-square test is expected to reject; the effect size is Cramer's V. "
             "The permutation column gives the V that label shuffling produces at the same n, i.e. what "
             "'no association' looks like here. Cohen bands use w = 0.1 / 0.3 / 0.5 divided by "
             "sqrt(min(rows, cols) - 1), a convention, not a criterion derived for this use.")
    L.append("")
    L.append("Frozen router (primary):")
    L.append("")
    L += independence_table(fz)
    L.append("")
    L.append("Refit standardiser (sensitivity):")
    L.append("")
    L += independence_table(rf)
    L.append("")

    L.append("## 4. Reverse direction: how well does the shape output predict the group?")
    L.append("")
    L.append("Fitted on the FairFace train split, evaluated once on the validation split (official split), "
             "images with landmarks. Information gain is the out-of-sample reduction in cross-entropy relative "
             "to the train-set group prior, as a fraction of the prior's cross-entropy (0 = the label tells you "
             "nothing, 1 = it determines the group). Balanced accuracy for the shape label uses the rule "
             "argmax P(group | label) / P(group).")
    L.append("")
    L += reverse_table(rev, ("race", "race_adults", "gender", "gender_adults", "age"))
    L.append("")
    L.append("Lift P(race | shape label) / P(race) from the train split (frozen router, all ages); 1.00 = the "
             "label carries no information about that group:")
    L.append("")
    L += lift_table(sl, races)
    L.append("")
    L.append(f"Largest lift: a face labelled {top_lift[0]} is {top_lift[2]:.2f} times as likely as the base rate to "
             f"carry the {top_lift[1]} label; smallest: {low_lift[0]} -> {low_lift[1]} at {low_lift[2]:.2f}.")
    L.append("")
    L.append("Refit standardiser, race:")
    L.append("")
    L += reverse_table(r["reverse"]["refit"], ("race", "race_adults"))
    L.append("")

    L.append("## 5. What the paper can and cannot now claim about demographic generality")
    L.append("")
    L.append(
        f"On FairFace the router's top-1 shape is associated with annotated race with Cramer's V "
        f"{race_v['cramers_v']:.3f} {ci(race_v['cramers_v_ci95'])} ({race_v['cohen_band']} by Cohen's convention; "
        f"{race_va['cramers_v']:.3f} {ci(race_va['cramers_v_ci95'])} on adults), against a no-association value of "
        f"at most {race_v['permutation_null_v']['p99']:.3f} at this sample size. In the reverse direction the shape "
        f"label alone predicts race on held-out images with balanced accuracy {sl['balanced_accuracy']:.3f} "
        f"{ci(sl['balanced_accuracy_ci95'])} against chance {rev['race']['baselines']['balanced_accuracy_chance']:.3f}, "
        f"macro AUC {sl['macro_ovr_auc']:.3f} {ci(sl['macro_ovr_auc_ci95'])}, and removes "
        f"{100 * sl['information_gain_fraction']:.1f} percent ({100 * sl['information_gain_fraction_ci95'][0]:.1f} "
        f"to {100 * sl['information_gain_fraction_ci95'][1]:.1f}) "
        f"of the uncertainty about race (adults: balanced accuracy {sla['balanced_accuracy']:.3f}, "
        f"{100 * sla['information_gain_fraction']:.1f} percent); the four ratio features underneath it reach "
        f"balanced accuracy {rat['balanced_accuracy']:.3f} and macro AUC {rat['macro_ovr_auc']:.3f}. "
        f"Landmark detection fails on {100 * fr[0][1]['failure_rate']:.1f} to {100 * fr[-1][1]['failure_rate']:.1f} "
        f"percent of images depending on race group. For comparison, the association of the same shape label "
        f"with annotated gender is V {fz['gender_adults']['independence']['cramers_v']:.3f} on adults and with "
        f"age band V {fz['age']['independence']['cramers_v']:.3f}; race is "
        f"{'not ' if race_v['cramers_v'] < max(fz['gender_adults']['independence']['cramers_v'], fz['age']['independence']['cramers_v']) else ''}"
        f"the attribute the label tracks most strongly. The paper can therefore state the measured size of the "
        f"association between the shape key and perceived race in both directions, and that a shape-keyed index "
        f"is to that measured extent an ethnicity-keyed index, which bears directly on the forensic exclusion use "
        f"case. It cannot claim that the stage 1 instability result (kappa on LFW) generalises across demographic "
        f"groups: per-group stability needs identity labels, Part B (RFW) has not been run, and FairFace cannot "
        f"supply it. The limitations section should say plainly that per-group stability remains unmeasured.")
    L.append("")

    L.append("## Caveats")
    L.append("")
    L.append("- Race, gender and age are annotator-perceived labels from FairFace, not self-identification; the "
             "seven race categories are FairFace's.")
    L.append("- FairFace has no identity labels; the same person may appear more than once, so bootstrap intervals "
             "that resample images may be somewhat too narrow, and a person could appear in both the train and "
             "validation splits of the reverse-direction test.")
    L.append("- FairFace images are Flickr photographs (YFCC-100M) with uncontrolled pose, expression and "
             "resolution; part of any group difference may come from how photographs of each group were taken "
             "and selected rather than from face geometry. The router cannot separate the two.")
    L.append("- The rules were written for adult faces; children's age bands are reported but the adult-only "
             "analyses are the relevant ones for the forensic use case.")
    L.append("- Analyses are conditional on MediaPipe finding landmarks; group differences in failure rate "
             "(section 1) change which faces enter the cross-tabulations.")
    L.append("- The padding 1.25 crops were used so that forehead and chin landmarks are inside the image; the "
             "tighter padding 0.25 release was not analysed.")
    L.append("")
    L.append(f"Seed {r['seed']}. Generated {_dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds')}.")
    L.append("")
    return "\n".join(L)


def main():
    r = read_json(part_a.OUT_JSON)
    OUT_MD.write_text(render(r), encoding="utf-8")
    print(f"[verdict] -> {OUT_MD}")


if __name__ == "__main__":
    main()
