# Writes results/stage2/VERDICT.md from the stage 2 result files.
#
# PILOT2.md describes the three outcomes in words ("near 0.2", "clearly
# higher, say above 0.5", "labels flip" / "stable in-domain"). They are
# operationalised by the constants below, in one place. The call is a
# mechanical, provisional reading; the author makes the final call.
#
# Run:  python -m faceshape_pilot.stage2.verdict

import numpy as np

from faceshape_pilot import config, m1_consistency
from faceshape_pilot.common import environment_info, read_json
from faceshape_pilot.stage2 import kaggle
from faceshape_pilot.stage2.step0_check import OUT_JSON as STEP0_JSON
from faceshape_pilot.stage2.step1_train_cnn import OUT_JSON as STEP1_JSON
from faceshape_pilot.stage2.step2_lfw_stability import OUT_JSON as STEP2_JSON
from faceshape_pilot.stage2.step3_perturbation import FACTORS, OUT_JSON as STEP3_JSON, STRENGTHS

OUT_MD = config.STAGE2_DIR / "VERDICT.md"

# CNN kappa on LFW "near 0.2" (like the geometric router): at or below this.
CNN_KAPPA_NEAR_GEOMETRIC_MAX = 0.35
# CNN kappa "clearly higher": above this (PILOT2.md: "say above 0.5").
CNN_KAPPA_CLEARLY_HIGHER = 0.50
# In-domain, two perturbed copies at full strength: "labels flip" at or
# below this kappa, "stable" at or above STABLE.
IN_DOMAIN_FLIP_KAPPA_MAX = 0.60
IN_DOMAIN_STABLE_KAPPA_MIN = 0.80

OUTCOMES = {
    "categories": "CNN kappa also near 0.2, and labels flip in-domain under perturbation: the categories, "
                  "not the method, are unstable.",
    "rules": "CNN kappa clearly higher: the stage 1 finding was partly about the geometric rules.",
    "domain": "CNN kappa near 0.2 but labels stable in-domain: domain shift is doing the work.",
}


def decide(s2, s3):
    k_lfw = s2["cohen_kappa_top1"]
    k_dom = s3["conditions"]["strength_1.00"]["pair_kappa"]
    if k_lfw > CNN_KAPPA_CLEARLY_HIGHER:
        return "rules", k_lfw, k_dom
    if k_lfw <= CNN_KAPPA_NEAR_GEOMETRIC_MAX and k_dom <= IN_DOMAIN_FLIP_KAPPA_MAX:
        return "categories", k_lfw, k_dom
    if k_lfw <= CNN_KAPPA_NEAR_GEOMETRIC_MAX and k_dom >= IN_DOMAIN_STABLE_KAPPA_MIN:
        return "domain", k_lfw, k_dom
    return None, k_lfw, k_dom


def _matched_rows(geo, cnn_res):
    # Each router's (fraction searched, p_topk) points, with the other
    # router's curve interpolated at the same fraction.
    curves = {}
    for name, res in (("geometric", geo), ("CNN", cnn_res)):
        mp = res["multi_probe"]
        curves[name] = (np.array([m["mean_gallery_fraction_searched"] for m in mp]),
                        np.array([m["p_topk"] for m in mp]))
    rows = []
    for name, other in (("geometric", "CNN"), ("CNN", "geometric")):
        x, y = curves[name]
        ox, oy = curves[other]
        for k, (xi, yi) in enumerate(zip(x, y), start=1):
            if xi >= 1.0:
                break
            oi = float(np.interp(xi, np.r_[0.0, ox], np.r_[0.0, oy]))
            rows.append(f"| {name} | {k} | {xi:.3f} | {yi:.3f} | {oi:.3f} ({other}) |")
    return rows


def _f(v, nd=3):
    return "" if v is None else f"{v:.{nd}f}"


def _ci(c, nd=3):
    return "" if c is None or c.get("low") is None else f"({c['low']:.{nd}f}, {c['high']:.{nd}f})"


def render(s0, s1, s2, s3, integ, geo1):
    outcome, k_lfw, k_dom = decide(s2, s3)
    t = s1["test"]
    lines = [
        "# Stage 2 verdict: is the instability in the rule set or in the categories?",
        "",
    ]
    if outcome is None:
        lines += [
            f"**Provisional reading: none of the three PILOT2 outcomes applies cleanly.** CNN kappa on LFW is "
            f"{k_lfw:.3f}; in-domain kappa between two perturbed copies at full strength is {k_dom:.3f}. "
            "See the thresholds below; the author makes the final call.",
        ]
    else:
        lines += [f"**Provisional reading: {OUTCOMES[outcome]}**",
                  "",
                  f"CNN kappa on LFW identities {k_lfw:.3f}; in-domain kappa between two perturbed copies at full "
                  f"strength {k_dom:.3f}. Derived mechanically by `python -m faceshape_pilot.stage2.verdict`; the "
                  "author makes the final call."]
    lines += [
        "",
        "Thresholds (constants at the top of `src/faceshape_pilot/stage2/verdict.py`): "
        f"\"near 0.2\" = kappa <= {CNN_KAPPA_NEAR_GEOMETRIC_MAX}; \"clearly higher\" = kappa > {CNN_KAPPA_CLEARLY_HIGHER}; "
        f"in-domain \"labels flip\" = perturbed-pair kappa at s = 1 <= {IN_DOMAIN_FLIP_KAPPA_MAX}; "
        f"\"stable\" = >= {IN_DOMAIN_STABLE_KAPPA_MIN}.",
        "",
        "## Comparison on LFW (identical images, pairs and bootstrap)",
        "",
        "| Router | Classes | Own-test accuracy | p_top1 | Chance | Kappa (95% CI) |",
        "|---|---|---|---|---|---|",
    ]
    for row in s2["comparison_table"]:
        acc = "not applicable" if row["own_test_accuracy"] is None else _f(row["own_test_accuracy"])
        lines.append(f"| {row['router']} | {row['classes']} | {acc} | {_f(row['p_top1'])} | {_f(row['chance'])} | "
                     f"{_f(row['kappa'])} {_ci(row['kappa_ci95'])} |")
    lines += [
        "",
        "Kappa is the comparison metric: the routers have different numbers of classes, so their chance "
        "agreement differs and raw p_top1 is not comparable.",
        "",
        "Kappa is not the routing cost. For an index key what matters is how many same-identity pairs stay "
        "together for a given fraction of images searched (`consistency_routers_topk.png`). At each router's own "
        "top-k operating points, interpolating the other router's curve:",
        "",
        "| Router at its top-k | k | Fraction of images searched | Pairs kept (this router) | Pairs kept (other router, interpolated) |",
        "|---|---|---|---|---|",
        *_matched_rows(geo1, s2),
        "",
        f"CNN top-1 distribution on LFW: "
        + ", ".join(f"{c} {v:.3f}" for c, v in s2["lfw_top1_distribution"].items()) + ".",
        "",
        "CNN per-class consistency on LFW (p_top1 given the probe's class):",
        "",
        "| Class | Share of images | Ordered pairs | p_top1 given probe class | 95% CI |",
        "|---|---|---|---|---|",
        *[f"| {c} | {_f(v['share_of_images'])} | {v['n_ordered_pairs_with_probe_in_class']} | "
          f"{_f(v['p_top1_given_probe_class'])} | {_ci(v['ci95'])} |" for c, v in s2["per_class"].items()],
        "",
        "CNN multi-probe on LFW:",
        "",
        "| k | p_topk | Mean fraction of images in probed classes |",
        "|---|---|---|",
        *[f"| {m['k']} | {_f(m['p_topk'])} | {_f(m['mean_gallery_fraction_searched'])} |" for m in s2["multi_probe"]],
        "",
        "## The CNN on its own task (Kaggle Face Shape Dataset)",
        "",
        f"Selected backbone: **{s1['selected_backbone']}** (by validation accuracy: "
        + ", ".join(f"{n} {r['best_val_acc']:.4f} at epoch {r['best_epoch']}" for n, r in s1["runs"].items())
        + "). The test split was evaluated once, with the selected model.",
        "",
        f"- Test accuracy on the {t['n_detected']} test images with a detected face: **{t['accuracy_on_detected']:.4f}**",
        f"- Counting the {t['n_test'] - t['n_detected']} detection failures as wrong: {t['accuracy_counting_detection_failures_as_wrong']:.4f}",
        f"- Excluding the {t['n_detected'] - t['n_excluding_duplicates']} test images that have a near-duplicate in "
        f"training: {t['accuracy_excluding_test_images_with_train_duplicate']:.4f}",
        f"- Per class: " + ", ".join(f"{c} {v:.3f}" for c, v in t["per_class_accuracy_on_detected"].items()),
        f"- Published reference: {s1['published_reference']}.",
        "",
        f"Dataset integrity (`{kaggle.OUT_JSON.name}`): {len(integ['decode_failed'])} files fail to decode; "
        f"{integ['near_duplicates']['n_pairs']} confirmed near-duplicate pairs "
        f"({integ['near_duplicates']['n_train_test_pairs']} across the train/test split, "
        f"{integ['near_duplicates']['n_pairs_with_conflicting_labels']} with different labels on the same photograph). "
        "Counts are a lower bound (re-cropped duplicates are not detected).",
        "",
        "## In-domain perturbation control (Kaggle test images)",
        "",
        f"{s3['images']['used']} of {s3['images']['kaggle_test_total']} test images used "
        f"(excluded: {s3['images']['excluded']}); {s3['copies_per_image_per_condition']} perturbed copies per image "
        f"per condition. Clean accuracy on these images {s3['clean_accuracy_on_used_images']:.4f}; clean predictions "
        f"match Step 1: {s3['clean_predictions_match_step1']}.",
        "",
        "| Strength s | Label survival (95% CI) | Pair agreement | Pair kappa (95% CI) | Accuracy | Copies without a face |",
        "|---|---|---|---|---|---|",
    ]
    for s in STRENGTHS:
        c = s3["conditions"][f"strength_{s:.2f}"]
        lines.append(f"| {s:.2f} | {_f(c['survival'])} {_ci(c['survival_ci95'])} | {_f(c['pair_agreement'])} | "
                     f"{_f(c['pair_kappa'])} {_ci(c['pair_kappa_ci95'])} | {_f(c['accuracy'])} | {c['n_copies_no_face']} |")
    lines += [
        "",
        "One factor at a time, s = 1:",
        "",
        "| Factor | Label survival | Pair kappa (95% CI) |",
        "|---|---|---|",
        *[f"| {f} | {_f(s3['conditions'][f'only_{f}_strength_1.00']['survival'])} | "
          f"{_f(s3['conditions'][f'only_{f}_strength_1.00']['pair_kappa'])} "
          f"{_ci(s3['conditions'][f'only_{f}_strength_1.00']['pair_kappa_ci95'])} |" for f in FACTORS],
        "",
        "Perturbation parameters at s = 1 (every range is multiplied by s; per-copy draws are in "
        f"`{s3['params_file']}`, example CNN inputs in `{s3['examples_file']}`):",
        "",
        *[f"- {k}: {v}" for k, v in s3["parameters_at_strength_1"].items()],
        "",
        "Plots: `perturbation_curve.png`, `consistency_routers_topk.png`.",
        "",
        "## Step 0: GPU",
        "",
        f"- Providers bound at runtime (read back from the ONNX sessions): {s0['providers_in_use_gpu_run']}",
        f"- ArcFace extraction time: {s0['embedding_seconds']['cpu_run']} s on CPU, {s0['embedding_seconds']['gpu_run']} s on GPU",
        f"- 640x640 fallback detector usage in stage 1: {s0['fallback_detector_usage_in_stage1']} (detector size: images); removed",
        f"- Router cache bit-identical: {s0['router_cache']['bit_identical']}; measurement 1 identical: "
        f"{s0['results']['consistency.json']['identical']}; measurement 2 identical: {s0['results']['distribution.json']['identical']}",
        f"- Measurement 3 changed in {s0['results']['partition_comparison.json']['n_differing_values']} values. "
        f"Cause: GPU and CPU embeddings differ by at most {s0['embeddings']['max_abs_component_diff']:.2e} per "
        f"component (minimum cosine {s0['embeddings']['cosine_cpu_vs_gpu_min']:.6f}), which moves k-means to a "
        "different local optimum. The current measurement 3 code on the archived CPU embeddings reproduces the CPU "
        f"numbers exactly: {s0['measurement3_code_on_cpu_embeddings']['reproduces_archived_cpu_numbers_exactly']}.",
        f"- k-means AUC across {s0['kmeans_seed_sensitivity']['n_seeds']} k-means seeds: CPU embeddings "
        f"{s0['kmeans_seed_sensitivity']['cpu']['auc_min']:.3f} to {s0['kmeans_seed_sensitivity']['cpu']['auc_max']:.3f}, "
        f"GPU embeddings {s0['kmeans_seed_sensitivity']['gpu']['auc_min']:.3f} to {s0['kmeans_seed_sensitivity']['gpu']['auc_max']:.3f}. "
        "The stage 1 shape-versus-k-means AUC comparison lies inside this seed range and should not be read as a "
        "difference between the partitions.",
        "",
        "## Caveats",
        "",
        "- The CNN is trained on five classes; the geometric router has seven. Kappa corrects for chance agreement "
        "but not for every effect of a different taxonomy.",
        "- The planar pose model omits 3D effects, and the training augmentation overlaps with the perturbations; "
        "both make the in-domain control favour stability.",
        "- Two perturbed copies of one photograph differ less than two photographs of one person taken years apart "
        "(hair, make-up, age, camera), so in-domain pair kappa is an upper bound on what real within-identity "
        "variation would allow.",
        f"- {config.DATASET_LIMITATIONS}",
        "- Kaggle consists of female celebrity portraits; LFW is mostly male news photography.",
        "",
        f"Seed {config.SEED}. Generated {environment_info()['utc_time']}.",
        "",
    ]
    return "\n".join(lines)


def main():
    s0, s1, s2, s3 = (read_json(p) for p in (STEP0_JSON, STEP1_JSON, STEP2_JSON, STEP3_JSON))
    integ = read_json(kaggle.OUT_JSON)
    geo1 = read_json(m1_consistency.OUT_JSON)
    OUT_MD.write_text(render(s0, s1, s2, s3, integ, geo1), encoding="utf-8")
    outcome = decide(s2, s3)[0]
    print(f"[stage2 verdict] {outcome or 'none of the three cleanly'}  -> {OUT_MD}")


if __name__ == "__main__":
    main()
