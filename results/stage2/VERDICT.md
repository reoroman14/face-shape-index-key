# Stage 2 verdict: is the instability in the rule set or in the categories?

**Provisional reading: CNN kappa near 0.2 but labels stable in-domain: domain shift is doing the work.**

CNN kappa on LFW identities 0.256; in-domain kappa between two perturbed copies at full strength 0.888. Derived mechanically by `python -m faceshape_pilot.stage2.verdict`; the author makes the final call.

Thresholds (constants at the top of `src/faceshape_pilot/stage2/verdict.py`): "near 0.2" = kappa <= 0.35; "clearly higher" = kappa > 0.5; in-domain "labels flip" = perturbed-pair kappa at s = 1 <= 0.6; "stable" = >= 0.8.

## Comparison on LFW (identical images, pairs and bootstrap)

| Router | Classes | Own-test accuracy | p_top1 | Chance | Kappa (95% CI) |
|---|---|---|---|---|---|
| geometric (stage 1) | 7 | not applicable | 0.327 | 0.185 | 0.177 (0.161, 0.193) |
| CNN (resnet50, Kaggle-trained) | 5 | 0.906 | 0.624 | 0.537 | 0.256 (0.230, 0.281) |

Kappa is the comparison metric: the routers have different numbers of classes, so their chance agreement differs and raw p_top1 is not comparable.

Kappa is not the routing cost. For an index key what matters is how many same-identity pairs stay together for a given fraction of images searched (`consistency_routers_topk.png`). At each router's own top-k operating points, interpolating the other router's curve:

| Router at its top-k | k | Fraction of images searched | Pairs kept (this router) | Pairs kept (other router, interpolated) |
|---|---|---|---|---|
| geometric | 1 | 0.184 | 0.327 | 0.222 (CNN) |
| geometric | 2 | 0.350 | 0.529 | 0.424 (CNN) |
| geometric | 3 | 0.484 | 0.669 | 0.586 (CNN) |
| geometric | 4 | 0.616 | 0.778 | 0.713 (CNN) |
| geometric | 5 | 0.753 | 0.866 | 0.833 (CNN) |
| geometric | 6 | 0.887 | 0.946 | 0.933 (CNN) |
| CNN | 1 | 0.515 | 0.624 | 0.695 (geometric) |
| CNN | 2 | 0.763 | 0.841 | 0.872 (geometric) |
| CNN | 3 | 0.908 | 0.948 | 0.956 (geometric) |
| CNN | 4 | 0.968 | 0.984 | 0.985 (geometric) |

CNN top-1 distribution on LFW: heart 0.009, oblong 0.704, oval 0.015, round 0.183, square 0.089.

CNN per-class consistency on LFW (p_top1 given the probe's class):

| Class | Share of images | Ordered pairs | p_top1 given probe class | 95% CI |
|---|---|---|---|---|
| heart | 0.009 | 161 | 0.149 | (0.072, 0.229) |
| oblong | 0.704 | 9304 | 0.778 | (0.761, 0.793) |
| oval | 0.015 | 268 | 0.201 | (0.130, 0.271) |
| round | 0.183 | 2887 | 0.386 | (0.355, 0.419) |
| square | 0.089 | 1360 | 0.224 | (0.185, 0.260) |

CNN multi-probe on LFW:

| k | p_topk | Mean fraction of images in probed classes |
|---|---|---|
| 1 | 0.624 | 0.515 |
| 2 | 0.841 | 0.763 |
| 3 | 0.948 | 0.908 |
| 4 | 0.984 | 0.968 |
| 5 | 1.000 | 1.000 |

## The CNN on its own task (Kaggle Face Shape Dataset)

Selected backbone: **resnet50** (by validation accuracy: efficientnet_b0 0.8875 at epoch 15, resnet50 0.9050 at epoch 20). The test split was evaluated once, with the selected model.

- Test accuracy on the 1000 test images with a detected face: **0.9060**
- Counting the 0 detection failures as wrong: 0.9060
- Excluding the 10 test images that have a near-duplicate in training: 0.9051
- Per class: heart 0.930, oblong 0.960, oval 0.805, round 0.885, square 0.950
- Published reference: up to 96.32 percent on this corpus (grd2024transfer).

Dataset integrity (`kaggle_integrity.json`): 2 files fail to decode; 33 confirmed near-duplicate pairs (11 across the train/test split, 2 with different labels on the same photograph). Counts are a lower bound (re-cropped duplicates are not detected).

## In-domain perturbation control (Kaggle test images)

1000 of 1000 test images used (excluded: {'no_face_clean': 0, 'no_landmarks_clean': 0}); 10 perturbed copies per image per condition. Clean accuracy on these images 0.9060; clean predictions match Step 1: True.

| Strength s | Label survival (95% CI) | Pair agreement | Pair kappa (95% CI) | Accuracy | Copies without a face |
|---|---|---|---|---|---|
| 0.00 | 1.000 (1.000, 1.000) | 1.000 | 1.000 (1.000, 1.000) | 0.906 | 0 |
| 0.25 | 0.967 (0.959, 0.975) | 0.961 | 0.951 (0.939, 0.961) | 0.906 | 0 |
| 0.50 | 0.950 (0.940, 0.960) | 0.948 | 0.935 (0.924, 0.947) | 0.901 | 0 |
| 0.75 | 0.936 (0.926, 0.947) | 0.937 | 0.921 (0.907, 0.933) | 0.889 | 0 |
| 1.00 | 0.915 (0.903, 0.928) | 0.910 | 0.888 (0.872, 0.903) | 0.877 | 0 |

One factor at a time, s = 1:

| Factor | Label survival | Pair kappa (95% CI) |
|---|---|---|
| pose | 0.967 | 0.956 (0.945, 0.966) |
| expression | 0.959 | 0.958 (0.948, 0.967) |
| lighting | 0.972 | 0.952 (0.942, 0.963) |
| crop | 0.945 | 0.924 (0.911, 0.937) |

Perturbation parameters at s = 1 (every range is multiplied by s; per-copy draws are in `perturbation_params.csv`, example CNN inputs in `perturbation_examples.jpg`):

- yaw_deg_max_abs: 15.0
- pitch_deg_max_abs: 10.0
- pose_model: planar face 0.15 m wide at 1.0 m, rotated about its centre
- jaw_drop_max_fraction_of_face_height: 0.04
- mouth_corner_max_abs_fraction_of_face_height: 0.03
- mouth_corner_sigma_fraction_of_face_width: 0.07
- lighting_gradient_max_abs: 0.25
- brightness_max_abs: 0.2
- contrast_factor_range: [0.8, 1.25]
- gamma_range: [0.8, 1.25]
- crop_shift_max_abs_fraction_of_side: 0.04
- crop_scale_max_abs: 0.05
- sampling: each parameter uniform in [-s, s] x max (jaw drop in [0, s x max]; contrast and gamma uniform in log space); gradient direction uniform in [0, 360) at every s > 0
- order: expression -> pose -> lighting -> SCRFD re-detection -> crop jitter

Plots: `perturbation_curve.png`, `consistency_routers_topk.png`.

## Step 0: GPU

- Providers bound at runtime (read back from the ONNX sessions): {'detection': ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'recognition': ['CUDAExecutionProvider', 'CPUExecutionProvider']}
- ArcFace extraction time: 3053.3 s on CPU, 187.2 s on GPU
- 640x640 fallback detector usage in stage 1: {'320': 9164} (detector size: images); removed
- Router cache bit-identical: True; measurement 1 identical: True; measurement 2 identical: True
- Measurement 3 changed in 97 values. Cause: GPU and CPU embeddings differ by at most 1.48e-03 per component (minimum cosine 0.999953), which moves k-means to a different local optimum. The current measurement 3 code on the archived CPU embeddings reproduces the CPU numbers exactly: True.
- k-means AUC across 20 k-means seeds: CPU embeddings 0.607 to 0.734, GPU embeddings 0.607 to 0.746. The stage 1 shape-versus-k-means AUC comparison lies inside this seed range and should not be read as a difference between the partitions.

## Caveats

- The CNN is trained on five classes; the geometric router has seven. Kappa corrects for chance agreement but not for every effect of a different taxonomy.
- The planar pose model omits 3D effects, and the training augmentation overlaps with the perturbations; both make the in-domain control favour stability.
- Two perturbed copies of one photograph differ less than two photographs of one person taken years apart (hair, make-up, age, camera), so in-domain pair kappa is an upper bound on what real within-identity variation would allow.
- LFW is demographically skewed (predominantly male, lighter-skinned, middle-aged public figures) and consists of news photographs of public figures. This is acceptable for a pilot that asks whether the routing mechanism works at all. It is not acceptable for the fairness analysis in the paper, which requires a demographically balanced set.
- Kaggle consists of female celebrity portraits; LFW is mostly male news photography.

Seed 20261017. Generated 2026-10-18T00:19:02+00:00.
