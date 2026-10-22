# Stage 4, Part B: per-group within-identity stability on RFW

Generated 2026-10-22 from `results/stage4/part_b.json`. Seed 20261017, 1000 bootstrap resamples over identities.

## Provenance

- Dataset: RFW (Racial Faces in-the-Wild), test subsets only; derived from MS-Celeb-1M, withdrawn by its creator (Microsoft) in April 2019.
- Access: granted in writing by Mei Wang (Beijing University of Posts and Telecommunications) on 2026-10-20, in response to a request from Applied Science Private University.
- Obtained from the Google Drive link supplied by the maintainer, not a mirror. Downloaded: the four test subsets only (Caucasian, Asian, Indian, African); the MS1M-wo-RFW and BUPT training sets distributed alongside them were not downloaded.
- Verification: unpacked tree verified file by file against the official counts (40,607 images, 11,429 identities) before use; re-verified by faceshape_pilot.stage4.rfw on every run.
- Maintainer readme: `data/rfw/readme.txt` (sha256 `fbfa5fe0759b76ebb6e5455a93ba965c41f647bd56101df6f85eb1fe776c0be2`).

## Findings

**B1, pooled RFW standardiser (primary).** **Uneven.** Holm-corrected pairwise differences: p_top1 Caucasian - Asian = +0.017 (0.004, 0.030), Holm p = 0.024; p_top1 Caucasian - Indian = -0.028 (-0.043, -0.014), Holm p = 0.012; p_top1 Caucasian - African = -0.043 (-0.057, -0.029), Holm p = 0.012; p_top1 Asian - Indian = -0.045 (-0.060, -0.030), Holm p = 0.012; p_top1 Asian - African = -0.061 (-0.076, -0.046), Holm p = 0.012; Cohen's kappa Caucasian - Indian = -0.026 (-0.043, -0.010), Holm p = 0.016; Cohen's kappa Caucasian - African = +0.022 (0.006, 0.038), Holm p = 0.018; Cohen's kappa Asian - Indian = -0.039 (-0.055, -0.022), Holm p = 0.012; Cohen's kappa Indian - African = +0.049 (0.032, 0.065), Holm p = 0.012. Under this condition a person in the lower-stability group is less likely to be routed to the class of their own other images than a person in the higher-stability group.

**B2, per-group standardiser (sensitivity).** **Uneven.** Holm-corrected pairwise differences: p_top1 Indian - African = +0.020 (0.006, 0.034), Holm p = 0.036; Cohen's kappa Indian - African = +0.025 (0.010, 0.042), Holm p = 0.012. Under this condition a person in the lower-stability group is less likely to be routed to the class of their own other images than a person in the higher-stability group.

**B3, LFW standardiser transferred.** **Uneven.** Holm-corrected pairwise differences: p_top1 Caucasian - Indian = -0.041 (-0.055, -0.026), Holm p = 0.012; p_top1 Caucasian - African = -0.041 (-0.055, -0.028), Holm p = 0.012; p_top1 Asian - Indian = -0.049 (-0.063, -0.034), Holm p = 0.012; p_top1 Asian - African = -0.049 (-0.063, -0.035), Holm p = 0.012; Cohen's kappa Caucasian - African = +0.030 (0.014, 0.046), Holm p = 0.012; Cohen's kappa Asian - Indian = -0.032 (-0.047, -0.015), Holm p = 0.016; Cohen's kappa Indian - African = +0.051 (0.033, 0.068), Holm p = 0.012. Under this condition a person in the lower-stability group is less likely to be routed to the class of their own other images than a person in the higher-stability group.

Decision rule: a difference is reported as a finding only if it survives Holm correction across the six group pairs (alpha 0.05), separately for p_top1 and kappa. Uncorrected decisions are listed below but are not used for any claim.

## Detector failure per group

| Group | Images | No landmarks | Image failure rate (95% CI) | Identities | Dropped below 2 | Identity drop rate (95% CI) |
|---|---|---|---|---|---|---|
| Caucasian | 10196 | 89 | 0.0087 (0.0070, 0.0105) | 2958 | 8 | 0.0027 (0.0010, 0.0047) |
| Asian | 9688 | 61 | 0.0063 (0.0047, 0.0079) | 2492 | 3 | 0.0012 (0.0000, 0.0028) |
| Indian | 10308 | 59 | 0.0057 (0.0043, 0.0072) | 2984 | 5 | 0.0017 (0.0003, 0.0034) |
| African | 10415 | 97 | 0.0093 (0.0073, 0.0112) | 2995 | 7 | 0.0023 (0.0007, 0.0040) |

- Image failure rate, highest minus lowest: African minus Indian = +0.0036 (0.0011, 0.0060), bootstrap p = 0.00999.
- Identity drop rate, highest minus lowest: Caucasian minus Asian = +0.0015 (-0.0007, 0.0037), bootstrap p = 0.174.
- Chi-square of failure against group (images independent): chi2 = 12.86, dof = 3, p = 0.00496.

**Uneven.** The face mesh fails more often on African than on Indian images (+0.0036 (0.0011, 0.0060)). This is a channel of unequal treatment upstream of the shape label, invisible in any accuracy number.

Step 0 (first 200 images per group): Caucasian 0.975, Asian 0.975, Indian 0.990, African 0.990; LFW reference 9154 of 9164 = 0.9989.

## B1, pooled RFW standardiser (primary)

one standardiser fitted on the ratio features of all four RFW groups pooled (primary).

| Group | Images | Identities | Pairs | p_top1 (95% CI) | Kappa (95% CI) | Chance agreement | Largest class (share) | Effective classes (95% CI) |
|---|---|---|---|---|---|---|---|---|
| Caucasian | 10100 | 2950 | 13094 | 0.291 (0.281, 0.300) | 0.131 (0.120, 0.142) | 0.182 | oval (0.312) | 5.48 (5.36, 5.59) |
| Asian | 9624 | 2489 | 13850 | 0.273 (0.264, 0.284) | 0.119 (0.107, 0.130) | 0.176 | oval (0.283) | 5.69 (5.58, 5.79) |
| Indian | 10244 | 2979 | 13270 | 0.319 (0.308, 0.330) | 0.157 (0.146, 0.170) | 0.190 | square (0.318) | 5.26 (5.14, 5.39) |
| African | 10312 | 2988 | 13503 | 0.334 (0.323, 0.345) | 0.109 (0.097, 0.121) | 0.251 | oval (0.346) | 3.98 (3.91, 4.06) |

Top-1 class shares:

| Group | oval | round | square | rectangle | heart | diamond | triangle |
|---|---|---|---|---|---|---|---|
| Caucasian | 0.312 | 0.093 | 0.173 | 0.141 | 0.087 | 0.086 | 0.108 |
| Asian | 0.283 | 0.084 | 0.168 | 0.071 | 0.134 | 0.172 | 0.088 |
| Indian | 0.207 | 0.096 | 0.318 | 0.114 | 0.078 | 0.084 | 0.103 |
| African | 0.346 | 0.318 | 0.135 | 0.033 | 0.080 | 0.023 | 0.066 |

Multi-probe (p_topk / mean gallery fraction searched), k = 1 to 3:

| Group | k=1: p_topk / gallery | k=2: p_topk / gallery | k=3: p_topk / gallery |
|---|---|---|---|
| Caucasian | 0.291 / 0.183 | 0.490 / 0.350 | 0.633 / 0.487 |
| Asian | 0.273 / 0.176 | 0.473 / 0.346 | 0.617 / 0.483 |
| Indian | 0.319 / 0.191 | 0.510 / 0.345 | 0.651 / 0.483 |
| African | 0.334 / 0.252 | 0.563 / 0.465 | 0.721 / 0.626 |

Between groups:

- p_top1 spread: African minus Asian = +0.061, CI of that pair's difference (0.046, 0.076), CI of replicate max-min (0.047, 0.076). Holm-significant pairs: 5 of 6 (uncorrected: 5 of 6).
- Cohen's kappa spread: Indian minus African = +0.049, CI of that pair's difference (0.032, 0.065), CI of replicate max-min (0.034, 0.065). Holm-significant pairs: 4 of 6 (uncorrected: 4 of 6).

| Metric | Pair | Difference (95% CI) | CI excludes 0 | p (bootstrap) | p (Holm) | Significant, uncorrected | Significant, Holm |
|---|---|---|---|---|---|---|---|
| p_top1 | Caucasian - Asian | +0.017 (0.004, 0.030) | yes | 0.012 | 0.024 | yes | **yes** |
| p_top1 | Caucasian - Indian | -0.028 (-0.043, -0.014) | yes | 0.002 | 0.012 | yes | **yes** |
| p_top1 | Caucasian - African | -0.043 (-0.057, -0.029) | yes | 0.002 | 0.012 | yes | **yes** |
| p_top1 | Asian - Indian | -0.045 (-0.060, -0.030) | yes | 0.002 | 0.012 | yes | **yes** |
| p_top1 | Asian - African | -0.061 (-0.076, -0.046) | yes | 0.002 | 0.012 | yes | **yes** |
| p_top1 | Indian - African | -0.015 (-0.029, 0.000) | no | 0.0579 | 0.0579 | no | no |
| Cohen's kappa | Caucasian - Asian | +0.013 (-0.003, 0.028) | no | 0.128 | 0.256 | no | no |
| Cohen's kappa | Caucasian - Indian | -0.026 (-0.043, -0.010) | yes | 0.004 | 0.016 | yes | **yes** |
| Cohen's kappa | Caucasian - African | +0.022 (0.006, 0.038) | yes | 0.00599 | 0.018 | yes | **yes** |
| Cohen's kappa | Asian - Indian | -0.039 (-0.055, -0.022) | yes | 0.002 | 0.012 | yes | **yes** |
| Cohen's kappa | Asian - African | +0.010 (-0.007, 0.026) | no | 0.246 | 0.256 | no | no |
| Cohen's kappa | Indian - African | +0.049 (0.032, 0.065) | yes | 0.002 | 0.012 | yes | **yes** |

## B2, per-group standardiser (sensitivity)

a separate standardiser fitted inside each RFW group (sensitivity).

| Group | Images | Identities | Pairs | p_top1 (95% CI) | Kappa (95% CI) | Chance agreement | Largest class (share) | Effective classes (95% CI) |
|---|---|---|---|---|---|---|---|---|
| Caucasian | 10100 | 2950 | 13094 | 0.289 (0.281, 0.300) | 0.136 (0.125, 0.147) | 0.177 | oval (0.287) | 5.65 (5.54, 5.75) |
| Asian | 9624 | 2489 | 13850 | 0.293 (0.283, 0.303) | 0.131 (0.119, 0.143) | 0.186 | oval (0.297) | 5.37 (5.26, 5.48) |
| Indian | 10244 | 2979 | 13270 | 0.298 (0.288, 0.309) | 0.142 (0.131, 0.155) | 0.180 | oval (0.294) | 5.54 (5.43, 5.65) |
| African | 10312 | 2988 | 13503 | 0.278 (0.268, 0.288) | 0.117 (0.105, 0.127) | 0.182 | oval (0.297) | 5.49 (5.38, 5.60) |

Top-1 class shares:

| Group | oval | round | square | rectangle | heart | diamond | triangle |
|---|---|---|---|---|---|---|---|
| Caucasian | 0.287 | 0.139 | 0.200 | 0.094 | 0.102 | 0.074 | 0.103 |
| Asian | 0.297 | 0.136 | 0.222 | 0.077 | 0.108 | 0.073 | 0.088 |
| Indian | 0.294 | 0.137 | 0.205 | 0.097 | 0.092 | 0.086 | 0.090 |
| African | 0.297 | 0.141 | 0.206 | 0.092 | 0.084 | 0.082 | 0.099 |

Multi-probe (p_topk / mean gallery fraction searched), k = 1 to 3:

| Group | k=1: p_topk / gallery | k=2: p_topk / gallery | k=3: p_topk / gallery |
|---|---|---|---|
| Caucasian | 0.289 / 0.177 | 0.487 / 0.344 | 0.631 / 0.482 |
| Asian | 0.293 / 0.186 | 0.493 / 0.351 | 0.631 / 0.487 |
| Indian | 0.298 / 0.181 | 0.508 / 0.344 | 0.654 / 0.484 |
| African | 0.278 / 0.182 | 0.475 / 0.347 | 0.618 / 0.487 |

Between groups:

- p_top1 spread: Indian minus African = +0.020, CI of that pair's difference (0.006, 0.034), CI of replicate max-min (0.009, 0.034). Holm-significant pairs: 1 of 6 (uncorrected: 1 of 6).
- Cohen's kappa spread: Indian minus African = +0.025, CI of that pair's difference (0.010, 0.042), CI of replicate max-min (0.013, 0.042). Holm-significant pairs: 1 of 6 (uncorrected: 2 of 6).

| Metric | Pair | Difference (95% CI) | CI excludes 0 | p (bootstrap) | p (Holm) | Significant, uncorrected | Significant, Holm |
|---|---|---|---|---|---|---|---|
| p_top1 | Caucasian - Asian | -0.003 (-0.017, 0.010) | no | 0.661 | 1 | no | no |
| p_top1 | Caucasian - Indian | -0.008 (-0.023, 0.006) | no | 0.226 | 0.677 | no | no |
| p_top1 | Caucasian - African | +0.011 (-0.003, 0.026) | no | 0.126 | 0.503 | no | no |
| p_top1 | Asian - Indian | -0.005 (-0.019, 0.010) | no | 0.519 | 1 | no | no |
| p_top1 | Asian - African | +0.015 (-0.000, 0.030) | no | 0.0539 | 0.27 | no | no |
| p_top1 | Indian - African | +0.020 (0.006, 0.034) | yes | 0.00599 | 0.036 | yes | **yes** |
| Cohen's kappa | Caucasian - Asian | +0.005 (-0.011, 0.022) | no | 0.591 | 0.819 | no | no |
| Cohen's kappa | Caucasian - Indian | -0.007 (-0.022, 0.009) | no | 0.41 | 0.819 | no | no |
| Cohen's kappa | Caucasian - African | +0.019 (0.003, 0.034) | yes | 0.018 | 0.0899 | yes | no |
| Cohen's kappa | Asian - Indian | -0.011 (-0.027, 0.005) | no | 0.198 | 0.593 | no | no |
| Cohen's kappa | Asian - African | +0.014 (-0.002, 0.032) | no | 0.0819 | 0.328 | no | no |
| Cohen's kappa | Indian - African | +0.025 (0.010, 0.042) | yes | 0.002 | 0.012 | yes | **yes** |

## B3, LFW standardiser transferred

the stage 1 LFW standardiser applied unchanged (transfer).

| Group | Images | Identities | Pairs | p_top1 (95% CI) | Kappa (95% CI) | Chance agreement | Largest class (share) | Effective classes (95% CI) |
|---|---|---|---|---|---|---|---|---|
| Caucasian | 10100 | 2950 | 13094 | 0.287 (0.277, 0.297) | 0.136 (0.125, 0.147) | 0.173 | oval (0.261) | 5.77 (5.66, 5.85) |
| Asian | 9624 | 2489 | 13850 | 0.279 (0.269, 0.289) | 0.125 (0.113, 0.137) | 0.176 | heart (0.230) | 5.68 (5.58, 5.76) |
| Indian | 10244 | 2979 | 13270 | 0.328 (0.317, 0.341) | 0.157 (0.145, 0.170) | 0.202 | square (0.347) | 4.96 (4.82, 5.08) |
| African | 10312 | 2988 | 13503 | 0.328 (0.318, 0.338) | 0.106 (0.094, 0.117) | 0.248 | round (0.368) | 4.03 (3.96, 4.11) |

Top-1 class shares:

| Group | oval | round | square | rectangle | heart | diamond | triangle |
|---|---|---|---|---|---|---|---|
| Caucasian | 0.261 | 0.127 | 0.197 | 0.095 | 0.173 | 0.060 | 0.087 |
| Asian | 0.223 | 0.117 | 0.190 | 0.041 | 0.230 | 0.132 | 0.066 |
| Indian | 0.173 | 0.127 | 0.347 | 0.073 | 0.144 | 0.057 | 0.079 |
| African | 0.250 | 0.368 | 0.140 | 0.020 | 0.167 | 0.012 | 0.043 |

Multi-probe (p_topk / mean gallery fraction searched), k = 1 to 3:

| Group | k=1: p_topk / gallery | k=2: p_topk / gallery | k=3: p_topk / gallery |
|---|---|---|---|
| Caucasian | 0.287 / 0.174 | 0.485 / 0.345 | 0.631 / 0.483 |
| Asian | 0.279 / 0.176 | 0.472 / 0.345 | 0.615 / 0.480 |
| Indian | 0.328 / 0.203 | 0.517 / 0.359 | 0.657 / 0.495 |
| African | 0.328 / 0.248 | 0.558 / 0.464 | 0.737 / 0.651 |

Between groups:

- p_top1 spread: Indian minus Asian = +0.049, CI of that pair's difference (0.034, 0.063), CI of replicate max-min (0.040, 0.064). Holm-significant pairs: 4 of 6 (uncorrected: 4 of 6).
- Cohen's kappa spread: Indian minus African = +0.051, CI of that pair's difference (0.033, 0.068), CI of replicate max-min (0.034, 0.068). Holm-significant pairs: 3 of 6 (uncorrected: 5 of 6).

| Metric | Pair | Difference (95% CI) | CI excludes 0 | p (bootstrap) | p (Holm) | Significant, uncorrected | Significant, Holm |
|---|---|---|---|---|---|---|---|
| p_top1 | Caucasian - Asian | +0.007 (-0.006, 0.020) | no | 0.274 | 0.547 | no | no |
| p_top1 | Caucasian - Indian | -0.041 (-0.055, -0.026) | yes | 0.002 | 0.012 | yes | **yes** |
| p_top1 | Caucasian - African | -0.041 (-0.055, -0.028) | yes | 0.002 | 0.012 | yes | **yes** |
| p_top1 | Asian - Indian | -0.049 (-0.063, -0.034) | yes | 0.002 | 0.012 | yes | **yes** |
| p_top1 | Asian - African | -0.049 (-0.063, -0.035) | yes | 0.002 | 0.012 | yes | **yes** |
| p_top1 | Indian - African | +0.000 (-0.016, 0.014) | no | 0.947 | 0.947 | no | no |
| Cohen's kappa | Caucasian - Asian | +0.011 (-0.005, 0.026) | no | 0.184 | 0.184 | no | no |
| Cohen's kappa | Caucasian - Indian | -0.021 (-0.037, -0.004) | yes | 0.018 | 0.0539 | yes | no |
| Cohen's kappa | Caucasian - African | +0.030 (0.014, 0.046) | yes | 0.002 | 0.012 | yes | **yes** |
| Cohen's kappa | Asian - Indian | -0.032 (-0.047, -0.015) | yes | 0.004 | 0.016 | yes | **yes** |
| Cohen's kappa | Asian - African | +0.019 (0.002, 0.036) | yes | 0.018 | 0.0539 | yes | no |
| Cohen's kappa | Indian - African | +0.051 (0.033, 0.068) | yes | 0.002 | 0.012 | yes | **yes** |

## Standardiser conditions compared

Fraction of RFW images with landmarks whose top-1 class is the same under two conditions:

- B1_pooled vs B2_per_group: 0.797
- B1_pooled vs B3_lfw: 0.818
- B2_per_group vs B3_lfw: 0.742

## Data accounting

Router run: 40607 images, 40301 with landmarks, 0 decode failures, image sizes 400x400, 264 s. Keep rule: landmark detection succeeded, and the identity still has >= 2 such images.

Outputs: `part_b.json`, `stability_by_group.png`, `landmark_failure_by_group.png`, `run_manifest.json`.
