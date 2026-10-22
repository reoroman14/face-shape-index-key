# Stage 3 verdict: does the taxonomy behave the same across demographic groups?

**Part A (FairFace, distribution across groups) is reported below. Part B (RFW, per-group stability) has not been run: RFW access has not been obtained. Per-group identity consistency (p_top1, kappa) therefore remains unmeasured; nothing here substitutes for it and LFW does not cover it.**

Generated mechanically by `python -m faceshape_pilot.stage3.verdict` from `part_a.json`. The author makes the final call on wording for the paper.

## Data and router

- FairFace, Hugging Face `HuggingFaceM4/FairFace` at revision `54d573cdb8b5af490ba8da9da2799628f6e5c496`, padding 1.25, 2104494732 bytes, every shard sha256-verified (`fairface_download.json`).
- 97698 images (train 86744, validation 10954). The 108,501 in the FairFace paper include images that were not publicly released; the released train and validation splits are all that is analysed.
- Decode failures: 0. Landmarks found: 90324; no face found: 7374. Image sizes: 448x448.
- Router: the stage 1 code and model file (MediaPipe sha256 `64184e229b263107...`), prototypes unchanged. Primary analysis uses the stage 1 standardiser fitted on LFW (**frozen**); the sensitivity analysis re-fits only the median/IQR standardiser on FairFace (**refit**). Top-1 agreement between the two variants: 0.712.
- Labels are FairFace's annotator-perceived race, gender and age, not self-identification. Race groups: East Asian, Indian, Black, White, Middle Eastern, Latino_Hispanic, Southeast Asian.
- Seed 20261017; bootstrap B = 1000 over images; permutation null with 1000 shuffles.

## 1. Landmark detection failure per group

A detector that fails more often on one group confounds everything downstream, so this comes first. All analyses below are conditional on landmarks being found.

| Race | Images | Failed | Failure rate (95% Wilson CI) |
|---|---|---|---|
| East Asian | 13837 | 983 | 0.0710 (0.0669, 0.0754) |
| Indian | 13835 | 921 | 0.0666 (0.0625, 0.0708) |
| Black | 13789 | 957 | 0.0694 (0.0653, 0.0738) |
| White | 18612 | 1742 | 0.0936 (0.0895, 0.0979) |
| Middle Eastern | 10425 | 1091 | 0.1047 (0.0989, 0.1107) |
| Latino_Hispanic | 14990 | 1027 | 0.0685 (0.0646, 0.0727) |
| Southeast Asian | 12210 | 653 | 0.0535 (0.0496, 0.0576) |
| **all** | 97698 | 7374 | 0.0755 |

Failure vs race: chi-square 336.7, dof 6, p 1.1e-69, Cramer's V 0.059.

| Race (adults) | Images | Failed | Failure rate (95% Wilson CI) |
|---|---|---|---|
| East Asian | 10103 | 809 | 0.0801 (0.0749, 0.0855) |
| Indian | 10220 | 773 | 0.0756 (0.0707, 0.0809) |
| Black | 9436 | 769 | 0.0815 (0.0761, 0.0872) |
| White | 15421 | 1505 | 0.0976 (0.0930, 0.1024) |
| Middle Eastern | 8624 | 981 | 0.1138 (0.1072, 0.1206) |
| Latino_Hispanic | 11191 | 779 | 0.0696 (0.0650, 0.0745) |
| Southeast Asian | 8664 | 501 | 0.0578 (0.0531, 0.0629) |
| **all** | 73659 | 6117 | 0.0830 |

Failure vs race (adults): chi-square 257.4, dof 6, p 1.1e-52, Cramer's V 0.059.

| Gender | Images | Failed | Failure rate (95% Wilson CI) |
|---|---|---|---|
| Male | 51778 | 4531 | 0.0875 (0.0851, 0.0900) |
| Female | 45920 | 2843 | 0.0619 (0.0597, 0.0642) |
| **all** | 97698 | 7374 | 0.0755 |

Failure vs gender: chi-square 228.5, dof 1, p 1.3e-51, Cramer's V 0.048.

| Age | Images | Failed | Failure rate (95% Wilson CI) |
|---|---|---|---|
| 0-2 | 1991 | 38 | 0.0191 (0.0139, 0.0261) |
| 3-9 | 11764 | 561 | 0.0477 (0.0440, 0.0517) |
| 10-19 | 10284 | 658 | 0.0640 (0.0594, 0.0689) |
| 20-29 | 28898 | 2200 | 0.0761 (0.0731, 0.0792) |
| 30-39 | 21580 | 1876 | 0.0869 (0.0832, 0.0908) |
| 40-49 | 12097 | 1084 | 0.0896 (0.0846, 0.0948) |
| 50-59 | 7024 | 624 | 0.0888 (0.0824, 0.0957) |
| 60-69 | 3100 | 271 | 0.0874 (0.0780, 0.0979) |
| more than 70 | 960 | 62 | 0.0646 (0.0507, 0.0819) |
| **all** | 97698 | 7374 | 0.0755 |

Failure vs age: chi-square 341.7, dof 8, p 5.3e-69, Cramer's V 0.059.

Lowest race-group failure rate: Southeast Asian 0.0535; highest: Middle Eastern 0.1047. Plot: `landmark_failure_by_group.png`.

## 2. Seven-class distribution within each group (frozen router)

Shares of each group's faces by top-1 class, with the per-group ceiling on candidate-set reduction 1 / max_k pi_k (bootstrap 95% CI). Per-cell CIs, adjusted residuals and lifts are in `part_a.json`. Plot: `shape_by_group.png`.

| Race | n | oval | round | square | rectangle | heart | diamond | triangle | Largest | 1/max (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| East Asian | 12854 | 0.144 | 0.167 | 0.204 | 0.024 | 0.277 | 0.117 | 0.068 | heart | 3.613 (3.513, 3.715) |
| Indian | 12914 | 0.117 | 0.189 | 0.297 | 0.026 | 0.276 | 0.052 | 0.042 | square | 3.363 (3.275, 3.457) |
| Black | 12832 | 0.150 | 0.280 | 0.138 | 0.015 | 0.354 | 0.031 | 0.032 | heart | 2.824 (2.759, 2.894) |
| White | 16870 | 0.196 | 0.178 | 0.220 | 0.047 | 0.225 | 0.065 | 0.069 | heart | 4.443 (4.311, 4.539) |
| Middle Eastern | 9334 | 0.184 | 0.153 | 0.227 | 0.067 | 0.240 | 0.060 | 0.068 | heart | 4.161 (4.016, 4.304) |
| Latino_Hispanic | 13963 | 0.167 | 0.198 | 0.233 | 0.033 | 0.236 | 0.067 | 0.066 | heart | 4.234 (4.112, 4.314) |
| Southeast Asian | 11557 | 0.137 | 0.203 | 0.203 | 0.023 | 0.275 | 0.086 | 0.073 | heart | 3.639 (3.529, 3.740) |
| **all** | 90324 | 0.157 | 0.196 | 0.218 | 0.033 | 0.268 | 0.068 | 0.060 | heart | 3.735 (3.696, 3.775) |

| Race (adults) | n | oval | round | square | rectangle | heart | diamond | triangle | Largest | 1/max (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| East Asian | 9294 | 0.164 | 0.162 | 0.224 | 0.030 | 0.200 | 0.135 | 0.083 | square | 4.455 (4.294, 4.634) |
| Indian | 9447 | 0.128 | 0.193 | 0.354 | 0.033 | 0.181 | 0.058 | 0.053 | square | 2.828 (2.750, 2.913) |
| Black | 8667 | 0.179 | 0.329 | 0.166 | 0.021 | 0.224 | 0.037 | 0.044 | round | 3.037 (2.945, 3.130) |
| White | 13916 | 0.214 | 0.173 | 0.227 | 0.054 | 0.181 | 0.071 | 0.079 | square | 4.401 (4.275, 4.531) |
| Middle Eastern | 7643 | 0.199 | 0.148 | 0.245 | 0.078 | 0.190 | 0.063 | 0.077 | square | 4.081 (3.927, 4.242) |
| Latino_Hispanic | 10412 | 0.184 | 0.209 | 0.250 | 0.040 | 0.163 | 0.073 | 0.082 | square | 3.992 (3.859, 4.122) |
| Southeast Asian | 8163 | 0.156 | 0.207 | 0.226 | 0.028 | 0.197 | 0.096 | 0.090 | square | 4.427 (4.265, 4.598) |
| **all** | 67542 | 0.177 | 0.201 | 0.242 | 0.041 | 0.189 | 0.076 | 0.073 | square | 4.130 (4.078, 4.187) |

| Gender | n | oval | round | square | rectangle | heart | diamond | triangle | Largest | 1/max (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| Male | 47247 | 0.147 | 0.201 | 0.254 | 0.037 | 0.262 | 0.047 | 0.052 | heart | 3.821 (3.765, 3.879) |
| Female | 43077 | 0.169 | 0.191 | 0.178 | 0.029 | 0.274 | 0.091 | 0.068 | heart | 3.644 (3.591, 3.703) |
| **all** | 90324 | 0.157 | 0.196 | 0.218 | 0.033 | 0.268 | 0.068 | 0.060 | heart | 3.735 (3.697, 3.776) |

| Age | n | oval | round | square | rectangle | heart | diamond | triangle | Largest | 1/max (95% CI) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-2 | 1953 | 0.008 | 0.202 | 0.215 | 0.001 | 0.557 | 0.012 | 0.007 | heart | 1.795 (1.728, 1.866) |
| 3-9 | 11203 | 0.062 | 0.165 | 0.131 | 0.006 | 0.590 | 0.036 | 0.011 | heart | 1.695 (1.669, 1.721) |
| 10-19 | 9626 | 0.158 | 0.195 | 0.148 | 0.018 | 0.384 | 0.064 | 0.034 | heart | 2.607 (2.544, 2.671) |
| 20-29 | 26698 | 0.209 | 0.181 | 0.190 | 0.038 | 0.232 | 0.085 | 0.066 | heart | 4.317 (4.227, 4.411) |
| 30-39 | 19704 | 0.180 | 0.196 | 0.250 | 0.044 | 0.179 | 0.076 | 0.074 | square | 3.993 (3.895, 4.092) |
| 40-49 | 11013 | 0.135 | 0.218 | 0.299 | 0.041 | 0.151 | 0.069 | 0.086 | square | 3.339 (3.249, 3.433) |
| 50-59 | 6400 | 0.130 | 0.244 | 0.315 | 0.043 | 0.128 | 0.061 | 0.079 | square | 3.178 (3.071, 3.291) |
| 60-69 | 2829 | 0.144 | 0.245 | 0.291 | 0.038 | 0.152 | 0.054 | 0.076 | square | 3.433 (3.239, 3.637) |
| more than 70 | 898 | 0.164 | 0.271 | 0.245 | 0.027 | 0.182 | 0.059 | 0.053 | round | 3.695 (3.354, 4.022) |
| **all** | 90324 | 0.157 | 0.196 | 0.218 | 0.033 | 0.268 | 0.068 | 0.060 | heart | 3.735 (3.693, 3.776) |

Race-group ceilings range from 2.82 (Black) to 4.44 (White); pooled 3.73.

## 3. Independence of predicted shape and each attribute

At this sample size every chi-square test is expected to reject; the effect size is Cramer's V. The permutation column gives the V that label shuffling produces at the same n, i.e. what 'no association' looks like here. Cohen bands use w = 0.1 / 0.3 / 0.5 divided by sqrt(min(rows, cols) - 1), a convention, not a criterion derived for this use.

Frozen router (primary):

| Analysis | n | Chi-square | dof | p | Cramer's V (95% CI) | Bias-corrected V (95% CI) | Permutation null V, 99th pct | Cohen band |
|---|---|---|---|---|---|---|---|---|
| race, all ages | 90324 | 4158.3 | 36 | < 1e-300 | 0.088 (0.085, 0.091) | 0.087 (0.085, 0.090) | 0.011 | small |
| race, adults (20+) | 67542 | 3490.4 | 36 | < 1e-300 | 0.093 (0.090, 0.097) | 0.092 (0.089, 0.096) | 0.012 | small |
| gender, all ages | 90324 | 1445.3 | 6 | 3.7e-309 | 0.126 (0.120, 0.133) | 0.126 (0.119, 0.133) | 0.013 | small |
| gender, adults (20+) | 67542 | 1597.1 | 6 | < 1e-300 | 0.154 (0.147, 0.161) | 0.153 (0.147, 0.161) | 0.016 | small |
| age band | 90324 | 12435.4 | 48 | < 1e-300 | 0.151 (0.149, 0.155) | 0.151 (0.149, 0.154) | 0.012 | medium |

Refit standardiser (sensitivity):

| Analysis | n | Chi-square | dof | p | Cramer's V (95% CI) | Bias-corrected V (95% CI) | Permutation null V, 99th pct | Cohen band |
|---|---|---|---|---|---|---|---|---|
| race, all ages | 90324 | 4616.0 | 36 | < 1e-300 | 0.092 (0.090, 0.095) | 0.092 (0.089, 0.095) | 0.010 | small |
| race, adults (20+) | 67542 | 3840.6 | 36 | < 1e-300 | 0.097 (0.094, 0.101) | 0.097 (0.094, 0.101) | 0.012 | small |
| gender, all ages | 90324 | 1478.9 | 6 | < 1e-300 | 0.128 (0.122, 0.135) | 0.128 (0.122, 0.134) | 0.013 | small |
| gender, adults (20+) | 67542 | 1595.3 | 6 | < 1e-300 | 0.154 (0.146, 0.161) | 0.153 (0.146, 0.161) | 0.017 | small |
| age band | 90324 | 14776.8 | 48 | < 1e-300 | 0.165 (0.162, 0.168) | 0.165 (0.162, 0.168) | 0.012 | medium |

## 4. Reverse direction: how well does the shape output predict the group?

Fitted on the FairFace train split, evaluated once on the validation split (official split), images with landmarks. Information gain is the out-of-sample reduction in cross-entropy relative to the train-set group prior, as a fraction of the prior's cross-entropy (0 = the label tells you nothing, 1 = it determines the group). Balanced accuracy for the shape label uses the rule argmax P(group | label) / P(group).

| Target | Predictor | n test | Accuracy (95% CI) | Majority baseline | Balanced acc. (95% CI) | Chance | Macro AUC (95% CI) | Info gain, fraction of prior entropy (95% CI) |
|---|---|---|---|---|---|---|---|---|
| race, all ages | top-1 shape label | 10118 | 0.212 (0.204, 0.220) | 0.187 | 0.203 (0.197, 0.210) | 0.143 | 0.566 (0.560, 0.572) | 0.013 (0.011, 0.015) |
| race, all ages | 7 shape probabilities | 10118 | 0.225 (0.217, 0.233) | 0.187 | 0.216 (0.210, 0.224) | 0.143 | 0.592 (0.586, 0.597) | 0.021 (0.019, 0.024) |
| race, all ages | 4 ratio features (L, F, J, A) | 10118 | 0.236 (0.228, 0.244) | 0.187 | 0.233 (0.224, 0.240) | 0.143 | 0.610 (0.604, 0.617) | 0.035 (0.031, 0.039) |
| race, adults (20+) | top-1 shape label | 7541 | 0.222 (0.214, 0.232) | 0.201 | 0.212 (0.203, 0.220) | 0.143 | 0.568 (0.561, 0.575) | 0.015 (0.013, 0.018) |
| race, adults (20+) | 7 shape probabilities | 7541 | 0.229 (0.220, 0.239) | 0.201 | 0.223 (0.215, 0.231) | 0.143 | 0.598 (0.591, 0.606) | 0.024 (0.021, 0.028) |
| race, adults (20+) | 4 ratio features (L, F, J, A) | 7541 | 0.240 (0.230, 0.250) | 0.201 | 0.242 (0.232, 0.251) | 0.143 | 0.611 (0.604, 0.619) | 0.035 (0.030, 0.039) |
| gender, all ages | top-1 shape label | 10118 | 0.550 (0.541, 0.560) | 0.523 | 0.549 (0.540, 0.559) | 0.500 | 0.568 (0.557, 0.578) | 0.011 (0.008, 0.015) |
| gender, all ages | 7 shape probabilities | 10118 | 0.576 (0.567, 0.585) | 0.523 | 0.576 (0.567, 0.585) | 0.500 | 0.602 (0.592, 0.613) | 0.021 (0.016, 0.026) |
| gender, all ages | 4 ratio features (L, F, J, A) | 10118 | 0.585 (0.575, 0.594) | 0.523 | 0.584 (0.574, 0.593) | 0.500 | 0.620 (0.608, 0.630) | 0.032 (0.026, 0.038) |
| gender, adults (20+) | top-1 shape label | 7541 | 0.560 (0.548, 0.571) | 0.527 | 0.561 (0.550, 0.572) | 0.500 | 0.580 (0.567, 0.592) | 0.015 (0.010, 0.020) |
| gender, adults (20+) | 7 shape probabilities | 7541 | 0.590 (0.579, 0.602) | 0.527 | 0.587 (0.577, 0.598) | 0.500 | 0.617 (0.605, 0.631) | 0.027 (0.021, 0.033) |
| gender, adults (20+) | 4 ratio features (L, F, J, A) | 7541 | 0.599 (0.588, 0.609) | 0.527 | 0.595 (0.584, 0.605) | 0.500 | 0.634 (0.621, 0.646) | 0.040 (0.032, 0.048) |
| age band | top-1 shape label | 10118 | 0.306 (0.297, 0.315) | 0.302 | 0.176 (0.165, 0.186) | 0.111 | 0.618 (0.610, 0.626) | 0.037 (0.034, 0.041) |
| age band | 7 shape probabilities | 10118 | 0.324 (0.314, 0.333) | 0.302 | 0.221 (0.210, 0.231) | 0.111 | 0.657 (0.648, 0.666) | 0.058 (0.053, 0.063) |
| age band | 4 ratio features (L, F, J, A) | 10118 | 0.335 (0.327, 0.344) | 0.302 | 0.239 (0.227, 0.253) | 0.111 | 0.659 (0.651, 0.668) | 0.070 (0.065, 0.076) |

Lift P(race | shape label) / P(race) from the train split (frozen router, all ages); 1.00 = the label carries no information about that group:

| Shape label | East Asian | Indian | Black | White | Middle Eastern | Latino_Hispanic | Southeast Asian |
|---|---|---|---|---|---|---|---|
| oval | 0.91 | 0.75 | 0.96 | 1.24 | 1.17 | 1.06 | 0.87 |
| round | 0.86 | 0.96 | 1.41 | 0.91 | 0.78 | 1.01 | 1.04 |
| square | 0.94 | 1.35 | 0.63 | 1.01 | 1.04 | 1.07 | 0.94 |
| rectangle | 0.72 | 0.81 | 0.47 | 1.43 | 2.01 | 0.99 | 0.69 |
| heart | 1.03 | 1.03 | 1.33 | 0.84 | 0.89 | 0.89 | 1.02 |
| diamond | 1.71 | 0.76 | 0.47 | 0.96 | 0.89 | 0.97 | 1.25 |
| triangle | 1.14 | 0.72 | 0.53 | 1.14 | 1.15 | 1.10 | 1.22 |

Largest lift: a face labelled rectangle is 2.01 times as likely as the base rate to carry the Middle Eastern label; smallest: diamond -> Black at 0.47.

Refit standardiser, race:

| Target | Predictor | n test | Accuracy (95% CI) | Majority baseline | Balanced acc. (95% CI) | Chance | Macro AUC (95% CI) | Info gain, fraction of prior entropy (95% CI) |
|---|---|---|---|---|---|---|---|---|
| race, all ages | top-1 shape label | 10118 | 0.216 (0.208, 0.224) | 0.187 | 0.204 (0.198, 0.211) | 0.143 | 0.568 (0.562, 0.574) | 0.014 (0.012, 0.017) |
| race, all ages | 7 shape probabilities | 10118 | 0.228 (0.220, 0.236) | 0.187 | 0.225 (0.217, 0.232) | 0.143 | 0.597 (0.592, 0.603) | 0.024 (0.021, 0.027) |
| race, all ages | 4 ratio features (L, F, J, A) | 10118 | 0.236 (0.228, 0.244) | 0.187 | 0.233 (0.224, 0.240) | 0.143 | 0.610 (0.604, 0.617) | 0.035 (0.031, 0.039) |
| race, adults (20+) | top-1 shape label | 7541 | 0.226 (0.217, 0.236) | 0.201 | 0.212 (0.204, 0.220) | 0.143 | 0.569 (0.562, 0.576) | 0.016 (0.013, 0.019) |
| race, adults (20+) | 7 shape probabilities | 7541 | 0.236 (0.226, 0.245) | 0.201 | 0.232 (0.224, 0.240) | 0.143 | 0.603 (0.596, 0.611) | 0.027 (0.024, 0.031) |
| race, adults (20+) | 4 ratio features (L, F, J, A) | 7541 | 0.240 (0.230, 0.250) | 0.201 | 0.242 (0.232, 0.251) | 0.143 | 0.611 (0.604, 0.619) | 0.035 (0.030, 0.039) |

## 5. What the paper can and cannot now claim about demographic generality

On FairFace the router's top-1 shape is associated with annotated race with Cramer's V 0.088 (0.085, 0.091) (small by Cohen's convention; 0.093 (0.090, 0.097) on adults), against a no-association value of at most 0.011 at this sample size. In the reverse direction the shape label alone predicts race on held-out images with balanced accuracy 0.203 (0.197, 0.210) against chance 0.143, macro AUC 0.566 (0.560, 0.572), and removes 1.3 percent (1.1 to 1.5) of the uncertainty about race (adults: balanced accuracy 0.212, 1.5 percent); the four ratio features underneath it reach balanced accuracy 0.233 and macro AUC 0.610. Landmark detection fails on 5.3 to 10.5 percent of images depending on race group. For comparison, the association of the same shape label with annotated gender is V 0.154 on adults and with age band V 0.151; race is not the attribute the label tracks most strongly. The paper can therefore state the measured size of the association between the shape key and perceived race in both directions, and that a shape-keyed index is to that measured extent an ethnicity-keyed index, which bears directly on the forensic exclusion use case. It cannot claim that the stage 1 instability result (kappa on LFW) generalises across demographic groups: per-group stability needs identity labels, Part B (RFW) has not been run, and FairFace cannot supply it. The limitations section should say plainly that per-group stability remains unmeasured.

## Caveats

- Race, gender and age are annotator-perceived labels from FairFace, not self-identification; the seven race categories are FairFace's.
- FairFace has no identity labels; the same person may appear more than once, so bootstrap intervals that resample images may be somewhat too narrow, and a person could appear in both the train and validation splits of the reverse-direction test.
- FairFace images are Flickr photographs (YFCC-100M) with uncontrolled pose, expression and resolution; part of any group difference may come from how photographs of each group were taken and selected rather than from face geometry. The router cannot separate the two.
- The rules were written for adult faces; children's age bands are reported but the adult-only analyses are the relevant ones for the forensic use case.
- Analyses are conditional on MediaPipe finding landmarks; group differences in failure rate (section 1) change which faces enter the cross-tabulations.
- The padding 1.25 crops were used so that forehead and chin landmarks are inside the image; the tighter padding 0.25 release was not analysed.

Seed 20261017. Generated 2026-10-18T18:54:34+00:00.
