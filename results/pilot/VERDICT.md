# Pilot verdict

**Provisional call: Red.** Derived mechanically from the result files by `python -m faceshape_pilot.verdict`; the author makes the final call.

Reasons:
- p_top1 = 0.327 is below 0.6

## Criteria

| Quantity | Value | 95% CI | Threshold |
|---|---|---|---|
| p_top1 (identity consistency) | 0.327 | [0.314, 0.341] | Green > 0.85; Red < 0.6 |
| Largest class share (images) | 0.303 |  | Green < 0.35 |
| AUC shape - AUC k-means | +0.006 | [-0.008, +0.019] | Green >= -0.01; Red < -0.05 with CI below 0 |
| Linear probe balanced accuracy | 0.249 |  | Red >= 0.7 (chance 0.143) |

Thresholds marked in PILOT.md as approximate or verbal were operationalised as follows (constants at the top of `src/faceshape_pilot/verdict.py`): "at or above" = AUC difference >= -0.01; "clearly dominates" = AUC difference < -0.05 with the bootstrap CI below zero; "high accuracy" = balanced accuracy >= 0.7.

## Shape versus k-means at real operating points

The AUC row above joins (0, 0) to each curve's first operating point by a straight line. The tables below compare the curves only where a partition actually has an operating point, interpolating the other curve.

At k-means operating points:

| k-means buckets probed | Penetration | k-means recall | Shape recall (interpolated) | Shape - k-means |
|---|---|---|---|---|
| 1 | 0.422 | 0.859 | 0.780 | -0.079 |
| 2 | 0.847 | 1.000 | 0.962 | -0.038 |
| 3 | 0.878 | 1.000 | 0.971 | -0.029 |
| 4 | 0.909 | 1.000 | 0.979 | -0.021 |
| 5 | 0.939 | 1.000 | 0.986 | -0.014 |
| 6 | 0.970 | 1.000 | 0.993 | -0.007 |
| 7 | 1.000 | 1.000 | 1.000 | +0.000 |

At shape operating points:

| Shape buckets probed | Penetration | Shape recall | k-means recall (interpolated) | Shape - k-means |
|---|---|---|---|---|
| 1 | 0.183 | 0.535 | 0.373 | +0.163 |
| 2 | 0.349 | 0.726 | 0.710 | +0.016 |
| 3 | 0.482 | 0.824 | 0.879 | -0.055 |
| 4 | 0.616 | 0.883 | 0.923 | -0.040 |
| 5 | 0.754 | 0.933 | 0.969 | -0.036 |
| 6 | 0.888 | 0.974 | 1.000 | -0.026 |
| 7 | 1.000 | 1.000 | 1.000 | +0.000 |

Gallery bucket shares: shape 0.308, 0.119, 0.207, 0.079, 0.100, 0.076, 0.111; k-means 0.019, 0.440, 0.408, 0.016, 0.014, 0.031, 0.071.

## Per-class identity consistency

| Class | Share of images | Ordered pairs (probe in class) | p_top1 given probe class | 95% CI |
|---|---|---|---|---|
| oval | 0.303 | 4065 | 0.413 | [0.387, 0.437] |
| round | 0.119 | 1720 | 0.264 | [0.232, 0.295] |
| square | 0.211 | 3080 | 0.409 | [0.378, 0.439] |
| rectangle | 0.080 | 1147 | 0.265 | [0.216, 0.309] |
| heart | 0.102 | 1465 | 0.238 | [0.197, 0.279] |
| diamond | 0.075 | 1050 | 0.181 | [0.143, 0.218] |
| triangle | 0.110 | 1453 | 0.234 | [0.194, 0.275] |

Cohen's kappa on pair labels: 0.177 [0.161, 0.193]; chance agreement from the class marginals: 0.185.

## Other key numbers

Multi-probe (from `consistency.json`):

| k | p_topk | Mean fraction of gallery searched |
|---|---|---|
| 1 | 0.327 | 0.184 |
| 2 | 0.529 | 0.350 |
| 3 | 0.669 | 0.484 |

Class distribution over images: largest class oval, max_reduction 3.30x, entropy 2.62 of 2.81 bits. Over identities: largest class oval, max_reduction 2.97x.

Routing recall at one bucket probed: shape 0.535 at penetration 0.183; k-means 0.859 at penetration 0.422.

## Data

LFW (funneled), identities with at least two images: 9164 images, 1680 identities. Landmark failures 10, embedding failures 0. Analysis set after removing failures and identities left with fewer than two images: 9151 images, 1677 identities. Seed 20261017. Probe/gallery split: 1677 probes, 7474 gallery images (`partition_split.csv`).

## Limitations

- LFW is demographically skewed (predominantly male, lighter-skinned, middle-aged public figures) and consists of news photographs of public figures. This is acceptable for a pilot that asks whether the routing mechanism works at all. It is not acceptable for the fairness analysis in the paper, which requires a demographically balanced set.
- The router is one explicit rule set (prototypes in `router.py`) fixed before the run. Its identity consistency and class balance are properties of this rule set on LFW; a different rule set or a learned router (CNN, transformer) may behave differently.
- Router features are standardised over the pilot images, so class balance is partly a design consequence; see the note in `distribution.json`.
- The linear probe predicts the router's label, not a human label. Low router consistency caps what any identity-centred embedding can predict, so a low probe score does not by itself show that the embedding lacks shape information.
- No human face shape labels are used anywhere in this pilot.

Generated 2026-10-17T21:01:08+00:00.
