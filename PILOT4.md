# Stage 4, Part B: per-group within-identity stability on RFW

## Why this stage exists

Section 6 measures within-identity stability on LFW: p_top1 = 0.327, kappa =
0.177. Section 8 measures how the taxonomy distributes across annotated race on
FairFace, but it cannot measure stability there, because p_top1 needs several
images per identity and FairFace carries no identity labels.

That leaves the sharpest version of the fairness question unanswered. The paper
currently argues that an exclusion procedure keyed on shape would not
distribute its errors evenly, and supports that only with per-cell lifts from an
aggregate association. If routing failures concentrate in one demographic
group, the harm is concrete rather than inferred: a person in that group is
more likely to be missed by the system than a person in another group, under
the same router and the same operating point.

RFW is the one corpus that answers this directly. It carries identity labels
and a group label, four subsets, roughly 2,500 to 3,000 identities each, every
identity with at least two images.

This stage measures p_top1 and kappa separately for Caucasian, Asian, Indian
and African, and reports whether the differences survive a bootstrap over
identities.

## Data

Already downloaded and verified:

```
C:\FaceShape\data\rfw\images\
  data\<Group>\<identityID>\<identityID>_<faceID>.jpg
  txts\<Group>\<Group>_images.txt
```

Group is one of Caucasian, Asian, Indian, African.

Verified counts from the official `*_images.txt` lists:

| subset | images | identities | identities with >= 2 images |
|---|---|---|---|
| Caucasian | 10,196 | 2,958 | 2,958 |
| Asian | 9,688 | 2,492 | 2,492 |
| Indian | 10,308 | 2,984 | 2,984 |
| African | 10,415 | 2,995 | 2,995 |
| total | 40,607 | 11,429 | 11,429 |

Every identity is pairable. Under `MAX_PAIRS_PER_IDENTITY = 10` this gives
roughly 13,300 to 14,000 within-identity pairs per group, so the four groups
are balanced to within 5 percent of each other on pair count. No subsampling to
equalise group size is applied; the bootstrap over identities carries the
difference in identity count correctly.

Enumerate images by walking the `data\<Group>` tree, not by reading the txt
lists. Cross-check the resulting counts against the table above and fail loudly
on any mismatch, in the same style as `data.load_lfw_index`.

Do not read `*_people.txt` as an identity list. Per the RFW readme, that file
is the set of identities overlapping MS-Celeb-1M, not the full roster.

## Provenance note that must appear in the output

RFW images are selected from MS-Celeb-1M, which its creator withdrew in April
2019. Access to RFW was granted in writing by Mei Wang (Beijing University of
Posts and Telecommunications) on 2026-10-20 in response to a request from
Applied Science Private University. The corpus was taken from the Google Drive
link the maintainer supplied, not from a mirror, and only the four test subsets
were downloaded; the MS1M-wo-RFW and BUPT training sets distributed alongside
them were not. The unpacked tree was verified file by file against the official
counts (40,607 images, 11,429 identities) before use. The maintainer's
`readme.txt` is kept at `data/rfw/readme.txt`. Every result file this stage writes must
carry a `provenance` block recording this. The paper's Ethical Considerations
section is amended in the same commit.

## What to compute

`m1_consistency.compute(ds)` is already dataset-agnostic. It needs an object
exposing `.probs`, `.top1`, `.identity`, `.n_images`, `.accounting` and
`.images_of()`. Do not modify `m1_consistency.py`.

### Do not run the embedder

`AnalysisSet` loads both the router cache and the ArcFace embedding cache,
because measurement 3 needs embeddings. Part B calls only measurement 1, which
never touches `.embeddings`. Running ArcFace over 40,607 images would cost
hours and change nothing. Build a separate lightweight set class for RFW that
mirrors only the interface above, and keep the `keep` rule as: landmark
detection succeeded, and the identity still has at least two surviving images.

### Three standardiser conditions

`router.fit_standardiser` is fit on whatever ratio features it is given, so the
choice of fitting population is itself a scientific decision, and the stage-3
recalibration result (0.288 of faces change class when the reference population
changes) means the three conditions below are not interchangeable.

Extract landmarks once. Compute `ratio_features` once. Then apply three
different affine transforms to the cached ratios and re-run the softmax. The
marginal cost of conditions 2 and 3 is negligible.

**Condition B1, pooled RFW standardiser (primary).** Fit one standardiser on
the ratio features of all four subsets pooled, apply it to all four. This is
the deployment-realistic case: one router, one calibration, one mixed
population. Per-group differences measured here are the headline result, and
they are the ones that bear on the exclusion-tool argument.

**Condition B2, per-group standardiser.** Fit a separate standardiser inside
each subset. This asks whether a group-specific calibration rescues stability,
which is the obvious objection to B1. Report it as a sensitivity analysis.

**Condition B3, LFW standardiser transferred.** Load the standardiser already
cached from the LFW run (`results/pilot/cache/router_meta.json` holds
`standardiser`, and `router.npz` holds `standardiser_median` and
`standardiser_scale`). Apply it unchanged to RFW ratio features. This measures
what happens when a router calibrated on a demographically narrow population
meets a diverse one, which is exactly the deployment failure the paper warns
about.

Report all three. Lead with B1.

### Per group, per condition

- `p_top1` with 95 percent percentile bootstrap CI over identities
- `cohen_kappa_top1` with the same CI
- `chance_agreement_from_marginals`
- class distribution, and effective number of classes as
  `1 / sum(share^2)`, to detect collapse of the kind stage 2 found
- the full `multi_probe` table and `per_class` block that
  `m1_consistency.compute` already returns

### Between-group comparison

Point estimates that differ are not a finding on their own. For each of the six
group pairs, and for p_top1 and kappa separately:

- resample identities independently within each of the two groups, same
  `N_BOOTSTRAP = 1000`, same `SEED`
- report the difference, its 95 percent CI, and whether the CI excludes zero
- also report the spread, max group minus min group, with its CI

Apply a Holm correction across the six pairwise comparisons and report both
corrected and uncorrected decisions. Do not describe a difference as
significant on the uncorrected value alone.

### Detector failure rate per group

Record, per group, how many images MediaPipe failed to landmark and how many
identities consequently fell below two images. Report the rate per group and
bootstrap a CI on the difference between the highest and lowest group.

This is a result in its own right and must be reported whichever way it comes
out. A face mesh that fails more often on one group is a second, independent
channel of unequal treatment, upstream of the shape label and not visible in
any accuracy number. If the rates are flat, say so plainly.

## Step 0 before the full run

Run landmark extraction on the first 200 images of each group and print the
detection rate per group. RFW crops are looser than LFW funneled crops, and if
the rate is far below the LFW rate the geometry is not comparable and the full
run should not start. Stop and report rather than continuing.

Expected LFW reference: check `results/pilot/cache/router_meta.json` for
`n_landmark_ok` against `n_images` and print both numbers alongside the RFW
sample rates.

## Config additions

```python
# Stage 4 (PILOT4.md), Part B: RFW per-group stability.
# RFW test set, loosely cropped .jpg images, obtained from the authors.
# Images derive from MS-Celeb-1M, which was withdrawn by Microsoft in 2019;
# access granted in writing by Mei Wang on 2026-10-20.
STAGE4_DIR = REPO_ROOT / "results" / "stage4"
STAGE4_CACHE = STAGE4_DIR / "cache"
RFW_DIR = DATA_DIR / "rfw" / "images"
RFW_GROUPS = ("Caucasian", "Asian", "Indian", "African")
RFW_EXPECTED = {
    "Caucasian": (10196, 2958),
    "Asian": (9688, 2492),
    "Indian": (10308, 2984),
    "African": (10415, 2995),
}
```

`run_header` hardcodes the LFW dataset string. Give it a `dataset` argument
defaulting to the current string, so stage 4 can pass its own, rather than
writing a misleading header.

## Files to create

```
src/faceshape_pilot/stage4/__init__.py
src/faceshape_pilot/stage4/rfw.py            # enumeration, verification, RFWSet
src/faceshape_pilot/stage4/extract_rfw.py    # landmarks + ratio features, cached
src/faceshape_pilot/stage4/part_b.py         # the three conditions, per-group stats, pairwise tests
src/faceshape_pilot/stage4/verdict.py        # VERDICT.md
src/faceshape_pilot/stage4/run_stage4.py     # entry point, --manifest-only, --step0
```

Outputs under `results/stage4/`: `part_b.json`, `VERDICT.md`,
`run_manifest.json`, plots.

## Reporting rules

Carry over from the earlier stages, and they are not negotiable:

- `SEED = 20261017` everywhere. No unseeded randomness.
- Failures are recorded as failures. Nothing is imputed.
- Every result file carries the environment block and the provenance block.
- The verdict reports what was measured, not what was hoped for. If per-group
  stability is flat, that is the finding and the paper says the exclusion
  concern is not supported by per-group measurement. If it is uneven, that is
  the finding and the paper says so with the CI attached.
- Do not write anything into `main.tex`. This stage produces numbers; the
  paper is edited separately after the numbers are reviewed.
