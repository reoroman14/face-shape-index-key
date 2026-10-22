# Pilot study: is face shape viable as an index key?

Purpose: decide in days, not months, whether the paper's premise holds, before
committing to the expensive multi-annotator dataset. Three measurements. No
annotation required. Runs on an RTX 4060 (8 GB).

Write results to `results/pilot/`. Do not write any number into `main.tex`
until it has been produced by a run in this repository.

---

## Why these three numbers

The paper claims a face shape router can prune a gallery without losing the
true match. Three things must hold, and the literature gives reason to doubt
each one.

1. The router must give the **same** label to two photos of the same person.
   Published face shape accuracy (around 96 percent on the Kaggle set) does not
   measure this at all: that dataset assigns one fixed label per celebrity, so
   stability across pose, expression, hair and camera is never tested. A
   forensic reliability study of a face shape system reported mean Cohen's
   kappa 0.203 across observers (`valladares2025reliability`), which suggests
   the categories themselves may be unstable.
2. The classes must be **reasonably balanced**, or the achievable reduction is
   small whatever the accuracy.
3. Shape must carry information the embedding does **not already encode**.
   Terhorst et al. (`terhorst2021soft`) show deep face embeddings retain
   substantial soft-biometric content. If a learned partition of the embedding
   space separates identities at least as well as shape does, an explicit shape
   router is redundant and the paper has no mechanism.

---

## Data

LFW, restricted to identities with at least two images (roughly 1,680
identities). Obtain via `sklearn.datasets.fetch_lfw_people(min_faces_per_person=2)`
or the deepface loader.

LFW is demographically skewed and consists of public figures. That is
acceptable for a pilot, which asks whether the mechanism works at all. It is
not acceptable for the fairness analysis in the paper, which needs a balanced
set. Record this limitation in the results file.

## Components

- Landmark router: MediaPipe Face Mesh (468 points), CPU. Derive forehead
  width, cheekbone width, jaw width, face length and jaw angle; classify into
  the seven classes with explicit geometric rules, and emit a probability
  distribution rather than a hard label.
- Embedding: ArcFace via `insightface` or `deepface`. One embedding per image.
- Keep the router and the embedding independent. Do not train the router on
  identity labels.

---

## Measurement 1: identity consistency

For every identity, form all within-identity image pairs (cap at 10 pairs per
identity to avoid letting a few large classes dominate).

    p_top1 = fraction of pairs whose two images receive the same top-1 shape

`p_top1` upper-bounds hard-routing recall: if a probe and its gallery mate get
different labels, the mate is never compared and no matcher can recover it.

Also report, for `k = 2, 3`:

    p_topk = fraction of pairs where the gallery image's top-1 shape
             falls within the probe's top-k shapes

This is the multi-probe policy. Report the mean number of classes searched at
each k, since that is the cost side of the trade.

Output: `results/pilot/consistency.json`, plus a plot of p_topk against mean
classes searched.

## Measurement 2: class distribution

Distribution of top-1 shape over all images, and over identities.

    max_reduction = 1 / (share of the largest class)

Report the distribution, `max_reduction`, and the entropy of the distribution.
A heavily oval-and-round distribution caps the achievable speed-up regardless
of how good the router is.

Output: `results/pilot/distribution.json` and a bar chart.

## Measurement 3: shape versus a learned partition

This is the decisive one. Partition the same images two ways into the same
number of buckets:

- by predicted face shape (7 buckets)
- by k-means on the ArcFace embeddings, k = 7, which is what an IVF coarse
  quantiser does

For each partitioning, over held-out probes:

    routing_recall = fraction of probes whose gallery mate is in a searched bucket
    penetration    = mean fraction of the gallery searched

Plot routing_recall against penetration for both, sweeping the number of
buckets probed. The two curves answer the paper's central question directly.

Also run a linear probe: logistic regression from ArcFace embedding to the
router's shape label, reporting balanced accuracy with a proper train/test
split. High accuracy means shape is already recoverable from the embedding and
adds no independent signal.

Output: `results/pilot/partition_comparison.json` and the recall-penetration plot.

---

## Decision rule

Write the outcome into `results/pilot/VERDICT.md` as one of:

- **Green.** `p_top1` above about 0.85, largest class below about 0.35, and the
  shape curve at or above the k-means curve. Proceed to the full study as
  specified in `main.tex`.
- **Amber.** `p_top1` between roughly 0.6 and 0.85, or shape close to but below
  k-means. The hard-partition framing is dead but multi-probe or conformal
  routing may survive. Rewrite the method around soft routing before scaling up.
- **Red.** `p_top1` below about 0.6, or k-means clearly dominates, or the linear
  probe recovers shape with high accuracy. The mechanism does not work. Pivot
  the paper to a measured negative result: a 2022 paper asserts that rough face
  shape filtering improves recognition speed and accuracy without testing it
  (`duan2022faceshape`), and showing carefully that it does not is a genuine
  contribution, especially paired with the annotation-agreement evidence.

A red outcome is not a failed project. It is a cheaper, more honest paper, and
it is reached in days rather than after building a dataset.

---

## Rules for this pilot

- Fix and record the random seed. Report the split.
- Any number appearing in the paper must be reproducible from a script here.
- Never write a placeholder or expected number into the paper. Empty is correct
  until measured.
- Report the per-class breakdown of `p_top1`. The forensic study found round and
  oval agreed well and the rest did not; expect the same pattern and record it.
