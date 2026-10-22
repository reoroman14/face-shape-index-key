# Pilot stage 2: is the instability in the rule set or in the categories?

The stage 1 result (`results/pilot/VERDICT.md`) is that a geometric seven-class
router agrees with itself across two images of one identity at
p_top1 = 0.327, Cohen's kappa = 0.177.

A reviewer's first objection will be: your hand-written geometric rules are
simply weak. This stage answers that objection, and it is the difference
between a weak paper and a strong one. Write results to `results/stage2/`.

The same rules apply as in PILOT.md: no placeholder numbers, every value
reproducible from a script, seed recorded, failures recorded as failures.

---

## Step 0: enable the GPU

Stage 1 spent 3,053 seconds on ArcFace because `embed.py` requests
`CPUExecutionProvider` with `ctx_id=-1`. The machine has an RTX 4060 with 8 GB.

- Replace `onnxruntime` with `onnxruntime-gpu` in requirements.
- Use `CUDAExecutionProvider` with a CPU fallback, and `ctx_id=0`.
- Verify the provider actually in use at runtime and record it in the manifest.
  Do not assume it: onnxruntime silently falls back to CPU when CUDA libraries
  are missing, which would make this step invisible.
- Drop the second `FaceAnalysis` instance at det_size 640. LFW faces are
  centred in 250x250 images, so the 320 detector carries essentially all of it,
  and the second instance doubles the loaded weights for no measurable gain.
  Record how many images needed the fallback in stage 1 before removing it; if
  that count is not available from the cache, re-measure it once.

Re-run stage 1 after this change and confirm the numbers are unchanged. They
must be: the provider affects speed, not arithmetic. A difference means
something else moved, and that must be explained before continuing.

## Step 1: a learned router

Train a CNN face shape classifier on the public Kaggle Face Shape Dataset
(`lama2019faceshapedataset`): 5,000 images of female celebrities in five
classes, heart, oblong, oval, round, square, with the published 800/200 split
per class.

- Fine-tune a pretrained backbone that fits in 8 GB, for example ResNet-50 or
  EfficientNet-B0.
- Report its accuracy on its own test split. This is the control that matters:
  the published literature reports up to 96.32 percent on this corpus
  (`grd2024transfer`), and the argument only works if our classifier is
  demonstrably competent on the task as the field defines it.
- Save per-image class probabilities, not hard labels.

## Step 2: the same stability measurement

Run the trained CNN over the identical LFW analysis set used in stage 1 and
compute p_top1, Cohen's kappa against the class marginals, the per-class
breakdown and the multi-probe curve, with the identical pairing protocol and
the identical bootstrap over identities.

Produce one table:

| Router | Classes | Own-test accuracy | p_top1 | Chance | Kappa (95% CI) |
|---|---|---|---|---|---|
| geometric | 7 | not applicable | 0.327 | 0.185 | 0.177 (0.161, 0.193) |
| CNN | 5 | measured | measured | measured | measured |

Compare on **kappa**, not on raw agreement. The two routers have different
numbers of classes, so their chance agreement differs and raw p_top1 is not
comparable; kappa corrects for exactly that.

## Step 3: the domain-shift control

Step 2 has a confound that must be handled, not mentioned in passing. Kaggle is
female celebrity portraits; LFW is mostly male news photography. A low CNN kappa
on LFW could mean the categories are unstable, or merely that the CNN is out of
domain. These are different papers.

Control for it in-domain. Take the Kaggle **test** images and apply realistic
within-identity variation that does not change a person's bone structure:

- small yaw and pitch warps, roughly plus or minus 10 to 15 degrees
- mild expression-like deformation around the mouth and jaw
- lighting and contrast changes
- crop and scale jitter of a few percent

Then measure how often the predicted class survives the perturbation, as a
function of perturbation strength. If an in-domain classifier with high test
accuracy flips its label under variation that any two photographs of one person
would exhibit, the instability belongs to the categories and the LFW result is
not a domain artefact.

Report the perturbation parameters exactly. A reader must be able to judge
whether the perturbations are realistic or were tuned to produce the result.

## What each outcome means

- **CNN kappa also near 0.2, and labels flip in-domain under perturbation.**
  The strongest version of the paper. The categories, not the method, are
  unstable, and the claim generalises beyond our rule set.
- **CNN kappa clearly higher, say above 0.5.** The stage 1 finding was partly
  about our geometric rules. The paper narrows to a comparison of routers and
  reports honestly that a learned router is materially more stable, though it
  must still be checked against the recall and penetration that implies.
- **CNN kappa near 0.2 but labels are stable in-domain.** Domain shift is doing
  the work. Neither claim is supported; the study needs a corpus with multiple
  images per identity and face shape labels, which does not currently exist, and
  that gap becomes the paper's contribution.

Write the outcome to `results/stage2/VERDICT.md` with the comparison table and
the perturbation curve, and state which of the three it is.
