# Face shape is not an index key

Code and result files for the paper *Face Shape Is Not an Index Key:
Within-Identity Instability in Seven-Class Face Shape Routing*.

The paper asks whether a seven-class face shape label (oval, round, square,
rectangle, heart, diamond, triangle) can be used to partition a face gallery so
that identification searches only one bucket. Published face shape work reports
classification accuracy, which cannot answer this, because the standard corpus
assigns one fixed label per identity. The decisive quantity is whether two
photographs of the same person receive the same class. We measure it.

## What was found

| Measurement | Value |
|---|---|
| Within-identity top-1 agreement on LFW | 0.327 (95% CI 0.314 to 0.341) |
| Chance agreement from class marginals | 0.185 |
| Cohen's kappa | **0.177** (0.161 to 0.193) |
| Human inter-observer kappa, published | 0.203 |
| CNN router: Kaggle test accuracy, then LFW kappa | 0.906, then 0.256 with 0.704 of images in one class |
| Faces relabelled when the standardiser is refit | 0.288 of 97,698 |
| Per-group kappa across four RFW subsets | 0.106 to 0.157 |

Under a single shared calibration the four RFW groups are not treated alike:
the African subset receives two of seven classes for 0.664 of its faces, has
the highest raw agreement (0.334) and the lowest kappa (0.109), and recovers
its mates only by searching 0.252 of the gallery against 0.176 to 0.191 for the
others. Refitting the identical rules inside each group removes most of that
gap, which locates the effect in the calibration reference rather than in
facial geometry.

## Reproducing

Every number is produced under `SEED = 20261017`. Measurements 1 and 2 are
bit-identical between CPU and GPU; measurement 3 is not, because embedding
differences of order 1e-3 move k-means to a different local optimum. The
reported figures are from the GPU run.

```
pip install -r requirements.txt

python -m faceshape_pilot.run_pilot            # stage 1: LFW, geometric router
python -m faceshape_pilot.stage2.run_stage2    # stage 2: learned router, transfer
python -m faceshape_pilot.stage3.run_stage3    # stage 3: FairFace, recalibration
python -m faceshape_pilot.stage4.run_stage4    # stage 4: RFW, per-group stability
```

Each stage writes a `VERDICT.md`, a `run_manifest.json` and the result JSON
files under `results/`. The specifications the stages were written against are
in `PILOT.md`, `PILOT2.md`, `PILOT3.md` and `PILOT4.md`.

## Data

No dataset is redistributed here.

- **LFW** is downloaded automatically by `sklearn.datasets.fetch_lfw_people`.
- **The five-class face shape corpus** is the Kaggle dataset
  `niten19/face-shape-dataset`; the pipeline downloads it with the Kaggle API.
- **FairFace** is pulled from Hugging Face at a pinned revision
  (`HuggingFaceM4/FairFace`, revision `54d573cd`, padding 1.25).
- **RFW** is not public. It is obtained by request from its authors at
  http://www.whdeng.cn/RFW/index.html. We hold written permission from the
  corpus maintainer. RFW images derive from MS-Celeb-1M, which Microsoft
  withdrew in 2019; the paper's Ethical Considerations section sets out why we
  used it and what we did not download. Place the four test subsets under
  `data/rfw/images/` and the pipeline verifies the tree against the official
  counts (40,607 images, 11,429 identities) before running.

## Layout

```
src/faceshape_pilot/      router, measurements, and the four stages
results/pilot/            stage 1
results/stage2/           stage 2
results/stage3/           stage 3
results/stage4/           stage 4
figures/                  the three figures used in the paper
PILOT*.md                 the stage specifications
```

Cached landmark and embedding arrays and two large per-image CSV dumps are
excluded from the repository; rerunning the stage that produced them
regenerates them.

## Citation

A citation will be added here once the paper has a DOI.

## License

MIT, see `LICENSE`. The result files are released under the same terms. Dataset
terms are set by each dataset's own licence and are not altered by this
repository.
