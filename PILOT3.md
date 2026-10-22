# Stage 3: does the taxonomy behave the same across demographic groups?

The paper currently states that LFW is demographically narrow and makes no
fairness claim. A reviewer will ask whether the conclusion holds for faces
outside that distribution. This stage answers it.

**Correction to an earlier plan.** Re-running stage 1 on FairFace was proposed
and is not possible. FairFace has no identity labels and reports no count of
individuals; its images come from YFCC-100M with per-image race, gender and age
only. The p_top1 statistic needs two images of the same person, so FairFace
cannot produce it. The stage below therefore splits into two parts that answer
different questions with different corpora.

Write results to `results/stage3/`. Same rules as before: no placeholder
numbers, every value reproducible, seed recorded, failures recorded as failures.

---

## Part A: distribution across groups (FairFace, no identities needed)

Run the existing geometric router over FairFace and cross-tabulate the
predicted shape against the supplied race, gender and age labels.

Report:

- The seven-class distribution within each of the seven race groups, each
  gender and each age band.
- A test of independence between predicted shape and each attribute
  (chi-square with Cramer's V, which is interpretable at this sample size),
  with bootstrap confidence intervals.
- The per-group ceiling on candidate-set reduction, `1 / max_k pi_k`, since a
  group whose faces concentrate in one class gets less pruning than the pooled
  figure suggests.
- Landmark detection failure rate per group. A detector that fails more often
  on one group is itself a finding and would confound the rest.

Why this matters beyond fairness reporting. The router's prototypes were fixed
before any measurement and derive from a taxonomy developed in Western styling
practice. If predicted shape turns out to be strongly associated with race, then
a shape-keyed index is partly an ethnicity-keyed index. That is a direct ethical
problem for the forensic exclusion use case asserted in
`eswa2026deepfaceshape`, and it belongs in the paper whichever way the numbers
fall.

State the association in both directions: how much the shape distribution
differs across groups, and how much a shape label would let you infer the group.

## Part B: per-group stability (RFW, identity-labelled)

RFW (Racial Faces in the Wild) carries roughly 3,000 identities and 10,000
images for each of Caucasian, Indian, Asian and African, about 3.4 images per
person, which is what the pairing protocol needs. It is released for research
use on application at the project page, so **request access now**: the request
is the long pole, not the computation.

With it, repeat measurement 1 exactly as in stage 1, reporting per race group:

| Group | Identities | Pairs | p_top1 | Chance | Kappa (95% CI) |

Use the identical pairing cap, the identical bootstrap over identities, and the
identical router. The comparison of interest is whether kappa differs across
groups by more than the bootstrap intervals allow.

Three outcomes, and all three are reportable:

- **Kappa uniformly low across groups.** The strongest form of the paper's
  claim: instability is a property of the taxonomy, not of one population.
- **Kappa low overall but materially lower for some groups.** The attribute is
  both unusable and unequally unusable, which sharpens the ethical argument
  against the forensic use case.
- **Kappa materially higher in some group.** The claim narrows and must be
  restated as a property of certain populations, and the LFW result is partly a
  population artefact.

## If RFW access is refused or slow

Do not substitute a corpus without identity labels; the measurement is not
defined on one. Alternatives with identities and demographic diversity are
BUPT-Balancedface, and IJB-C under its NIST licence. If none is available in
time, report Part A alone and state plainly in the limitations that per-group
stability remains unmeasured, rather than implying LFW covers it.

## Reporting

Write `results/stage3/VERDICT.md` with the cross-tabulations from Part A, the
per-group table from Part B if available, and a one-paragraph statement of what
the paper can and cannot now claim about demographic generality.
