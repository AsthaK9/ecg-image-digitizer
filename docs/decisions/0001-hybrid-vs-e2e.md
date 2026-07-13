# ADR 0001: Hybrid Pipeline vs. End-to-End Regression

**Status:** Accepted

## Context

ECG image digitization can be approached as (a) a single end-to-end model
mapping image -> signal directly, or (b) a decomposed pipeline of
preprocessing, learned segmentation, and deterministic signal reconstruction.

## Decision

Use a decomposed hybrid pipeline: deskew/calibration -> segmentation model ->
deterministic column-scan reconstruction.

## Rationale

- Grid calibration (mm-per-pixel, paper speed, gain) is exact, known physics
  printed on the paper itself — not a relationship worth spending model
  capacity re-learning.
- Each stage is independently testable; calibration and reconstruction logic
  can be unit tested without a GPU or trained weights.
- Failure modes are interpretable: a bad prediction can be traced to a
  specific stage (bad calibration vs. bad segmentation vs. bad gap-filling)
  rather than being an opaque end-to-end failure.
- The SNR evaluation metric is sensitive to fine-grained temporal alignment;
  deterministic reconstruction gives more direct control over this than an
  implicit learned decoder would.

## Consequences

- More moving parts to build and integrate than a single model.
- Segmentation model quality becomes the primary bottleneck for overall
  system accuracy; reconstruction logic amplifies segmentation errors if the
  mask is poor (particularly under heavy occlusion).
- An end-to-end regression model remains a valid challenger approach and can
  be added later as an alternative pipeline stage without restructuring the
  rest of the codebase (see `pipeline/stages.py`).
