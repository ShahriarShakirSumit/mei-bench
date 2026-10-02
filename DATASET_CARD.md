# MEI-Bench Dataset Card

## 1. Dataset name
**MEI-Bench** (Minimal Evidence Intervention Benchmark)

## 2. Intended use
Diagnostic evaluation of *causal visual grounding* in vision-language large models
(VLLMs). MEI-Bench tests whether a model's answer to a visual question changes when
the minimal visual evidence is degraded, while remaining stable under a matched sham
degradation. The benchmark is intended for diagnostic and analysis purposes; it is
not intended as a leaderboard.

## 3. Source data
COCO val2017 images, referenced by `source_image_id` (numeric) and `image_path`
(COCO filename). The COCO images themselves are **not** redistributed in this
package. See `docs/coco_download_instructions.md`.

## 4. Data type
- Question/answer annotations
- Minimal evidence bounding boxes
- Sham (control) bounding boxes
- Intervention metadata (type, parameters, variant filename)
- Region-validity metadata (area fractions, IoU)
- Evaluation scripts and validators
- JSON schemas

## 5. Benchmark size
**615 items** after region-validity filtering. The filtering pipeline started from
959 candidate items and removed items that violated the area-fraction range, the
evidence/sham IoU constraint, or the area-match constraint. See
`data/metadata/fix_report.json` for the full breakdown.

## 6. Task types
- `object_identification`
- `attribute_verification`
- `spatial_relations`
- `counting`
- `text_in_image`

## 7. Intervention types
- `evidence_blur`  -- Gaussian blur applied inside the evidence box
- `sham_blur`      -- Gaussian blur applied inside an area-matched non-evidence box
- `evidence_gray`  -- Gray-out: replace the evidence box with a uniform gray fill (intensity 128)
- `sham_gray`      -- Gray-out of the sham box (uniform gray fill, intensity 128)
- `evidence_swap`  -- Replace evidence box content with content from another image

## 7.1 Metrics
- **Causal Sensitivity (CS)** -- fraction of items whose answer changes under any evidence intervention.
- **Sham Sensitivity (SI)** -- fraction of items whose answer changes under any sham intervention.
- **Grounding Gap (GG = CS - SI)** -- evidence-specific sensitivity above sham baseline.
- **Sham Robustness (SR = 1 - SI)** -- stability under non-evidence perturbations.
- **Grounding Specificity (GS = max(GG, 0) / max(CS, eps))** -- proportion of evidence sensitivity attributable to grounding rather than generic fragility.
- **Dose-response** -- the toolkit can sweep the blur kernel (`run_dose_response_inference`); the released results use a single kernel (k = 51).

## 8. Known limitations
- COCO val2017 covers only the COCO label distribution (80 object categories, common
  scenes). Diagnostic coverage is restricted to this distribution.
- Evidence regions are the bounding box of one annotated COCO instance (the queried object; the
  first-named object for spatial relations; one instance for counting). Items were not individually
  human-verified (`human_verified = false`). Cues outside the box may also be informative.
- The 86 text-in-image items were released without verified OCR answers (placeholder answer field),
  so no model answers them correctly and they do not contribute to grounding metrics.
- Intervention parameters (blur radius, gray fill value) are fixed per condition and
  do not exhaustively span the dose-response curve.
- All annotations are in English.

## 9. Ethical considerations
COCO images may contain depictions of people. We do not collect new images of
people, and we do not annotate any personal attributes. The benchmark does not
include identity recognition tasks. Scores on MEI-Bench should not be interpreted
as a safety certification for clinical, autonomous-driving, or surveillance
deployments.

## 10. Personal data statement
This package does not newly collect any personal data. Some COCO images contain
people, but no identity information is added by MEI-Bench.

## 11. Human subjects statement
No human-subject data is collected. Annotation review involved the authors only and
did not involve crowdworkers or external participants.

## 12. Maintenance and versioning
- Version: `1.0.1` (the ACML 2026 version; annotations unchanged from 1.0.0)
- Versioning policy: semantic versioning. Schema-breaking changes increment the major
  version; new fields or items increment the minor version; metadata fixes increment
  the patch version.
- Issues and updates: tracked at https://github.com/ShahriarShakirSumit/mei-bench.
