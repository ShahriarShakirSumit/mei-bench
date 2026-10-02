# Responsible AI Notes

## Source data
MEI-Bench references images from the COCO 2017 validation split via image IDs
and filenames only.

## Raw COCO images
Not redistributed. Users download them from the official COCO source.

## Personal data
This package does not newly collect personal data. COCO images may contain
people, but MEI-Bench does not annotate identity, demographics, or any other
personal attribute.

## Human subjects
No new human-subject data collection is performed by this package. Annotation
review involved the project authors only.

## Limitations
- Evidence and sham regions depend on COCO annotations and on the
  region-validity filtering criteria. Items where no valid sham region could be
  found were dropped (see `data/metadata/fix_report.json`).
- Coverage is bounded by the COCO val2017 distribution (80 object categories,
  predominantly common-scene Western imagery).
- Intervention parameters are fixed per condition (single blur kernel k = 51); the toolkit
  can sweep the kernel, but no dose-response results are part of this release.

## Biases
COCO is known to over-represent certain object categories, scene types, and
imagery sources relative to a uniform world distribution. Diagnostic results
on MEI-Bench inherit this distribution.

## Intended use
- Causal visual grounding diagnostics.
- Evidence-dependence and sham-control analysis of VLLMs.
- Robustness studies that separate evidence sensitivity from generic
  perturbation fragility.

## Misuse warning
Scores on MEI-Bench are diagnostic and **must not** be interpreted as a safety
certification. They do not establish readiness for clinical, autonomous-driving,
surveillance, or any other safety-critical deployment.
