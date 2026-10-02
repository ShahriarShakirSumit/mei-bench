# Annotation Guidelines

## Question authoring
Each question is short-form, English, and targets exactly one of:
- `object_identification`
- `attribute_verification`
- `spatial_relations`
- `counting`
- `text_in_image`

A valid question must:
1. Have a determinate, observer-independent answer given the image.
2. Be answerable from the chosen evidence region alone, in the typical case.
3. Avoid commonsense-only reasoning that does not require visual evidence.

## Ground-truth answers
- Stored as a list of acceptable strings (case-insensitive, leading/trailing
  whitespace stripped).
- For counting tasks the canonical answer is the integer count rendered as a
  decimal string (`"3"`).

## Evidence-region selection
The evidence region is the smallest rectangular crop that suffices for a human
to answer the question. Authors selected evidence boxes from COCO bounding
boxes when applicable and tightened them for spatial-relation and counting
questions where multiple objects are involved.

## Sham-region selection
The sham region is an area-matched rectangle that does not overlap the evidence
region. The selection algorithm enforces:
- Sham/evidence IoU `<= 0.10` (strict).
- `|sham_area - evidence_area| / evidence_area <= 0.20` (strict).
- Area-fraction is reported per item; the released items span roughly `[0.005, 1.0]`.

## Human verification
Each item was reviewed at least once by a project author. Items that failed any
of the validity constraints, or for which the question could be answered
without inspecting the image, were dropped. The full filter counts are in
`data/metadata/fix_report.json`.
