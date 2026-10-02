# Data Provenance

- **Source images.** All MEI-Bench items reference images from the COCO 2017
  validation split (`val2017`), identified by `source_image_id` and the COCO
  filename (`image_path`). No COCO image is redistributed in this package.
- **Questions.** Author-written questions targeted to one of five task types.
- **Ground-truth answers.** Author-verified short-form answers, stored as
  `valid_answers` lists to allow simple synonym matches.
- **Evidence regions.** Bounding boxes derived from COCO instance and panoptic
  annotations and refined by manual review. Areas are normalised to area
  fractions of the image.
- **Sham regions.** Area-matched non-evidence regions selected automatically to
  satisfy the area-match and IoU constraints described in
  `data/metadata/region_validity_schema.json`. Sham boxes are stored on the
  variants that target a sham region.
- **Region-validity filtering.** 959 candidate items were filtered to 615 items
  by enforcing the area-fraction range, the evidence/sham IoU upper bound, and
  the area-match tolerance. Counts: see `data/metadata/fix_report.json`.
- **Intervention metadata.** Each item lists five variants with the
  intervention type, the operation parameters, and (where applicable) the sham
  bounding box. The intervention images themselves are regenerated locally by
  `scripts/generate_interventions.py`.

## Field naming reference

The released annotation file uses the names below. For convenience, the
generic dataset-specification field names map as follows:

| Generic name (E&D spec)           | Released field                                                    |
|-----------------------------------|-------------------------------------------------------------------|
| `coco_image_id`                   | `source_image_id`                                                 |
| `coco_file_name`                  | `image_path`                                                      |
| `ground_truth_answer`             | `valid_answers` (list; first entry is the canonical form)         |
| `evidence_box`                    | `evidence_region.bbox`                                            |
| `sham_box`                        | `variants[*].sham_region_bbox` (per applicable intervention)      |
| `evidence_area_fraction`          | `evidence_region.area_fraction`                                   |
| `sham_area_fraction`              | computed from `sham_region_bbox` and image size                   |
| `evidence_sham_iou`               | computed from `evidence_region.bbox` and `sham_region_bbox`       |
| `validity_checks`                 | `human_verified` (boolean; `false` for all 615 items)             |
| `intervention_types_available`    | `[v.intervention_type for v in variants]`                         |
| `intervention_parameters`         | `variants[*].intervention_params`                                 |
| `difficulty`                      | `difficulty` (`easy` / `medium` / `hard`)                         |
| `split`                           | not used (single split; all items are the diagnostic set)         |
| `source_metadata`                 | `source_dataset` + `source_image_id` + `image_path`               |
| `evaluation_metadata`             | computed by `mei_benchmark/evaluation/metrics.py`                 |
