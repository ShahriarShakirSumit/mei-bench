# MEI-Bench Datasheet

Following Gebru et al. (2021), this datasheet documents MEI-Bench v1.

## Motivation

- **Why was the dataset created?** To enable controlled evaluation of visual grounding in vision-language models through evidence/sham interventions.
- **Who created it?** The authors at UNSW Canberra, Torrens University Australia and Obuda University. Computation was performed on the National Computational Infrastructure (NCI) through the UNSW HPC Scheme.
- **Any other comments?** MEI-Bench is a *derivative* dataset: it adds annotations on top of COCO val2017 images and does not redistribute the underlying images.

## Composition

- **Instances:** 615 visual question answering items, each with one COCO image, one question, ground-truth answer(s), an evidence region (bounding box), an area-matched sham region with `IoU(S, E) < 0.1`, and five intervention variants (evidence-blur, evidence-gray, evidence-swap, sham-blur, sham-gray).
- **Total:** 615 items × (5 intervention variants + the original image) = 3,690 model queries per evaluated model.
- **Task types:** object identification (114), attribute verification (118), spatial relations (153), counting (144), text-in-image (86).
- **Pool:** Drawn from 532 unique COCO val2017 images (multiple items can share an image but never share an evidence region).
- **Splits:** No train/val/test split. MEI-Bench is an evaluation-only benchmark; models must be trained on external data.
- **Recommended uses:** Diagnostic evaluation of visual grounding for VLLMs with 7B to 13B parameters. Not intended as a training set.

## Collection process

- **Source data:** COCO val2017 images, captions, and bounding-box annotations (Lin et al., 2014).
- **Question generation:** Programmatic. Questions are templated from COCO category labels and bounding-box geometry. No human writers were involved.
- **Evidence regions:** COCO bounding boxes for the question's referenced object.
- **Sham regions:** Sampled from the same image, area-matched to the evidence region, with the disjointness and IoU constraints in Definition 2 of the paper.
- **Filtering:** Items violating Definition 2 were removed (109 IoU violations, 235 area-fraction violations, 0 area-match violations from a candidate pool of 959).

## Preprocessing / cleaning

- Images are used at original COCO resolution.
- Intervention images are saved as JPEG quality 95.
- Color labels for attribute-verification items inherit a known ~20% noise rate from automated color extraction; this affects accuracy on that task type but not the grounding metrics, which condition on correct answers.

## Uses

- **Recommended:** Diagnostic grounding evaluation, comparison of training recipes at fixed parameter scale, ablation of vision encoders.
- **Not recommended:** Training data; medical/legal/safety-critical decisions without domain-specific re-validation; any use that assumes generalization beyond natural-image VQA.
- **Known biases:** COCO inherits geographic, demographic, and cultural biases from its collection process. MEI-Bench inherits all of them and adds no further diversity controls.

## Distribution

- **License:** Annotations and metadata under CC-BY-4.0; code under MIT. COCO images themselves are not redistributed.
- **Where:** https://github.com/ShahriarShakirSumit/mei-bench
- **Intended audience:** ML researchers studying VLLM evaluation, grounding, and causal probing.

## Maintenance

See `MAINTENANCE.md` for the maintenance policy, planned releases, and contact information.
