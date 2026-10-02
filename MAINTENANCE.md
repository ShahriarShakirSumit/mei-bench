# MEI-Bench Maintenance Plan

## Versioning

MEI-Bench follows semantic versioning. **v1.0.0** was the first release (615 items, COCO val2017, single blur kernel `k=51`). **v1.0.1** (the ACML 2026 version) leaves the annotations unchanged and adds the version-pinned re-run predictions, the robustness conditions (further sham draws, saliency-, position- and texture-matched shams, instance-mask evidence) and documentation fixes.

## Hosting and Access

- Public repository: https://github.com/ShahriarShakirSumit/mei-bench (issues, releases and archived tarballs).

## Bug Reports and Issue Triage

- Issues will be tracked in the public repository.
- Critical bugs (annotation errors, intervention pipeline correctness, evaluator bugs) will be addressed in patch releases (v1.0.x) and announced in the repository changelog.
- Per-item annotation corrections will be applied without renumbering the benchmark; superseded items will be flagged in `data/CHANGELOG.md` rather than silently rewritten.

## Planned Updates

- **v1.1**: Human-annotator validation of 15% of items (Fleiss' κ for evidence-region adequacy and question-answer unambiguity).
- **v1.2**: Multi-level dose-response sweeps (`k ∈ {25, 51, 101, 201}`) on a fixed 200-item subset.
- **v2.0**: Domain expansion to a non-COCO image source (candidates: ScienceQA, DocVQA, RemoteCLIP) to test generalization beyond natural images.

## Long-term Reproducibility

- Tagged releases pin model HuggingFace revisions, library versions (`requirements.txt`), and random seeds.
- The release tarball includes:
  - `data/items.jsonl` (manifest of 615 items)
  - `data/metadata/fix_report.json` (region-validity filter audit)
  - `outputs/predictions/*.jsonl` (raw predictions for all evaluated models)
  - `outputs/metrics/*.json` (recomputed metrics)
  - Source code under `mei_benchmark/` and `reproduce/` (see README)

## Contact

Shahriar Shakir Sumit (m.sumit@unsw.edu.au), UNSW Canberra. Please open an issue in the repository for bugs and questions.
