# Intervention Generation

Each MEI-Bench item has five intervened images. `mei_benchmark/scripts/render_variants.py` renders them from the
boxes stored in `data/annotations/mei_bench.jsonl`; the operators are in `mei_benchmark/intervention/generator.py`.
Boxes are `[x, y, w, h]` in pixels and are turned into rectangular masks by `bbox_to_mask`
(`mei_benchmark/scripts/generate_interventions.py`). Images are written as JPEG, quality 95.

| variant | operation |
|---|---|
| `evidence_blur` | OpenCV Gaussian blur, kernel 51 x 51, sigma chosen by OpenCV (`sigma = 0`), inside the evidence box |
| `sham_blur` | the same blur inside the sham box (`sham_region_bbox` of the variant) |
| `evidence_gray` | evidence box filled with the uniform value 128 in every channel |
| `sham_gray` | the same fill inside the sham box |
| `evidence_swap` | evidence box colour-inverted, then its 4 x 4 blocks shuffled at random |

Blur and gray-out are deterministic: rendering `data/rerun/items.jsonl` reproduces the re-run's images pixel for pixel.
The swap shuffle uses NumPy's global random generator. The paper's swap images were made without a fixed seed;
`render_variants.py` seeds it per item, so its swap images are reproducible but not identical to the paper's.

Sham boxes are sampled by `generate_sham_region` (`generate_interventions.py`) under the paper's validity
constraints: IoU with the evidence box below 0.1 and area ratio in [0.8, 1.2]. `generate_interventions.py`
samples new sham boxes; `render_variants.py` uses the stored ones.

## Output layout

```
data/mei_bench/items.jsonl
data/mei_bench/000000006954.jpg                      -> link to the COCO image
data/mei_bench/variants/mei_000424_evidence_blur.jpg
data/mei_bench/variants/mei_000424_sham_blur.jpg
...
```
