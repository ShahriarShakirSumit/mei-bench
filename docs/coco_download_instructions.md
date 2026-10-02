# COCO val2017 Download Instructions

MEI-Bench references but does not redistribute COCO val2017. Users should
download COCO val2017 directly from the official source.

## Steps

```
mkdir -p external/coco
cd external/coco

# Images
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

## Expected layout
```
external/coco/val2017/000000000139.jpg
external/coco/val2017/...               (5000 images)
external/coco/annotations/instances_val2017.json
```

## Linkage
Each MEI-Bench annotation references a COCO image by its `image_path`
(e.g. `000000006954.jpg`) and `source_image_id` (the integer COCO image id as a
string). To resolve an item to a local file:

```
local_path = "external/coco/val2017/" + item["image_path"]
```

## License
- COCO annotations: CC BY 4.0.
- COCO images: governed by the Flickr Terms of Use of the original uploaders.
