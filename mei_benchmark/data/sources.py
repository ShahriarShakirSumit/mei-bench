"""
Source dataset adapters for MEI benchmark construction.

Provides unified access to COCO, OpenImages, and other source datasets
for selecting images and generating MEI benchmark items.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class SourceImage:
    """A source image with its metadata from the original dataset."""

    image_id: str
    image_path: str
    width: int
    height: int
    annotations: list[dict]  # Original annotations from source dataset
    caption: Optional[str] = None


@dataclass
class SourceAnnotation:
    """A normalized annotation from any source dataset."""

    category: str
    category_id: int
    bbox: list[float]  # [x, y, w, h] in pixel coords
    segmentation: Optional[list] = None  # Polygon or RLE
    area: float = 0.0
    iscrowd: bool = False


class SourceDatasetAdapter(ABC):
    """Abstract adapter for source datasets."""

    @abstractmethod
    def load(self) -> None:
        """Load the dataset metadata."""
        ...

    @abstractmethod
    def get_image(self, image_id: str) -> SourceImage:
        """Get a single image by ID."""
        ...

    @abstractmethod
    def list_images(self) -> list[str]:
        """List all available image IDs."""
        ...

    @abstractmethod
    def get_annotations(self, image_id: str) -> list[SourceAnnotation]:
        """Get annotations for an image."""
        ...

    @abstractmethod
    def get_categories(self) -> dict[int, str]:
        """Get category ID to name mapping."""
        ...


class COCOAdapter(SourceDatasetAdapter):
    """Adapter for COCO dataset (val2017).

    Works with both standard filesystem paths and squashfs-mounted COCO data.
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_file: str | Path,
    ):
        self.images_dir = Path(images_dir)
        self.annotations_file = Path(annotations_file)
        self._data: Optional[dict] = None
        self._image_index: dict[int, dict] = {}
        self._ann_index: dict[int, list[dict]] = {}
        self._categories: dict[int, str] = {}
        self._images_list: Optional[list[SourceImage]] = None

    @property
    def images(self) -> list[SourceImage]:
        """All images as SourceImage objects (cached)."""
        if self._data is None:
            self.load()
        if self._images_list is None:
            self._images_list = []
            for img_id in sorted(self._image_index.keys()):
                info = self._image_index[img_id]
                self._images_list.append(SourceImage(
                    image_id=str(img_id),
                    image_path=info["file_name"],
                    width=info["width"],
                    height=info["height"],
                    annotations=self._ann_index.get(img_id, []),
                ))
        return self._images_list

    @property
    def cat_id_to_name(self) -> dict[int, str]:
        """Category ID to name mapping."""
        if self._data is None:
            self.load()
        return dict(self._categories)

    @property
    def cat_name_to_id(self) -> dict[str, int]:
        """Category name to ID mapping."""
        if self._data is None:
            self.load()
        return {name: cid for cid, name in self._categories.items()}

    def __len__(self) -> int:
        if self._data is None:
            self.load()
        return len(self._image_index)

    def load(self) -> None:
        """Load COCO annotations JSON."""
        with open(self.annotations_file) as f:
            self._data = json.load(f)

        # Build image index
        for img in self._data.get("images", []):
            self._image_index[img["id"]] = img

        # Build annotation index (image_id -> annotations)
        for ann in self._data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self._ann_index:
                self._ann_index[img_id] = []
            self._ann_index[img_id].append(ann)

        # Build category index
        for cat in self._data.get("categories", []):
            self._categories[cat["id"]] = cat["name"]

    def get_image(self, image_id: str) -> SourceImage:
        """Get a COCO image by ID."""
        if self._data is None:
            self.load()

        img_id = int(image_id)
        img_info = self._image_index.get(img_id)
        if img_info is None:
            raise KeyError(f"Image {image_id} not found in COCO dataset")

        image_path = str(self.images_dir / img_info["file_name"])
        annotations = self._ann_index.get(img_id, [])

        return SourceImage(
            image_id=str(img_id),
            image_path=image_path,
            width=img_info["width"],
            height=img_info["height"],
            annotations=annotations,
        )

    def list_images(self) -> list[str]:
        """List all COCO image IDs."""
        if self._data is None:
            self.load()
        return [str(img_id) for img_id in sorted(self._image_index.keys())]

    def get_annotations(self, image_id: str | int) -> list[SourceAnnotation]:
        """Get normalized annotations for a COCO image."""
        if self._data is None:
            self.load()

        img_id = int(image_id)
        raw_anns = self._ann_index.get(img_id, [])

        annotations = []
        for ann in raw_anns:
            cat_name = self._categories.get(ann["category_id"], "unknown")
            annotations.append(
                SourceAnnotation(
                    category=cat_name,
                    category_id=ann["category_id"],
                    bbox=ann["bbox"],  # COCO format: [x, y, w, h]
                    segmentation=ann.get("segmentation"),
                    area=ann.get("area", 0.0),
                    iscrowd=bool(ann.get("iscrowd", 0)),
                )
            )
        return annotations

    def get_categories(self) -> dict[int, str]:
        """Get COCO category mapping."""
        if self._data is None:
            self.load()
        return dict(self._categories)

    def find_images_with_multiple_objects(
        self, category: Optional[str] = None, min_count: int = 2, max_count: int = 10
    ) -> list[tuple]:
        """Find images containing multiple instances of a category.

        If category is None, searches for any category with multiple instances.

        Returns:
            List of (SourceImage, count, category_id) tuples
        """
        if self._data is None:
            self.load()

        results = []

        if category is not None:
            # Find category ID
            cat_id = None
            for cid, name in self._categories.items():
                if name == category:
                    cat_id = cid
                    break
            if cat_id is None:
                raise ValueError(f"Category '{category}' not found")

            for img_id, anns in self._ann_index.items():
                count = sum(1 for a in anns if a["category_id"] == cat_id and not a.get("iscrowd"))
                if min_count <= count <= max_count:
                    info = self._image_index[img_id]
                    src = SourceImage(
                        image_id=str(img_id),
                        image_path=info["file_name"],
                        width=info["width"],
                        height=info["height"],
                        annotations=anns,
                    )
                    results.append((src, count, cat_id))
        else:
            # Find any category with multiple instances per image
            for img_id, anns in self._ann_index.items():
                cat_counts: dict[int, int] = {}
                for a in anns:
                    if not a.get("iscrowd"):
                        cid = a["category_id"]
                        cat_counts[cid] = cat_counts.get(cid, 0) + 1
                for cid, count in cat_counts.items():
                    if min_count <= count <= max_count:
                        info = self._image_index[img_id]
                        src = SourceImage(
                            image_id=str(img_id),
                            image_path=info["file_name"],
                            width=info["width"],
                            height=info["height"],
                            annotations=anns,
                        )
                        results.append((src, count, cid))
                        break  # One per image

        return results

    def find_images_with_small_objects(
        self, area_threshold: float = 1024.0
    ) -> list[str]:
        """Find images containing small objects (area < threshold).

        Useful for generating resolution ambiguity scenarios.
        """
        if self._data is None:
            self.load()

        result = set()
        for img_id, anns in self._ann_index.items():
            for ann in anns:
                if ann.get("area", 0) < area_threshold and not ann.get("iscrowd"):
                    result.add(str(img_id))
                    break

        return sorted(result)

    def find_images_with_occlusions(
        self, iou_threshold: float = 0.3
    ) -> list[str]:
        """Find images with overlapping objects (potential occlusions).

        Detects images where object bounding boxes overlap significantly.
        """
        if self._data is None:
            self.load()

        result = []
        for img_id, anns in self._ann_index.items():
            if len(anns) < 2:
                continue

            boxes = []
            for ann in anns:
                if ann.get("iscrowd"):
                    continue
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])

            if len(boxes) < 2:
                continue

            # Check pairwise IoU
            has_overlap = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    iou = _compute_iou(boxes[i], boxes[j])
                    if iou > iou_threshold:
                        has_overlap = True
                        break
                if has_overlap:
                    break

            if has_overlap:
                result.append(str(img_id))

        return result


def _compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / max(union, 1e-8)
