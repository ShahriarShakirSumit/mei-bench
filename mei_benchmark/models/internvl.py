"""
InternVL2 model wrapper for MEI benchmark.

Supports InternVL2 variants (2B, 8B, 26B, 40B, 76B).
Uses HuggingFace transformers backend.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from mei_benchmark.models.base import BaseVLM, VLMConfig, VLMResponse

logger = logging.getLogger(__name__)

# ---- InternVL2 image preprocessing helpers ----
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int) -> T.Compose:
    """Build the standard InternVL2 image transform."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list,
    width: int,
    height: int,
    image_size: int,
) -> tuple:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_ar)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = False,
) -> list[Image.Image]:
    """Split image into dynamic patches following InternVL2 protocol."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = best_ratio[0] * image_size
    target_height = best_ratio[1] * image_size
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _preprocess_image(
    pil_image: Image.Image,
    input_size: int = 448,
    max_num: int = 12,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Convert a PIL image to InternVL2's pixel_values tensor."""
    transform = _build_transform(input_size)
    images = _dynamic_preprocess(
        pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values.to(dtype).cuda()


class InternVL2Wrapper(BaseVLM):
    """Wrapper for InternVL2 models.

    Supports:
        - OpenGVLab/InternVL2-8B
        - OpenGVLab/InternVL2-26B
        - OpenGVLab/InternVL2_5-8B

    Usage:
        config = VLMConfig(
            model_name="InternVL2-8B",
            model_id="OpenGVLab/InternVL2-8B",
        )
        model = InternVL2Wrapper(config)
        model.load_model()
        response = model.predict(image, "What is in this image?")
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.processor = None

    def load_model(self) -> None:
        """Load InternVL2 model."""
        from transformers import AutoModel, AutoTokenizer
        import transformers.modeling_utils as _mu

        # Monkey-patch for transformers ≥5.0 compatibility:
        # InternVL2's custom model class may lack `all_tied_weights_keys`
        # which causes `mark_tied_weights_as_initialized` to crash.
        if hasattr(_mu.PreTrainedModel, "mark_tied_weights_as_initialized"):
            _orig_mark_tied = _mu.PreTrainedModel.mark_tied_weights_as_initialized

            def _safe_mark_tied(self_model, loading_info):
                if not hasattr(self_model, "all_tied_weights_keys"):
                    self_model.all_tied_weights_keys = {}
                return _orig_mark_tied(self_model, loading_info)

            _mu.PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied

        model_id = self.config.model_id
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        logger.info(f"Loading InternVL2 model: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )

        model_kwargs = {
            "torch_dtype": dtype,
            "cache_dir": self.config.cache_dir,
            # NOTE: InternVL2's custom modeling code calls
            # torch.linspace(...).item() in __init__, which is incompatible
            # with meta tensors created by device_map="auto" / low_cpu_mem_usage.
            # Load directly to CUDA instead.
            "trust_remote_code": True,
        }

        self.model = AutoModel.from_pretrained(model_id, **model_kwargs).cuda().eval()

        # Monkey-patch: transformers >=4.50 removed GenerationMixin from
        # PreTrainedModel, so InternLM2ForCausalLM lost .generate() and
        # .generation_config. Re-add them by patching the class hierarchy
        # and attaching a default GenerationConfig.
        from transformers.generation import GenerationMixin, GenerationConfig

        lang_model = self.model.language_model
        lang_cls = type(lang_model)
        if not issubclass(lang_cls, GenerationMixin):
            new_cls = type(lang_cls.__name__, (lang_cls, GenerationMixin), {})
            self.model.language_model.__class__ = new_cls
            logger.info("Patched language_model with GenerationMixin")
        if not hasattr(self.model.language_model, "generation_config"):
            self.model.language_model.generation_config = GenerationConfig()
            logger.info("Patched language_model with default GenerationConfig")

        self._loaded = True
        logger.info(f"InternVL2 model loaded: {model_id}")

    def predict(
        self,
        image: np.ndarray,
        question: str,
        choices: Optional[list[str]] = None,
    ) -> VLMResponse:
        """Generate a response for a single image-question pair."""
        assert self._loaded, "Model not loaded. Call load_model() first."

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        prompt_text = self.format_prompt(question, choices)

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.float16)

        # Preprocess image into pixel_values tensor
        pixel_values = _preprocess_image(pil_image, input_size=448, max_num=12, dtype=dtype)

        start_time = time.perf_counter()

        # InternVL2 uses its own chat method
        generation_config = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
        )
        if self.config.temperature > 0:
            generation_config["temperature"] = self.config.temperature

        raw_output = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt_text,
            generation_config,
        )

        latency = (time.perf_counter() - start_time) * 1000
        answer = self.extract_answer(raw_output, choices)

        return VLMResponse(
            answer=answer,
            raw_output=raw_output,
            latency_ms=latency,
        )

    def unload_model(self) -> None:
        """Release model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache()
        super().unload_model()
