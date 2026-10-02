"""
Qwen-VL family model wrapper for MEI benchmark.

Supports Qwen-VL-Chat and Qwen2-VL variants.
Uses HuggingFace transformers backend.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from mei_benchmark.models.base import BaseVLM, VLMConfig, VLMResponse

logger = logging.getLogger(__name__)


class QwenVLWrapper(BaseVLM):
    """Wrapper for Qwen-VL models.

    Supports:
        - Qwen/Qwen2-VL-7B-Instruct
        - Qwen/Qwen2-VL-72B-Instruct
        - Qwen/Qwen2.5-VL-7B-Instruct

    Usage:
        config = VLMConfig(
            model_name="Qwen2-VL-7B",
            model_id="Qwen/Qwen2-VL-7B-Instruct",
        )
        model = QwenVLWrapper(config)
        model.load_model()
        response = model.predict(image, "What is in the image?")
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Load Qwen-VL model and processor from HuggingFace."""
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        model_id = self.config.model_id
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        logger.info(f"Loading Qwen-VL model: {model_id}")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )

        model_kwargs = {
            "torch_dtype": dtype,
            "device_map": "auto",
            "cache_dir": self.config.cache_dir,
            "low_cpu_mem_usage": True,
        }
        if self.config.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                logger.warning(
                    "flash_attn not available (requires Ampere+ GPU). "
                    "Falling back to sdpa attention."
                )
                model_kwargs["attn_implementation"] = "sdpa"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, **model_kwargs
        )
        self.model.eval()
        self._loaded = True
        logger.info(f"Qwen-VL model loaded: {model_id}")

    def predict(
        self,
        image: np.ndarray,
        question: str,
        choices: Optional[list[str]] = None,
    ) -> VLMResponse:
        """Generate a response for a single image-question pair."""
        assert self._loaded, "Model not loaded. Call load_model() first."

        # Convert BGR numpy to RGB PIL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        prompt_text = self.format_prompt(question, choices)

        # Build Qwen2-VL conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        start_time = time.perf_counter()

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[-1]
        raw_output = self.processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        )

        latency = (time.perf_counter() - start_time) * 1000
        answer = self.extract_answer(raw_output, choices)

        return VLMResponse(
            answer=answer,
            raw_output=raw_output,
            latency_ms=latency,
            tokens_used=int(output_ids.shape[-1]),
        )

    def predict_batch(
        self,
        images: list[np.ndarray],
        questions: list[str],
        choices_list: Optional[list[list[str]]] = None,
    ) -> list[VLMResponse]:
        """Batch prediction (sequential for Qwen-VL)."""
        results = []
        for i, (img, q) in enumerate(zip(images, questions)):
            ch = choices_list[i] if choices_list else None
            results.append(self.predict(img, q, ch))
        return results

    def unload_model(self) -> None:
        """Release model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        super().unload_model()
