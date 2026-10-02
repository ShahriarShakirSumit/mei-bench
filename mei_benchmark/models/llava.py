"""
LLaVA family model wrapper for MEI benchmark.

Supports LLaVA-1.5, LLaVA-1.6/NeXT, and LLaVA-OneVision variants.
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


class LLaVAWrapper(BaseVLM):
    """Wrapper for LLaVA models using HuggingFace transformers.

    Supports:
        - llava-hf/llava-1.5-7b-hf
        - llava-hf/llava-1.5-13b-hf
        - llava-hf/llava-v1.6-mistral-7b-hf
        - llava-hf/llava-onevision-qwen2-7b-ov-hf

    Usage:
        config = VLMConfig(
            model_name="LLaVA-1.5-7B",
            model_id="llava-hf/llava-1.5-7b-hf",
        )
        model = LLaVAWrapper(config)
        model.load_model()
        response = model.predict(image, "What is in the image?")
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Load LLaVA model and processor from HuggingFace."""
        from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration

        model_id = self.config.model_id
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.float16)

        logger.info(f"Loading LLaVA model: {model_id}")

        # Determine model class
        if "v1.6" in model_id or "next" in model_id.lower():
            model_cls = LlavaNextForConditionalGeneration
        else:
            model_cls = LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
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
                    "Falling back to eager attention."
                )
                model_kwargs["attn_implementation"] = "eager"

        self.model = model_cls.from_pretrained(model_id, **model_kwargs)
        self.model.eval()
        self._loaded = True
        logger.info(f"LLaVA model loaded: {model_id}")

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

        prompt = self._build_conversation(question, choices)

        start_time = time.perf_counter()

        inputs = self.processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
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

    def _build_conversation(
        self, question: str, choices: Optional[list[str]] = None
    ) -> str:
        """Build the conversation prompt for LLaVA.

        Uses the chat template expected by the processor.
        """
        prompt_text = self.format_prompt(question, choices)

        # LLaVA-1.5 format
        if "1.5" in self.config.model_id:
            return f"USER: <image>\n{prompt_text}\nASSISTANT:"

        # LLaVA-1.6 / NeXT format
        if "v1.6" in self.config.model_id or "next" in self.config.model_id.lower():
            return f"[INST] <image>\n{prompt_text} [/INST]"

        # OneVision / default - use chat template
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

    def predict_batch(
        self,
        images: list[np.ndarray],
        questions: list[str],
        choices_list: Optional[list[list[str]]] = None,
    ) -> list[VLMResponse]:
        """Batch prediction with proper padding."""
        assert self._loaded, "Model not loaded. Call load_model() first."

        results = []
        batch_size = self.config.batch_size

        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch_images = images[start:end]
            batch_questions = questions[start:end]
            batch_choices = choices_list[start:end] if choices_list else [None] * (end - start)

            # Process batch items individually (LLaVA batched inference is tricky)
            for img, q, ch in zip(batch_images, batch_questions, batch_choices):
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
