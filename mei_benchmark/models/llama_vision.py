"""
Meta Llama 3.2 Vision model wrapper for MEI benchmark.

Supports Llama-3.2-11B-Vision-Instruct via MllamaForConditionalGeneration.
Requires transformers >= 4.45.
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


class LlamaVisionWrapper(BaseVLM):
    """Wrapper for Meta Llama 3.2 Vision models.

    Supports:
        - meta-llama/Llama-3.2-11B-Vision-Instruct
        - unsloth/Llama-3.2-11B-Vision-Instruct (ungated mirror)

    Uses MllamaForConditionalGeneration from transformers >= 4.45.

    Usage:
        config = VLMConfig(
            model_name="Llama-3.2-11B-Vision",
            model_id="unsloth/Llama-3.2-11B-Vision-Instruct",
        )
        model = LlamaVisionWrapper(config)
        model.load_model()
        response = model.predict(image, "What is in the image?")
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Load Llama 3.2 Vision model and processor."""
        from transformers import AutoProcessor, MllamaForConditionalGeneration

        model_id = self.config.model_id
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        logger.info(f"Loading Llama 3.2 Vision: {model_id}")

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

        # Llama 3.2 Vision does NOT support flash attention 2 natively
        # Use sdpa which is the default and works well

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id, **model_kwargs
        )
        self.model.eval()
        self._loaded = True
        logger.info(f"Llama 3.2 Vision loaded: {model_id}")

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

        # Llama 3.2 Vision uses chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        start_time = time.perf_counter()

        inputs = self.processor(
            images=pil_image,
            text=input_text,
            return_tensors="pt",
        ).to(self.model.device)
        # Cast floating-point tensors to model dtype (avoid casting int tensors)
        model_dtype = self.model.dtype
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                inputs[k] = v.to(dtype=model_dtype)

        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
        }
        if self.config.temperature > 0:
            gen_kwargs["temperature"] = self.config.temperature

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only generated tokens
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
