"""
Microsoft Phi-3.5 Vision model wrapper for MEI benchmark.

Supports Phi-3.5-vision-instruct (4.2B) via trust_remote_code.
Uses flash_attn when available.
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


class Phi3VisionWrapper(BaseVLM):
    """Wrapper for Microsoft Phi-3.5-vision-instruct.

    Uses AutoModelForCausalLM with trust_remote_code=True as the
    model has its own modeling code on HuggingFace Hub.

    Usage:
        config = VLMConfig(
            model_name="Phi-3.5-Vision",
            model_id="microsoft/Phi-3.5-vision-instruct",
        )
        model = Phi3VisionWrapper(config)
        model.load_model()
        response = model.predict(image, "What is in the image?")
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Load Phi-3.5 Vision model and processor."""
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        model_id = self.config.model_id
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.float16)

        logger.info(f"Loading Phi-3.5 Vision: {model_id}")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )

        # Load config first and override _attn_implementation
        # Phi-3.5 config.json has _attn_implementation="flash_attention_2" baked in;
        # we must override it in the config object before loading the model.
        model_config = AutoConfig.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )

        if self.config.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                model_config._attn_implementation = "flash_attention_2"
                dtype = torch.bfloat16  # Phi-3.5 needs bf16 for flash attention
                logger.info("Using Flash Attention 2 (bfloat16)")
            except ImportError:
                model_config._attn_implementation = "eager"
                logger.warning("flash_attn not available, using eager attention")
        else:
            model_config._attn_implementation = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=model_config,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self._loaded = True
        logger.info(f"Phi-3.5 Vision loaded: {model_id}")

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

        # Phi-3.5 Vision uses chat template with <|image_1|> placeholder
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{prompt_text}",
            }
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start_time = time.perf_counter()

        inputs = self.processor(
            prompt,
            [pil_image],
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
        }
        if self.config.temperature > 0:
            gen_kwargs["temperature"] = self.config.temperature

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                **gen_kwargs,
            )

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
