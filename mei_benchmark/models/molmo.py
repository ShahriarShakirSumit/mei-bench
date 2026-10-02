"""
Molmo model wrapper for MEI benchmark.

Supports Molmo-7B-D-0924 from Allen AI.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from mei_benchmark.models.base import BaseVLM, VLMConfig, VLMResponse

logger = logging.getLogger(__name__)


class MolmoWrapper(BaseVLM):
    """Wrapper for Allen AI Molmo vision-language models.

    Uses AutoModelForCausalLM + AutoProcessor with trust_remote_code=True.

    Supports:
        - allenai/Molmo-7B-D-0924
        - allenai/Molmo-7B-O-0924

    Usage:
        config = VLMConfig(
            model_name="Molmo-7B-D",
            model_id="allenai/Molmo-7B-D-0924",
        )
        model = MolmoWrapper(config)
        model.load_model()
        response = model.predict(image, "What is in the image?")
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Load Molmo model and processor."""
        model_id = self.config.model_id
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        logger.info(f"Loading Molmo: {model_id}")

        # Molmo uses trust_remote_code for its custom architecture
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )

        model_kwargs = {
            "torch_dtype": dtype,
            "cache_dir": self.config.cache_dir,
            "trust_remote_code": True,
            "device_map": self.config.device,
        }

        # Molmo supports BF16 well on Ampere+; use eager on V100 (sm_70)
        # Always set attn_implementation explicitly to avoid flash_attn import errors
        if self.config.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                model_kwargs["attn_implementation"] = "eager"
        else:
            model_kwargs["attn_implementation"] = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self.model.eval()
        self._loaded = True
        logger.info(f"Molmo loaded: {model_id}")

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

        start_time = time.perf_counter()

        # Molmo processor takes images and text directly
        inputs = self.processor.process(
            images=[pil_image],
            text=prompt_text,
        )

        # Move to device and cast to model dtype
        model_dtype = next(self.model.parameters()).dtype
        inputs = {
            k: v.to(device=self.model.device, dtype=model_dtype).unsqueeze(0)
            if isinstance(v, torch.Tensor) and v.is_floating_point()
            else (v.to(device=self.model.device).unsqueeze(0) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }

        gen_kwargs = {}
        if self.config.temperature > 0:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = 0.9
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = False

        from transformers import GenerationConfig

        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            use_cache=True,
            **gen_kwargs,
        )

        with torch.inference_mode():
            output_ids = self.model.generate_from_batch(
                inputs,
                generation_config=generation_config,
            )

        # Only decode newly generated tokens
        generated_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        raw_output = self.processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()

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
