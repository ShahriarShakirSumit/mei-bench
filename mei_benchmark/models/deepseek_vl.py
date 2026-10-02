"""
DeepSeek-VL model wrapper for MEI benchmark.

Supports deepseek-vl-7b-chat using the deepseek-vl package.
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


class DeepSeekVLWrapper(BaseVLM):
    """Wrapper for DeepSeek-VL models.

    Uses the deepseek-vl package for model loading and inference.

    Supports:
        - deepseek-ai/deepseek-vl-7b-chat
        - deepseek-ai/deepseek-vl-1.3b-chat

    Usage:
        config = VLMConfig(
            model_name="DeepSeek-VL-7B",
            model_id="deepseek-ai/deepseek-vl-7b-chat",
        )
        model = DeepSeekVLWrapper(config)
        model.load_model()
        response = model.predict(image, "What is in the image?")
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.vl_chat_processor = None

    def load_model(self) -> None:
        """Load DeepSeek-VL model, tokenizer, and VL processor."""
        from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
        from transformers import AutoModelForCausalLM

        model_id = self.config.model_id
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.float16)

        logger.info(f"Loading DeepSeek-VL: {model_id}")

        self.vl_chat_processor = VLChatProcessor.from_pretrained(
            model_id,
            cache_dir=self.config.cache_dir,
        )
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
        )
        # Ensure all parameters are the same dtype (DeepSeek-VL SAM uses mixed dtypes)
        self.model = self.model.to(dtype=dtype, device=self.config.device)
        self.model.eval()
        self._loaded = True
        logger.info(f"DeepSeek-VL loaded: {model_id}")

    def predict(
        self,
        image: np.ndarray,
        question: str,
        choices: Optional[list[str]] = None,
    ) -> VLMResponse:
        """Generate a response for a single image-question pair."""
        assert self._loaded, "Model not loaded. Call load_model() first."

        from deepseek_vl.utils.io import load_pil_images

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        prompt_text = self.format_prompt(question, choices)

        # Build DeepSeek-VL conversation format
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt_text}",
                "images": [pil_image],
            },
            {"role": "Assistant", "content": ""},
        ]

        start_time = time.perf_counter()

        # Process the conversation
        # NOTE: VLChatProcessorOutput.to() defaults to bfloat16 for pixel_values.
        # We must pass the model dtype explicitly to avoid dtype mismatch
        # on V100 (which doesn't natively support bfloat16).
        model_dtype = next(self.model.parameters()).dtype
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=[pil_image],
            force_batchify=True,
        ).to(self.model.device, dtype=model_dtype)

        # Run image encoder
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
            "use_cache": True,
        }
        if self.config.temperature > 0:
            gen_kwargs["temperature"] = self.config.temperature

        with torch.inference_mode():
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs,
            )

        raw_output = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()

        latency = (time.perf_counter() - start_time) * 1000
        answer = self.extract_answer(raw_output, choices)

        return VLMResponse(
            answer=answer,
            raw_output=raw_output,
            latency_ms=latency,
            tokens_used=int(outputs.shape[-1]),
        )

    def unload_model(self) -> None:
        """Release model from GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.vl_chat_processor is not None:
            del self.vl_chat_processor
            self.vl_chat_processor = None
        torch.cuda.empty_cache()
        super().unload_model()
