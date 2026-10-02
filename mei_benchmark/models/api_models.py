"""
API-based model wrappers for MEI benchmark.

Supports closed-source models: GPT-4o, Claude 3.5, Gemini 1.5 Pro.
Uses base64-encoded images sent via API.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from mei_benchmark.models.base import BaseVLM, VLMConfig, VLMResponse

logger = logging.getLogger(__name__)


def _numpy_to_base64(image: np.ndarray, fmt: str = "jpeg") -> str:
    """Convert a BGR numpy image to base64 string."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format=fmt.upper())
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class GPT4oWrapper(BaseVLM):
    """Wrapper for OpenAI GPT-4o / GPT-4V API.

    Requires OPENAI_API_KEY environment variable.

    Usage:
        config = VLMConfig(
            model_name="GPT-4o",
            model_id="gpt-4o",
            extra={"api_key_env": "OPENAI_API_KEY"},
        )
        model = GPT4oWrapper(config)
        model.load_model()
        response = model.predict(image, "What is in the image?")
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.client = None

    def load_model(self) -> None:
        """Initialize OpenAI client."""
        import os
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        api_key_env = self.config.extra.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Set {api_key_env} environment variable")

        self.client = OpenAI(api_key=api_key)
        self._loaded = True
        logger.info(f"OpenAI client initialized for {self.config.model_id}")

    def predict(
        self,
        image: np.ndarray,
        question: str,
        choices: Optional[list[str]] = None,
    ) -> VLMResponse:
        """Generate a response via OpenAI API."""
        assert self._loaded, "Call load_model() first."

        b64_image = _numpy_to_base64(image)
        prompt_text = self.format_prompt(question, choices)

        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )

        latency = (time.perf_counter() - start_time) * 1000
        raw_output = response.choices[0].message.content or ""
        tokens_used = (
            (response.usage.prompt_tokens + response.usage.completion_tokens)
            if response.usage else 0
        )

        answer = self.extract_answer(raw_output, choices)

        return VLMResponse(
            answer=answer,
            raw_output=raw_output,
            latency_ms=latency,
            tokens_used=tokens_used,
        )


class ClaudeWrapper(BaseVLM):
    """Wrapper for Anthropic Claude 3.5 Sonnet API.

    Requires ANTHROPIC_API_KEY environment variable.

    Usage:
        config = VLMConfig(
            model_name="Claude-3.5-Sonnet",
            model_id="claude-sonnet-4-20250514",
            extra={"api_key_env": "ANTHROPIC_API_KEY"},
        )
        model = ClaudeWrapper(config)
        model.load_model()
        response = model.predict(image, "What is in the image?")
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.client = None

    def load_model(self) -> None:
        """Initialize Anthropic client."""
        import os
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        api_key_env = self.config.extra.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Set {api_key_env} environment variable")

        self.client = anthropic.Anthropic(api_key=api_key)
        self._loaded = True
        logger.info(f"Anthropic client initialized for {self.config.model_id}")

    def predict(
        self,
        image: np.ndarray,
        question: str,
        choices: Optional[list[str]] = None,
    ) -> VLMResponse:
        """Generate a response via Anthropic API."""
        assert self._loaded, "Call load_model() first."

        b64_image = _numpy_to_base64(image)
        prompt_text = self.format_prompt(question, choices)

        start_time = time.perf_counter()

        response = self.client.messages.create(
            model=self.config.model_id,
            max_tokens=self.config.max_new_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_image,
                            },
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            temperature=self.config.temperature,
        )

        latency = (time.perf_counter() - start_time) * 1000
        raw_output = response.content[0].text if response.content else ""
        tokens_used = (
            response.usage.input_tokens + response.usage.output_tokens
        )

        answer = self.extract_answer(raw_output, choices)

        return VLMResponse(
            answer=answer,
            raw_output=raw_output,
            latency_ms=latency,
            tokens_used=tokens_used,
        )


class GeminiWrapper(BaseVLM):
    """Wrapper for Google Gemini 1.5 Pro API.

    Requires GOOGLE_API_KEY environment variable.

    Usage:
        config = VLMConfig(
            model_name="Gemini-1.5-Pro",
            model_id="gemini-1.5-pro",
            extra={"api_key_env": "GOOGLE_API_KEY"},
        )
        model = GeminiWrapper(config)
        model.load_model()
        response = model.predict(image, "What is in the image?")
    """

    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.client = None

    def load_model(self) -> None:
        """Initialize Gemini client."""
        import os
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")

        api_key_env = self.config.extra.get("api_key_env", "GOOGLE_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Set {api_key_env} environment variable")

        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.config.model_id)
        self._loaded = True
        logger.info(f"Gemini client initialized for {self.config.model_id}")

    def predict(
        self,
        image: np.ndarray,
        question: str,
        choices: Optional[list[str]] = None,
    ) -> VLMResponse:
        """Generate a response via Gemini API."""
        assert self._loaded, "Call load_model() first."

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        prompt_text = self.format_prompt(question, choices)

        start_time = time.perf_counter()

        response = self.client.generate_content(
            [pil_image, prompt_text],
            generation_config={
                "max_output_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
            },
        )

        latency = (time.perf_counter() - start_time) * 1000
        raw_output = response.text if response.text else ""

        answer = self.extract_answer(raw_output, choices)

        return VLMResponse(
            answer=answer,
            raw_output=raw_output,
            latency_ms=latency,
            tokens_used=0,  # Gemini doesn't always return usage
        )


# Model registry for CLI/config-based instantiation
MODEL_REGISTRY: dict[str, type[BaseVLM]] = {
    "gpt-4o": GPT4oWrapper,
    "gpt-4-vision-preview": GPT4oWrapper,
    "claude-sonnet-4-20250514": ClaudeWrapper,
    "claude-3-5-sonnet-20241022": ClaudeWrapper,
    "gemini-1.5-pro": GeminiWrapper,
    "gemini-2.0-flash": GeminiWrapper,
}


def get_api_model(model_id: str, model_name: Optional[str] = None, **kwargs) -> BaseVLM:
    """Factory function to get an API model wrapper.

    Args:
        model_id: The API model identifier
        model_name: Human-readable name (defaults to model_id)
        **kwargs: Additional VLMConfig fields

    Returns:
        Appropriate API model wrapper
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown API model: {model_id}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    config = VLMConfig(
        model_name=model_name or model_id,
        model_id=model_id,
        **kwargs,
    )
    return MODEL_REGISTRY[model_id](config)
