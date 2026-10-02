"""
Abstract base class for Vision-Language Model wrappers.

All model adapters must implement this interface to work with
the MEI evaluation pipeline.
"""

from __future__ import annotations

from mei_benchmark.paths import MODEL_CACHE

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VLMResponse:
    """Structured response from a VLM.

    Attributes:
        answer: The extracted answer string
        raw_output: Full model output text
        logprobs: Optional log-probabilities for the answer tokens
        confidence: Optional confidence score (0-1)
        latency_ms: Response latency in milliseconds
        tokens_used: Number of tokens consumed (input + output)
    """
    answer: str
    raw_output: str
    logprobs: Optional[list[float]] = None
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    tokens_used: int = 0


@dataclass
class VLMConfig:
    """Configuration for a VLM wrapper.

    Attributes:
        model_name: Human-readable model name (used in reports)
        model_id: Model identifier (e.g., huggingface ID or API model name)
        device: Device string (e.g., 'cuda:0', 'cpu')
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0 = greedy)
        batch_size: Batch size for offline inference
        dtype: Data type ('float16', 'bfloat16', 'float32')
        use_flash_attention: Whether to use flash attention
        cache_dir: Directory for model cache
        extra: Additional model-specific kwargs
    """
    model_name: str = "unknown"
    model_id: str = ""
    device: str = "cuda:0"
    max_new_tokens: int = 128
    temperature: float = 0.0
    batch_size: int = 1
    dtype: str = "float16"
    use_flash_attention: bool = True
    cache_dir: str = str(MODEL_CACHE)
    extra: dict = field(default_factory=dict)


class BaseVLM(ABC):
    """Abstract base class for Vision-Language Model wrappers.

    All model adapters used in MEI evaluation must implement:
    - load_model(): Initialize model weights and processor
    - predict(): Answer a visual question given an image
    - predict_batch(): Batch prediction for efficiency

    Usage:
        model = LLaVAWrapper(config)
        model.load_model()
        response = model.predict(image, "What color is the car?")
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        self._loaded = False

    @property
    def name(self) -> str:
        return self.config.model_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and processor into memory.

        Should be called once before predict(). Sets self._loaded = True.
        """
        ...

    @abstractmethod
    def predict(
        self,
        image: np.ndarray,
        question: str,
        choices: Optional[list[str]] = None,
    ) -> VLMResponse:
        """Generate a response for a single image-question pair.

        Args:
            image: BGR image as numpy array (H, W, 3)
            question: The question string
            choices: Optional answer choices for multiple-choice

        Returns:
            VLMResponse with extracted answer and metadata
        """
        ...

    def predict_batch(
        self,
        images: list[np.ndarray],
        questions: list[str],
        choices_list: Optional[list[list[str]]] = None,
    ) -> list[VLMResponse]:
        """Batch prediction. Default: sequential fallback.

        Override in subclasses for true batched inference.

        Args:
            images: List of BGR images
            questions: List of question strings
            choices_list: Optional list of choice lists

        Returns:
            List of VLMResponse
        """
        results = []
        for i, (img, q) in enumerate(zip(images, questions)):
            choices = choices_list[i] if choices_list else None
            results.append(self.predict(img, q, choices))
        return results

    def format_prompt(
        self,
        question: str,
        choices: Optional[list[str]] = None,
    ) -> str:
        """Format a question into a prompt for the model.

        Default implementation. Override for model-specific prompts.

        Args:
            question: The question
            choices: Optional answer choices

        Returns:
            Formatted prompt string
        """
        if choices:
            options = "\n".join(
                f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
            )
            return (
                f"{question}\n\n{options}\n\n"
                "Answer with just the letter of the correct option."
            )
        return f"{question}\nAnswer briefly in one or two words."

    def extract_answer(
        self,
        raw_output: str,
        choices: Optional[list[str]] = None,
    ) -> str:
        """Extract a clean answer from model output.

        Args:
            raw_output: Raw model output text
            choices: Optional answer choices

        Returns:
            Extracted answer string
        """
        text = raw_output.strip()

        if choices:
            # Try to extract letter answer (A), (B), etc.
            for i, choice in enumerate(choices):
                letter = chr(65 + i)
                if text.upper().startswith(letter) or f"({letter})" in text.upper():
                    return choice
            # Fallback: check if any choice is mentioned
            text_lower = text.lower()
            for choice in choices:
                if choice.lower() in text_lower:
                    return choice

        # Return the first line, cleaned
        first_line = text.split("\n")[0].strip()
        # Remove common prefixes
        for prefix in ["Answer:", "The answer is", "A:"]:
            if first_line.lower().startswith(prefix.lower()):
                first_line = first_line[len(prefix):].strip()

        return first_line

    def unload_model(self) -> None:
        """Release model from memory. Override for cleanup."""
        self._loaded = False
        logger.info(f"Unloaded model: {self.config.model_name}")

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<{self.__class__.__name__}({self.config.model_name}, {status})>"
