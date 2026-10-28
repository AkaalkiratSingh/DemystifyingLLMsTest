"""
End-to-end serving pipeline connecting tokenization, model execution, and decoding.
"""

from typing import Optional, Any, Union, List
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from utils.logging_utils import get_logger
from inference.generation.autoregressive import autoregressive_generate
from inference.sampling.sampler import Sampler

logger = get_logger("serving_pipeline")


class ServingPipeline:
    """
    End-to-end request processor: text -> tokenizer -> model -> decoding -> response.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config if config is not None else ModelConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        invalid_stop_ids = [
            token_id for token_id in self.inference_config.stop_token_ids
            if token_id >= self.config.vocab_size
        ]
        if invalid_stop_ids:
            raise ValueError(
                f"stop_token_ids must be smaller than model vocab_size "
                f"({self.config.vocab_size}); got {invalid_stop_ids}"
            )
        self.device = torch.device(device) if device else torch.device("cpu")

    def process(self, text: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Execute full inference pipeline for an incoming prompt.

        Args:
            text: Input prompt string.
            max_new_tokens: Optional token generation count override.

        Returns:
            Decoded completion string.
        """
        # Step 1: Tokenization
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            tokens = self.tokenizer.encode(text)
        else:
            from tokenizer import BPETokenizer
            self.tokenizer = BPETokenizer()
            tokens = self.tokenizer.encode(text)

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Step 2: Autoregressive model execution
        if self.model is None or not callable(self.model):
            raise RuntimeError("No model is available to process the request")
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

        sampler = Sampler(
            temperature=self.inference_config.temperature,
            top_k=self.inference_config.top_k,
            top_p=self.inference_config.top_p,
            repetition_penalty=self.inference_config.repetition_penalty,
            seed=self.inference_config.seed,
            device=self.device,
        )

        def sample(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
            return sampler.sample(logits, generated_tokens=history)

        max_tokens = (max_new_tokens if max_new_tokens is not None
                      else self.inference_config.max_new_tokens)
        output = autoregressive_generate(
            self.model,
            input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=(self.inference_config.stop_token_ids
                          if self.inference_config.early_stopping else None),
            sampler_with_history=sample,
            use_cache=self.inference_config.use_kv_cache,
        )

        # Step 3: Decoding
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(output[0].tolist())

        raise RuntimeError("A tokenizer with a decode method is required to return text")
