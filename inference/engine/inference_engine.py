"""
High-level inference engine coordinating model execution, caching, and sampling.
"""

from typing import Optional, List, Union, Any
import torch
import torch.nn as nn

from config.model_config import ModelConfig
from config.inference_config import InferenceConfig
from utils.logging_utils import get_logger
from inference.generation.autoregressive import autoregressive_generate
from inference.sampling.sampler import Sampler
from inference.generation.beam_search import BeamSearch

logger = get_logger("inference_engine")


class FullTransformerModel(nn.Module):
    """Fallback modular transformer wrapper combining embeddings, blocks, and lm_head."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        from model.embeddings import EmbeddingLayer
        from model.blocks import BlockStack
        from model.heads import LMHead

        self.embeddings = EmbeddingLayer(config=config)
        self.blocks = BlockStack(config=config)
        self.lm_head = LMHead(config=config)
        self.num_layers = config.n_layers
        if config.tie_weights:
            self.lm_head.tie_weights(self.embeddings.token_embedding.embedding.weight)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        kv_cache = kwargs.get("kv_cache")
        offset = kv_cache.get_seq_length() if kv_cache is not None else 0
        h = self.embeddings(input_ids, offset=offset)
        h = self.blocks(h, **kwargs)
        logits = self.lm_head(h)
        return logits


class InferenceEngine:
    """
    End-to-end inference engine managing model state, token generation loop,
    and sampling orchestration.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        inference_config: Optional[InferenceConfig] = None,
        model: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None
    ):
        self.model_config = model_config if model_config is not None else ModelConfig()

        self.inference_config = inference_config or InferenceConfig()
        self.inference_config.validate()
        self.device = torch.device(device) if device else torch.device("cpu")

        self.model = model if model is not None else FullTransformerModel(self.model_config)

    def _prepare_model(self) -> None:
        """
        Internal lifecycle helper to prepare model state before inference execution.
        """
        if self.model is not None and hasattr(self.model, "train"):
            self.model.to(self.device)
            self.model.eval()

    def generate_tokens(
        self,
        prompt_tokens: List[int],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> List[int]:
        """
        Generate a list of token IDs autoregressively from input token sequence.
        """
        self._prepare_model()

        max_tokens = max_new_tokens if max_new_tokens is not None else self.inference_config.max_new_tokens
        if max_tokens <= 0:
            return list(prompt_tokens)

        temp = temperature if temperature is not None else self.inference_config.temperature
        tokens = list(prompt_tokens)

        if self.model is None:
            raise RuntimeError("No model is available for token generation")
        if self.inference_config.beam_width > 1:
            beam_search = BeamSearch(
                beam_width=self.inference_config.beam_width,
                max_length=max_tokens,
                eos_token_id=(self.inference_config.stop_token_ids
                              if self.inference_config.early_stopping else None),
            )
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
            return beam_search.search(self.model, input_tensor)[0].tolist()
        sampler = Sampler(
            temperature=temp,
            top_k=top_k if top_k is not None else self.inference_config.top_k,
            top_p=top_p if top_p is not None else self.inference_config.top_p,
            repetition_penalty=(repetition_penalty if repetition_penalty is not None
                                else self.inference_config.repetition_penalty),
            seed=self.inference_config.seed,
            device=self.device,
        )

        def sample(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
            return sampler.sample(logits, generated_tokens=history)

        input_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)
        output = autoregressive_generate(
            self.model,
            input_tensor,
            max_new_tokens=max_tokens,
            eos_token_id=(self.inference_config.stop_token_ids
                          if self.inference_config.early_stopping and self.inference_config.stop_token_ids
                          else None),
            sampler_with_history=sample,
            use_cache=self.inference_config.use_kv_cache,
        )
        return output[0].tolist()

    def generate(
        self,
        prompt: str,
        tokenizer: Optional[Any] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        End-to-end string-to-string text generation.
        """
        # Tokenize prompt
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            prompt_ids = tokenizer.encode(prompt)
        else:
            from tokenizer import BPETokenizer
            tokenizer = BPETokenizer()
            prompt_ids = tokenizer.encode(prompt)

        # Generate tokens
        output_ids = self.generate_tokens(
            prompt_tokens=prompt_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

        # Decode generated tokens
        return tokenizer.decode(output_ids)
