"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
import math
from typing import Dict, Any, Optional, List
from utils.logging_utils import get_logger

logger = get_logger("request_handler")


class RequestValidationError(ValueError):
    """Raised when a completion request contains invalid fields."""


class RequestHandler:
    """
    Handles endpoint request validation, inference invocation, and response formatting.
    """

    def __init__(self, engine: Optional[Any] = None, tokenizer: Optional[Any] = None):
        self.engine = engine
        self.tokenizer = tokenizer

    def handle_completion(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process OpenAI-compatible /v1/completions request.

        Args:
            request_body: Dictionary containing 'prompt', optional 'max_tokens', 'temperature', etc.

        Returns:
            Formatted response dictionary.
        """
        if not isinstance(request_body, dict):
            raise RequestValidationError("request body must be a JSON object")

        prompt = request_body.get("prompt", "")
        max_tokens = request_body.get("max_tokens", 16)
        temperature = request_body.get("temperature", 0.8)
        top_k = request_body.get("top_k")
        top_p = request_body.get("top_p")
        repetition_penalty = request_body.get("repetition_penalty")

        if not isinstance(prompt, str):
            raise RequestValidationError("prompt must be a string")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise RequestValidationError("max_tokens must be a non-negative integer")
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise RequestValidationError("temperature must be a finite non-negative number")
        if top_k is not None and (
            not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0
        ):
            raise RequestValidationError("top_k must be a non-negative integer")
        if top_p is not None and (
            not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 <= top_p <= 1
        ):
            raise RequestValidationError("top_p must be a finite number in [0, 1]")
        if repetition_penalty is not None and (
            not isinstance(repetition_penalty, (int, float))
            or not math.isfinite(repetition_penalty) or repetition_penalty <= 0
        ):
            raise RequestValidationError("repetition_penalty must be a finite positive number")

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            from tokenizer import BPETokenizer
            self.tokenizer = BPETokenizer()
            prompt_tokens = self.tokenizer.encode(prompt)

        # Generate continuation
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        elif self.engine is not None and hasattr(self.engine, "generate"):
            # Keep compatibility with text-only engines that expose the
            # original generate(prompt, tokenizer, max_new_tokens) contract.
            generated_text = self.engine.generate(
                prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens
            )
            generated_tokens = None
        else:
            raise RuntimeError("No inference engine is configured for completion requests")

        if generated_tokens is None:
            response_text = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text
            try:
                completion_count = len(self.tokenizer.encode(response_text, add_bos=False))
            except TypeError:
                completion_count = len(self.tokenizer.encode(response_text))
        elif self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            completion_tokens = generated_tokens[len(prompt_tokens):]
            response_text = self.tokenizer.decode(completion_tokens)
            completion_count = max(0, len(generated_tokens) - len(prompt_tokens))
        else:
            response_text = generated_tokens
            completion_count = max(0, len(generated_tokens) - len(prompt_tokens))

        finish_reason = "length"
        if generated_tokens is not None and self.engine is not None:
            stop_ids = getattr(getattr(self.engine, "inference_config", None), "stop_token_ids", [])
            if generated_tokens and generated_tokens[-1] in stop_ids:
                finish_reason = "stop"

        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "broken-transformer",
            "text": response_text,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": completion_count,
                "total_tokens": len(prompt_tokens) + completion_count
            }
        }

    def handle_health(self) -> Dict[str, Any]:
        """Process /health endpoint check."""
        return {
            "status": "ok",
            "service": "broken-transformer-server",
            "timestamp": time.time()
        }

    def handle_models(self) -> Dict[str, Any]:
        """Process /v1/models endpoint check."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "broken-transformer",
                    "object": "model",
                    "owned_by": "openai-dsg-iitr",
                    "ready": True
                }
            ]
        }
