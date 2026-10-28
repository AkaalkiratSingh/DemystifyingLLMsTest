"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from utils.logging_utils import get_logger

logger = get_logger("request_handler")


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
        prompt = request_body.get("prompt", "")
        max_tokens = request_body.get("max_tokens", 16)
        temperature = request_body.get("temperature", 0.8)

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
                top_k=request_body.get("top_k"),
                top_p=request_body.get("top_p"),
                repetition_penalty=request_body.get("repetition_penalty"),
            )
        elif self.engine is not None and hasattr(self.engine, "generate"):
            generated_text = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
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
