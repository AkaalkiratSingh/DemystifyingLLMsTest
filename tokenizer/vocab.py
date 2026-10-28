"""Vocabulary manager."""
from typing import Dict, List, Optional
from .special_tokens import PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN
from config.constants import PAD_TOKEN_ID, UNK_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID


class Vocabulary:
    def __init__(self, vocab_dict: Optional[Dict[str, int]] = None):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        if vocab_dict is not None:
            for token, idx in vocab_dict.items():
                self.add_token(token, idx)

        self.pad_token_id = self._ensure_special(PAD_TOKEN, PAD_TOKEN_ID)
        self.unk_token_id = self._ensure_special(UNK_TOKEN, UNK_TOKEN_ID)
        self.bos_token_id = self._ensure_special(BOS_TOKEN, BOS_TOKEN_ID)
        self.eos_token_id = self._ensure_special(EOS_TOKEN, EOS_TOKEN_ID)

    def _ensure_special(self, token: str, preferred_id: int) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if preferred_id not in self._id_to_token:
            return self.add_token(token, preferred_id)
        return self.add_token(token)

    def add_token(self, token: str, idx: Optional[int] = None) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        if idx is None:
            idx = max(self._id_to_token, default=-1) + 1
        if idx in self._id_to_token and self._id_to_token[idx] != token:
            raise ValueError(f"Token ID {idx} is already assigned to {self._id_to_token[idx]!r}")
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token
        return idx

    def token_to_id(self, token: str) -> int:
        if token in self._token_to_id:
            return self._token_to_id[token]
        return self._token_to_id.get(token, self.unk_token_id)

    def id_to_token(self, idx: int) -> str:
        if idx in self._id_to_token:
            return self._id_to_token[idx]
        return self._id_to_token.get(idx, f"<token_{idx}>")

    def __len__(self) -> int:
        return len(self._token_to_id)

    @property
    def size(self) -> int:
        """Return the minimum embedding size that can represent every token ID."""
        return max(self._id_to_token, default=-1) + 1

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id
