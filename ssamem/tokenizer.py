from __future__ import annotations

import re
from typing import Iterable, Sequence

import torch


class SimpleTokenizer:
    """
    Minimal offline tokenizer for local demos.

    It is intentionally simple and only covers the methods used by the
    standalone ssamem pipeline.
    """

    def __init__(self, vocab_size: int = 512) -> None:
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.padding_side = "left"
        self._vocab_size = vocab_size
        self._vocab = {
            self.pad_token: self.pad_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.unk_token: self.unk_token_id,
        }
        self._inverse_vocab = {idx: token for token, idx in self._vocab.items()}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def _split(self, text: str) -> list[str]:
        tokens = re.findall(r"<PTR_0x[0-9A-Fa-f]+>|\w+|[^\w\s]", text)
        return tokens or [self.unk_token]

    def _token_to_id(self, token: str) -> int:
        if token in self._vocab:
            return self._vocab[token]
        if len(self._vocab) < self._vocab_size:
            next_id = len(self._vocab)
            self._vocab[token] = next_id
            self._inverse_vocab[next_id] = token
            return next_id
        return self.unk_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        token_ids = [self._token_to_id(token) for token in self._split(text)]
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        return token_ids

    def __call__(
        self,
        texts: str | Sequence[str],
        *,
        padding: bool = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor | list[list[int]]]:
        if isinstance(texts, str):
            texts = [texts]

        encoded = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if truncation and max_length is not None:
            encoded = [tokens[-max_length:] for tokens in encoded]

        if not padding:
            if return_tensors == "pt":
                raise ValueError("`return_tensors='pt'` requires `padding=True` in SimpleTokenizer.")
            return {"input_ids": encoded}

        max_len = max(len(tokens) for tokens in encoded)
        input_ids = []
        attention_mask = []
        for tokens in encoded:
            pad_len = max_len - len(tokens)
            if self.padding_side == "left":
                padded = [self.pad_token_id] * pad_len + tokens
                mask = [0] * pad_len + [1] * len(tokens)
            else:
                padded = tokens + [self.pad_token_id] * pad_len
                mask = [1] * len(tokens) + [0] * pad_len
            input_ids.append(padded)
            attention_mask.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids: Iterable[int] | torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        tokens = []
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            tokens.append(self._inverse_vocab.get(int(token_id), self.unk_token))
        return " ".join(tokens).strip()

    def batch_decode(self, batch_token_ids: Sequence[Iterable[int] | torch.Tensor], skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(token_ids, skip_special_tokens=skip_special_tokens) for token_ids in batch_token_ids]
