from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
import os
from src.inference_engine.config import SamplingParams
from huggingface_hub import login

class FrontendValidationError(ValueError):
    """Raised when a request cannot be safely admitted by the frontend."""


@dataclass
class Request:
    """Normalized user request passed from the frontend into the engine core."""

    request_id: str
    prompt: str = ""
    prompt_ids: list[int] = field(default_factory=list)
    max_new_tokens: int = 128
    sampling_params: SamplingParams | dict[str, Any] | None = None
    arrival_ts: float = field(default_factory=time.time)
    priority: int = 0
    slo_ttft_ms: float | None = None
    model: str | None = None
    stream: bool = False
    messages: list[dict[str, str]] | None = None

    def __post_init__(self) -> None:
        self.prompt = str(self.prompt or "")
        self.prompt_ids = [int(token_id) for token_id in (self.prompt_ids or [])]
        self.max_new_tokens = max(0, int(self.max_new_tokens))
        if self.max_new_tokens == 0 and self.sampling_params is not None:
            params = self.sampling_params
            if isinstance(params, SamplingParams):
                self.max_new_tokens = max(0, int(params.max_tokens or 0))
            elif isinstance(params, Mapping):
                self.max_new_tokens = max(0, int(params.get("max_tokens") or 0))
        if self.sampling_params is None:
            self.sampling_params = SamplingParams(max_tokens=self.max_new_tokens)
        elif isinstance(self.sampling_params, Mapping):
            self.sampling_params = SamplingParams(**dict(self.sampling_params))
        if isinstance(self.sampling_params, SamplingParams):
            if self.sampling_params.max_tokens is None:
                self.sampling_params.max_tokens = self.max_new_tokens
            self.max_new_tokens = max(self.max_new_tokens, int(self.sampling_params.max_tokens or 0))

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_ids)


class Frontend:
    """Frontend layer for tokenization, validation, and request normalization."""

    def __init__(
        self,
        *,
        tokenizer: Any | None = None,
        model_id: str | None = None,
        max_context_length: int = 32768,
        default_max_new_tokens: int = 128,
        enable_token_cache: bool = True,
        trust_remote_code: bool = True,
        revision: str | None = None,
        token: str | None = None,
        use_fast: bool = True,
    ) -> None:
        self.tokenizer = tokenizer or self._load_hf_tokenizer(
            model_id=model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
            use_fast=use_fast,
        )
        self.model_id = model_id
        self.max_context_length = max(1, int(max_context_length))
        self.default_max_new_tokens = max(0, int(default_max_new_tokens))
        self.enable_token_cache = bool(enable_token_cache)
        self._token_cache: dict[str, list[int]] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        tokenizer_name: str | None = None,
        trust_remote_code: bool = True,
        revision: str | None = None,
        token: str | None = None,
        use_fast: bool = True,
        **kwargs: Any,
    ) -> Frontend:
        """Construct a frontend backed by a Hugging Face tokenizer loaded from ``model_id``."""
        tokenizer = cls._load_hf_tokenizer(
            model_id=tokenizer_name or model_id,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
            use_fast=use_fast,
        )
        return cls(tokenizer=tokenizer, model_id=model_id, **kwargs)

    @staticmethod
    def _load_hf_tokenizer(
        *,
        model_id: str | None = None,
        trust_remote_code: bool = True,
        revision: str | None = None,
        token: str | None = None,
        use_fast: bool = True,
    ) -> Any | None:
        if model_id is None:
            return None
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise ImportError("transformers is required to load a Hugging Face tokenizer") from exc

        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "use_fast": use_fast,
        }
        if revision is not None:
            kwargs["revision"] = revision
        if token is not None:
            kwargs["token"] = token
        return AutoTokenizer.from_pretrained(model_id, **kwargs)

    def _cache_key(self, prompt: str, *, chat_template: bool = False) -> str:
        return f"{int(chat_template)}:{prompt}"

    def _encode_text(self, text: str) -> list[int]:
        if not text:
            return []
        cache_key = self._cache_key(text)
        if self.enable_token_cache and cache_key in self._token_cache:
            return self._token_cache[cache_key].copy()

        if self.tokenizer is None:
            raise FrontendValidationError("Tokenizer not available. Add a Tokenizer")
        if hasattr(self.tokenizer, "encode"):
            encoded = self.tokenizer.encode(text)
            ids = _as_int_list(encoded)
        else:
            encoded = self.tokenizer(text)
            if isinstance(encoded, Mapping):
                values = encoded.get("input_ids")
            elif isinstance(encoded, Sequence) and not isinstance(encoded, (str, bytes)):
                values = encoded
            else:
                raise FrontendValidationError("tokenizer did not return input_ids")
            ids = _as_int_list(values)

        if self.enable_token_cache:
            self._token_cache[cache_key] = ids.copy()
        return ids

    def _encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        if not messages:
            return []
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=None,
                )
            except TypeError:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            if encoded is not None:
                return _as_int_list(encoded)
        return self._encode_text(
            "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}".strip()
                for msg in messages
                if isinstance(msg, Mapping)
            )
        )

    def _normalize_prompt(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        prompt_ids: Iterable[int] | None = None,
    ) -> tuple[str, list[int]]:
        if prompt is not None and messages is not None:
            raise FrontendValidationError("prompt and messages are mutually exclusive")

        if messages is not None:
            prompt_text = self._encode_messages(messages)
            return self.detokenize(prompt_text), prompt_text

        if prompt is not None:
            return prompt, self._encode_text(prompt)

        if prompt_ids is not None:
            tokens = _as_int_list(prompt_ids)
            return "", tokens

        raise FrontendValidationError("a prompt, prompt_ids, or messages list is required")

    def validate_request(self, request: Request | Mapping[str, Any]) -> Request:
        """Validate a normalized request and return the same request in canonical form."""
        if isinstance(request, Request):
            normalized = request
        else:
            raw = dict(request)
            prompt = raw.get("prompt")
            messages = raw.get("messages")
            prompt_ids = raw.get("prompt_ids")
            normalized_prompt, normalized_ids = self._normalize_prompt(
                prompt=prompt,
                messages=messages,
                prompt_ids=prompt_ids,
            )
            params = raw.get("sampling_params")
            if params is None:
                params = SamplingParams(max_tokens=int(raw.get("max_new_tokens", self.default_max_new_tokens)))
                request_id = str(raw.get("request_id") or f"req-{int(time.time()*1000000)}")
                normalized = Request(
                    request_id=request_id,
                    prompt=normalized_prompt,
                    prompt_ids=normalized_ids,
                    max_new_tokens=int(raw.get("max_new_tokens", params.max_tokens or self.default_max_new_tokens)),
                    sampling_params=params,
                    arrival_ts=float(raw.get("arrival_ts", time.time())),
                    priority=int(raw.get("priority", 0)),
                    slo_ttft_ms=(None if raw.get("slo_ttft_ms") is None else float(raw["slo_ttft_ms"])),
                    model=raw.get("model"),
                    stream=bool(raw.get("stream", False)),
                    messages=messages,
            )

        if not normalized.prompt_ids and not normalized.prompt:
            raise FrontendValidationError("empty prompt is not allowed")

        if len(normalized.prompt_ids) > self.max_context_length:
            raise FrontendValidationError(
                f"prompt length exceeds max_context_length={self.max_context_length}"
            )

        required = int(normalized.max_new_tokens or 0)
        if required < 0:
            raise FrontendValidationError("max_new_tokens must be >= 0")

        if (len(normalized.prompt_ids) + required) > self.max_context_length:
            raise FrontendValidationError(
                "prompt and generated tokens exceed the configured context limit"
            )

        if isinstance(normalized.sampling_params, SamplingParams):
            if normalized.sampling_params.max_tokens is None:
                normalized.sampling_params.max_tokens = required
            normalized.max_new_tokens = max(required, int(normalized.sampling_params.max_tokens or 0))
        elif isinstance(normalized.sampling_params, Mapping):
            params = dict(normalized.sampling_params)
            if params.get("max_tokens") is None:
                params["max_tokens"] = required
            normalized.sampling_params = SamplingParams(**params)
            normalized.max_new_tokens = max(required, int(normalized.sampling_params.max_tokens or 0))

        if normalized.max_new_tokens and normalized.max_new_tokens > self.max_context_length:
            raise FrontendValidationError("max_new_tokens exceeds the model context limit")

        return normalized

    def prepare_request(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        prompt_ids: Iterable[int] | None = None,
        sampling_params: SamplingParams | dict[str, Any] | None = None,
        request_id: str | None = None,
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ) -> Request:
        """Convenience wrapper that tokenizes and validates one request."""

        finalized_prompt, finalized_ids = self._normalize_prompt(
            prompt=prompt,
            messages=messages,
            prompt_ids=prompt_ids,
        )
        params = sampling_params or SamplingParams()
        if max_new_tokens is not None:
            if isinstance(params, SamplingParams):
                params.max_tokens = max_new_tokens
            else:
                params = dict(params)
                params["max_tokens"] = max_new_tokens
        request = Request(
            request_id=str(request_id or f"req-{int(time.time()*1000000)}"),
            prompt=finalized_prompt,
            prompt_ids=finalized_ids,
            max_new_tokens=int(max_new_tokens or getattr(params, "max_tokens", self.default_max_new_tokens) or self.default_max_new_tokens),
            sampling_params=params,
            arrival_ts=float(kwargs.get("arrival_ts", time.time())),
            priority=int(kwargs.get("priority", 0)),
            slo_ttft_ms=(None if kwargs.get("slo_ttft_ms") is None else float(kwargs["slo_ttft_ms"])),
            model=kwargs.get("model"),
            stream=bool(kwargs.get("stream", False)),
            messages=messages,
        )
        return self.validate_request(request)

    def tokenize(self, text: str) -> list[int]:
        return self._encode_text(text)

    def detokenize(self, token_ids: Iterable[int]) -> str:
        ids = _as_int_list(token_ids)
        if not ids:
            return ""
        if hasattr(self.tokenizer, "decode"):
            try:
                return self.tokenizer.decode(ids, skip_special_tokens=True)
            except TypeError:
                return self.tokenizer.decode(ids)
        raise FrontendValidationError("Tokenizer not Available")

    def admit(self, request: Request, *,
              queue_state: Mapping[str, Any] | None = None,
              kv_free_blocks: int = 0) -> bool:
        """Simple explicit admission gate."""
        if queue_state is None:
            queue_state = {}
        estimated_wait = float(queue_state.get("estimated_wait_ms", 0.0))
        if request.slo_ttft_ms is not None and estimated_wait > request.slo_ttft_ms:
            return False
        required_blocks = max(1, (len(request.prompt_ids) + request.max_new_tokens + self.max_context_length - 1) // self.max_context_length)
        if kv_free_blocks < required_blocks:
            return False
        return True


def _as_int_list(values: Iterable[int] | Sequence[int] | Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        return [ord(ch) for ch in str(values)]
    if isinstance(values, Mapping):
        values = values.get("input_ids", [])
    if isinstance(values, (list, tuple, set)):
        return [int(token_id) for token_id in values]
    if hasattr(values, "tolist"):
        return [int(token_id) for token_id in values.tolist()]
    return [int(token_id) for token_id in values]


__all__ = ["Frontend", "FrontendValidationError", "Request"]
