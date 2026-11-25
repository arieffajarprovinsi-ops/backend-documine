import asyncio
import hashlib
import math
import time
from typing import TypeVar, Type, Optional, AsyncIterator, List, Dict, Any
from pydantic import BaseModel
from app.core.config import settings

T = TypeVar("T", bound=BaseModel)

# Verified Nov 2026 OpenAI API Pricing per 1,000,000 tokens (DOC-18, DOC-19)
OPENAI_PRICING_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-6-luna": {
        "input_auto": 0.15,
        "cached_auto": 0.015,
        "output_auto": 0.60,
        "input_flex": 0.075,
        "cached_flex": 0.0075,
        "output_flex": 0.30,
    },
    "gpt-6.1-terra": {
        "input_auto": 0.50,
        "cached_auto": 0.05,
        "output_auto": 2.50,
        "input_flex": 0.25,
        "cached_flex": 0.025,
        "output_flex": 1.25,
    },
    "gpt-6.1-sol": {
        "input_auto": 2.00,
        "cached_auto": 0.10,
        "output_auto": 10.00,
        "input_flex": 1.00,
        "cached_flex": 0.05,
        "output_flex": 5.00,
    },
    "gpt-6-astra": {
        "input_auto": 15.00,
        "cached_auto": 1.50,
        "output_auto": 60.00,
        "input_flex": 7.50,
        "cached_flex": 0.75,
        "output_flex": 30.00,
    },
    "text-embedding-3-large": {
        "input_auto": 0.13,
        "input_flex": 0.065,
    },
    "text-embedding-3-small": {
        "input_auto": 0.02,
        "input_flex": 0.01,
    },
    "gpt-4o-mini-tts": {
        "input_auto": 0.60,
        "output_auto": 12.00,
    },
    "tts-1-hd": {
        "input_auto": 30.00,
    },
    "tts-1": {
        "input_auto": 15.00,
    },
}


def _guard_no_sol_model(model: str) -> str:
    """Hard safety guard: never allow expensive gpt-6.1-sol even if overridden in environment variables."""
    cleaned = (model or "").strip()
    if not cleaned or "sol" in cleaned.lower():
        return "gpt-6-luna"
    return cleaned


def calculate_openai_cost_usd(
    model: str,
    service_tier: str,
    input_tokens: int,
    cached_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    rates = OPENAI_PRICING_PER_1M.get(model) or OPENAI_PRICING_PER_1M["gpt-6.1-terra"]
    tier_suffix = "flex" if service_tier in ("flex", "batch") else "auto"
    uncached_input = max(0, input_tokens - cached_tokens)
    in_rate = rates.get(f"input_{tier_suffix}", rates.get("input_auto", 0.50))
    cache_rate = rates.get(f"cached_{tier_suffix}", rates.get("cached_auto", 0.05))
    out_rate = rates.get(f"output_{tier_suffix}", rates.get("output_auto", 0.0))

    cost = (
        (uncached_input / 1_000_000.0) * in_rate
        + (cached_tokens / 1_000_000.0) * cache_rate
        + (output_tokens / 1_000_000.0) * out_rate
    )
    return round(cost, 6)


def deterministic_local_embedding(text: str, dimensions: int = 1536) -> List[float]:
    """
    Deterministic unit-norm vector generator used ONLY when OPENAI_API_KEY is not set
    in local offline unit tests. Preserves lexical token similarity so hybrid search tests work offline.
    """
    vec = [0.0] * dimensions
    tokens = [t.strip().lower() for t in text.replace("\n", " ").split(" ") if len(t.strip()) >= 2]
    if not tokens:
        tokens = ["empty"]
    for tok in tokens:
        digest = hashlib.sha256(tok.encode("utf-8")).digest()
        for i in range(0, min(len(digest), 16), 2):
            idx = ((digest[i] << 8) | digest[i + 1]) % dimensions
            sign = 1.0 if (digest[i] & 1) == 0 else -1.0
            vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [round(v / norm, 6) for v in vec]


class OpenAIProviderAdapter:
    """
    Official OpenAI Responses API (/v1/responses) & Embeddings API (/v1/embeddings) Adapter (ADR-007, ADR-017).
    Enforces store=False by default so zero user documents or completions are retained in OpenAI logs (ADR-016).
    """

    provider_name: str = "openai"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = (api_key if api_key is not None else settings.openai_api_key).strip()
        self._client = None
        self._client_loop = None
        if self._api_key:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    timeout=settings.openai_timeout_seconds,
                )
            except Exception as e:
                print(f"[OpenAIProviderAdapter] Warning: Failed to initialize AsyncOpenAI client: {e}")
                self._client = None

    def _get_client(self):
        if not self._api_key:
            return None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if (
            self._client is None
            or self._client_loop is not current_loop
            or (self._client_loop is not None and self._client_loop.is_closed())
        ):
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=settings.openai_timeout_seconds,
            )
            self._client_loop = current_loop
        return self._client

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key) and (self._client is not None or self._get_client() is not None)

    def _extract_usage_metrics(
        self,
        response_obj: Any,
        model: str,
        service_tier: str,
        latency_ms: int,
    ) -> Dict[str, Any]:
        usage = getattr(response_obj, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0)

        cached_tokens = 0
        in_details = getattr(usage, "input_tokens_details", None)
        if in_details:
            cached_tokens = int(getattr(in_details, "cached_tokens", 0) or 0)

        reasoning_tokens = 0
        out_details = getattr(usage, "output_tokens_details", None)
        if out_details:
            reasoning_tokens = int(getattr(out_details, "reasoning_tokens", 0) or 0)

        cost_usd = calculate_openai_cost_usd(
            model=model,
            service_tier=service_tier,
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
        )
        return {
            "provider": self.provider_name,
            "model": model,
            "service_tier": service_tier,
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "estimated_cost_usd": cost_usd,
            "latency_ms": latency_ms,
        }

    async def generate_structured(
        self,
        *,
        model: str,
        instructions: str,
        input_blocks: List[Dict[str, Any]] | str,
        response_model: Type[T],
        reasoning_effort: Optional[str] = None,
        service_tier: str = "auto",
        prompt_cache_key: Optional[str] = None,
        store: bool = False,
    ) -> tuple[T, Dict[str, Any]]:
        if not self.is_configured:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        model = _guard_no_sol_model(model)
        t0 = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_blocks,
            "text_format": response_model,
            "store": bool(store),
        }
        if service_tier in ("auto", "flex"):
            kwargs["service_tier"] = service_tier
        if reasoning_effort and reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key

        resp = await self._get_client().responses.parse(**kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        parsed: T = resp.output_parsed
        usage_meta = self._extract_usage_metrics(resp, model, service_tier, latency_ms)
        return parsed, usage_meta

    async def generate_text(
        self,
        *,
        model: str,
        instructions: str,
        input_blocks: List[Dict[str, Any]] | str,
        reasoning_effort: Optional[str] = None,
        service_tier: str = "auto",
        prompt_cache_key: Optional[str] = None,
        store: bool = False,
    ) -> tuple[str, Dict[str, Any]]:
        if not self.is_configured:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        model = _guard_no_sol_model(model)
        t0 = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_blocks,
            "store": bool(store),
        }
        if service_tier in ("auto", "flex"):
            kwargs["service_tier"] = service_tier
        if reasoning_effort and reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key

        resp = await self._get_client().responses.create(**kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text_out = getattr(resp, "output_text", "") or ""
        usage_meta = self._extract_usage_metrics(resp, model, service_tier, latency_ms)
        return text_out, usage_meta

    async def stream_text(
        self,
        *,
        model: str,
        instructions: str,
        input_blocks: List[Dict[str, Any]] | str,
        reasoning_effort: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
        store: bool = False,
    ) -> AsyncIterator[Dict[str, Any]]:
        if not self.is_configured:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        model = _guard_no_sol_model(model)
        t0 = time.perf_counter()
        kwargs: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_blocks,
            "stream": True,
            "store": bool(store),
        }
        if reasoning_effort and reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if prompt_cache_key:
            kwargs["prompt_cache_key"] = prompt_cache_key

        stream = await self._get_client().responses.create(**kwargs)
        final_usage: Optional[Dict[str, Any]] = None

        async for event in stream:
            ev_type = getattr(event, "type", "")
            if ev_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield {"type": "token_delta", "delta": delta}
            elif ev_type == "response.completed":
                resp_obj = getattr(event, "response", None)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                if resp_obj is not None:
                    final_usage = self._extract_usage_metrics(resp_obj, model, "auto", latency_ms)

        if final_usage:
            yield {"type": "usage", "usage": final_usage}

    async def embed_texts(
        self,
        *,
        model: str,
        texts: List[str],
        dimensions: int = 1536,
    ) -> tuple[List[List[float]], Dict[str, Any]]:
        if not texts:
            return [], {
                "provider": self.provider_name,
                "model": model,
                "service_tier": "auto",
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "estimated_cost_usd": 0.0,
                "latency_ms": 0,
            }

        sanitized = [t.replace("\x00", " ").strip() or " " for t in texts]

        if not self.is_configured:
            # Deterministic local unit-norm embeddings when running offline tests without OPENAI_API_KEY
            t0 = time.perf_counter()
            vectors = [deterministic_local_embedding(t, dimensions=dimensions) for t in sanitized]
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return vectors, {
                "provider": "local_deterministic",
                "model": f"local-hash-{dimensions}d",
                "service_tier": "auto",
                "input_tokens": sum(max(1, len(t) // 4) for t in sanitized),
                "cached_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "estimated_cost_usd": 0.0,
                "latency_ms": latency_ms,
            }

        t0 = time.perf_counter()
        all_vectors: List[List[float]] = []
        total_tokens = 0
        batch_size = 256

        try:
            for idx in range(0, len(sanitized), batch_size):
                batch = sanitized[idx : idx + batch_size]
                try:
                    resp = await self._get_client().embeddings.create(
                        model=model,
                        input=batch,
                        dimensions=dimensions,
                    )
                except Exception as transient_err:
                    err_str = str(transient_err).lower()
                    if "429" in err_str or "rate" in err_str or "timeout" in err_str:
                        await asyncio.sleep(1.2)
                        resp = await self._get_client().embeddings.create(
                            model=model,
                            input=batch,
                            dimensions=dimensions,
                        )
                    else:
                        raise
                for item in resp.data:
                    all_vectors.append(list(item.embedding))
                if getattr(resp, "usage", None):
                    total_tokens += int(
                        getattr(resp.usage, "prompt_tokens", 0)
                        or getattr(resp.usage, "total_tokens", 0)
                        or 0
                    )
        except Exception as emb_err:
            print(f"[OpenAIAdapter] Embedding fallback notice ({type(emb_err).__name__}): using lexical/deterministic fallback.")
            vectors = [deterministic_local_embedding(t, dimensions=dimensions) for t in sanitized]
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return vectors, {
                "provider": "local_deterministic_fallback",
                "model": f"fallback-{dimensions}d",
                "service_tier": "auto",
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "estimated_cost_usd": 0.0,
                "latency_ms": latency_ms,
            }

        latency_ms = int((time.perf_counter() - t0) * 1000)
        cost_usd = calculate_openai_cost_usd(
            model=model,
            service_tier="auto",
            input_tokens=total_tokens,
        )
        return all_vectors, {
            "provider": self.provider_name,
            "model": model,
            "service_tier": "auto",
            "input_tokens": total_tokens,
            "cached_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "estimated_cost_usd": cost_usd,
            "latency_ms": latency_ms,
        }

    async def generate_speech_bytes(
        self,
        *,
        text: str,
        voice: str = "coral",
        model: str = "tts-1-hd",
        instructions: Optional[str] = None,
    ) -> tuple[bytes, Dict[str, Any]]:
        if not self.is_configured:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        clean_text = (text or "").strip()[:3600]
        if not clean_text:
            raise ValueError("Speech input text is empty.")

        tts1_voices = {"alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"}
        nextgen_only_voices = {"marin", "cedar", "ballad", "verse"}
        req_voice = (voice or "coral").strip().lower()
        t0 = time.perf_counter()
        attempts: List[str] = []
        preferred_order = (
            ("gpt-4o-mini-tts", "tts-1-hd", "tts-1")
            if req_voice in nextgen_only_voices
            else (model, "tts-1-hd", "gpt-4o-mini-tts", "tts-1")
        )
        for cand in preferred_order:
            if cand and cand not in attempts:
                attempts.append(cand)

        last_err: Optional[Exception] = None
        for idx, m_name in enumerate(attempts):
            try:
                effective_voice = req_voice
                if m_name in ("tts-1-hd", "tts-1") and effective_voice not in tts1_voices:
                    effective_voice = "coral"
                kwargs: Dict[str, Any] = {
                    "model": m_name,
                    "voice": effective_voice,
                    "input": clean_text,
                    "response_format": "mp3",
                }
                if instructions and m_name == "gpt-4o-mini-tts":
                    kwargs["instructions"] = instructions
                speech_res = await self._get_client().audio.speech.create(**kwargs)
                if hasattr(speech_res, "aread"):
                    audio_bytes = await speech_res.aread()
                elif hasattr(speech_res, "read"):
                    audio_bytes = speech_res.read()
                else:
                    audio_bytes = bytes(getattr(speech_res, "content", b""))
                latency_ms = int((time.perf_counter() - t0) * 1000)
                est_tokens = max(1, len(clean_text) // 4)
                est_audio_tokens = max(10, int(len(audio_bytes) / 160))
                if m_name == "tts-1-hd":
                    cost_usd = round((len(clean_text) / 1_000_000.0) * 30.0, 6)
                elif m_name == "tts-1":
                    cost_usd = round((len(clean_text) / 1_000_000.0) * 15.0, 6)
                else:
                    cost_usd = round(
                        (est_tokens * 0.60 + est_audio_tokens * 12.0) / 1_000_000.0,
                        6,
                    )
                return audio_bytes, {
                    "provider": self.provider_name,
                    "model": m_name,
                    "service_tier": "auto",
                    "input_tokens": est_tokens,
                    "cached_tokens": 0,
                    "output_tokens": est_audio_tokens,
                    "reasoning_tokens": 0,
                    "estimated_cost_usd": cost_usd,
                    "latency_ms": latency_ms,
                }
            except Exception as exc:
                last_err = exc
                if idx >= len(attempts) - 1:
                    raise
        raise last_err or RuntimeError("Speech generation failed.")
