"""OpenAI Structured Output drafts for shortlisted real candidates."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import ValidationError

from src.models import FinalPersonalizedOffer, FinalScoredCandidate


LOGGER = logging.getLogger(__name__)


class FinalOfferGenerationError(RuntimeError):
    """Raised for safe, actionable offer-generation failures."""


@dataclass(frozen=True, slots=True)
class FinalOfferGenerationResult:
    """Candidate-aligned offers plus safe per-candidate errors."""

    offers: list[FinalPersonalizedOffer]
    errors: list[str]


class OpenAIFinalOfferProvider:
    """One Responses API Structured Output call per actual finalist."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout_seconds: float,
        prompt: str,
        max_retries: int = 2,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise FinalOfferGenerationError(
                "OPENAI_API_KEY is required for final offer generation. "
                "Use --dry-run-final to inspect limits without API calls."
            )
        if not prompt.strip():
            raise FinalOfferGenerationError("Final offer prompt is empty")
        self.model = model
        self.prompt = prompt
        self._max_retries = max_retries
        self._sleep = sleep
        self._client = client or OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )
        self.call_count = 0
        self.request_attempts = 0
        self.retries = 0
        self.usage: dict[str, int] = {}

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(
            exc,
            (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError),
        ):
            return True
        return isinstance(exc, APIStatusError) and exc.status_code >= 500

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        status = getattr(exc, "status_code", None)
        return (
            f"{type(exc).__name__}, HTTP {status}"
            if isinstance(status, int)
            else type(exc).__name__
        )

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        values = usage.model_dump() if hasattr(usage, "model_dump") else usage
        if not isinstance(values, dict):
            return
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = values.get(key)
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value

    def generate_offer(self, finalist: FinalScoredCandidate) -> FinalPersonalizedOffer:
        """Generate one un-sent draft from allow-listed candidate evidence."""

        self.call_count += 1
        candidate = finalist.candidate
        payload = {
            "candidate": {
                "name": candidate.name,
                "username": candidate.username,
                "platform": candidate.platform.value,
                "profile_url": str(candidate.profile_url),
                "biography": candidate.biography,
                "followers_count": candidate.followers_count,
                "engagement_rate": candidate.engagement_rate,
                "content_formats": candidate.content_formats,
                "data_confidence": candidate.data_confidence,
            },
            "score": finalist.score.total_score,
            "match_reason": finalist.match_reason,
            "evidence": finalist.evidence[:8],
            "business_context": (
                "LD LATTE sells women's clothing on Wildberries and Ozon; "
                "the goal is to discuss a suitable barter format, subject to human review"
            ),
        }
        for attempt in range(self._max_retries + 1):
            try:
                self.request_attempts += 1
                LOGGER.info(
                    "Generating final offer draft: model=%s platform=%s attempt=%d",
                    self.model,
                    candidate.platform.value,
                    attempt + 1,
                )
                response = self._client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": self.prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                payload,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    ],
                    text_format=FinalPersonalizedOffer,
                    store=False,
                )
                self._record_usage(response)
                parsed = getattr(response, "output_parsed", None)
                if parsed is None:
                    raise FinalOfferGenerationError(
                        "OpenAI returned no parsed offer; response content was not logged."
                    )
                offer = FinalPersonalizedOffer.model_validate(parsed)
                if offer.candidate_username.casefold() != candidate.username.casefold():
                    raise FinalOfferGenerationError(
                        "OpenAI offer returned a different candidate_username."
                    )
                return offer
            except (ValidationError, FinalOfferGenerationError):
                raise
            except Exception as exc:
                if self._retryable(exc) and attempt < self._max_retries:
                    self.retries += 1
                    delay = float(2**attempt)
                    LOGGER.warning(
                        "OpenAI offer retry %d/%d after %s; waiting %.1fs",
                        self.retries,
                        self._max_retries,
                        self._safe_error(exc),
                        delay,
                    )
                    self._sleep(delay)
                    continue
                raise FinalOfferGenerationError(
                    "OpenAI final offer request failed after "
                    f"{attempt + 1} attempt(s): {self._safe_error(exc)}."
                ) from exc
        raise FinalOfferGenerationError("OpenAI offer retry loop ended unexpectedly")


def generate_final_offers(
    finalists: list[FinalScoredCandidate],
    provider: Any,
) -> FinalOfferGenerationResult:
    """Call OpenAI only for finalists and preserve rows after partial failures."""

    offers: list[FinalPersonalizedOffer] = []
    errors: list[str] = []
    for finalist in finalists:
        username = finalist.candidate.username
        try:
            offers.append(provider.generate_offer(finalist))
        except Exception as exc:
            safe_error = (
                str(exc)
                if isinstance(exc, FinalOfferGenerationError)
                else f"{type(exc).__name__}: offer generation failed"
            )
            errors.append(f"{username}: {safe_error}"[:2_000])
            offers.append(
                FinalPersonalizedOffer(
                    candidate_username=username,
                    message=(
                        "Черновик не сгенерирован из-за ошибки API. Не отправлять; "
                        "подготовить предложение вручную только после проверки профиля."
                    ),
                    evidence_used=["offer_generation_failed"],
                    human_review_required=True,
                )
            )
    return FinalOfferGenerationResult(offers=offers, errors=errors)
