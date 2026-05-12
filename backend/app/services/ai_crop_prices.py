"""LLM-backed crop-price suggestion service.

Powers the "💡 Запропонувати ціни через AI" button on the Settings →
CropPricesCard. Asks gpt-4o-mini for plausible current-season
заготівельні ціни (procurement prices) for the 13 Ukrainian crops in
the user's chosen currency, returning a JSON object the frontend can
drop straight into the form for review-and-edit.

Key design points:
- **OpenAI JSON mode** (`response_format={"type": "json_object"}`)
  forces a valid top-level JSON object so we don't have to regex-parse
  free-form prose.
- **Strict prompt schema** — we name every expected key explicitly.
  Models occasionally hallucinate extra keys or drop some; we
  whitelist + default to None to keep the response shape stable.
- **Suggestion, not authority** — values are explicitly framed as
  *orientational* in the system prompt and again in the UI disclaimer.
  No write happens server-side; the frontend treats the response as
  pre-filled form state.
- **No live market feed** — this is course-project scope. A future
  upgrade would scrape Держстат / agro-exchanges and validate the LLM
  output against real recent numbers.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.integrations.openai.client import OpenAIClient, OpenAIError

log = logging.getLogger(__name__)


# All 13 supported crops — keys MUST match the `CropType` enum slugs
# and the `CropPricesRead` Pydantic fields. Order is the same as
# `ALL_CROPS` in `frontend/src/api/fields.ts` so the JSON keys line up
# with the visible UI rows.
EXPECTED_KEYS: tuple[str, ...] = (
    "wheat", "corn", "sunflower",
    "soybean", "rapeseed",
    "barley", "rye", "oats", "buckwheat",
    "peas",
    "sugar_beet", "potato",
    "corn_silage",
)


# Currency-specific guidance baked into the prompt — keeps the LLM
# anchored in the right ballpark so it doesn't quote UAH numbers for
# a USD request.
_CURRENCY_HINTS: dict[str, str] = {
    "UAH": (
        "Ціни вказуй у гривнях за тонну (UAH/т). "
        "Для України типові ціни: пшениця 7000–9500, кукурудза 6500–8500, "
        "соняшник 15000–22000, цукровий буряк 1500–2500, картопля 8000–18000."
    ),
    "USD": (
        "Quote prices in US dollars per tonne (USD/t). "
        "Typical Ukrainian export bands: wheat 170–230, corn 160–220, "
        "sunflower 380–550, soybean 380–460."
    ),
    "EUR": (
        "Quote prices in euros per tonne (EUR/t). "
        "Use FOB Black-Sea / EU-import reference levels typical for the season."
    ),
}


def _build_prompt(currency: str) -> list[dict[str, str]]:
    """Two-message prompt: a system instruction that pins down the
    schema and a user message that triggers the JSON emission. Split
    intentionally so JSON-mode sees both — some models drop schema
    constraints when they live in a single user-role message."""
    hint = _CURRENCY_HINTS.get(currency, _CURRENCY_HINTS["UAH"])
    keys_csv = ", ".join(EXPECTED_KEYS)
    system = (
        "Ти агрономічний ринковий аналітик. Твоя задача — оцінити "
        "поточні орієнтовні заготівельні ціни на основні українські "
        "сільськогосподарські культури за тонну. "
        f"{hint} "
        "Відповідай ТІЛЬКИ JSON-об'єктом з рівно цими ключами: "
        f"{keys_csv}. Значення — додатні числа (без лапок), цілі або "
        "з одним десятковим знаком, БЕЗ одиниць виміру. "
        "Не додавай жодних інших ключів, коментарів чи пояснень. "
        "Якщо ти не знаєш точної ціни на якусь культуру — все одно дай "
        "правдоподібну орієнтовну цифру; null не використовуй."
    )
    user = (
        f"Оціни поточні орієнтовні ціни у {currency}/тонна для цих "
        f"культур: {keys_csv}. Поверни тільки JSON."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def suggest_prices_via_openai(currency: str) -> dict[str, float]:
    """Call OpenAI JSON-mode → parse → validate.

    Returns a dict keyed by `EXPECTED_KEYS` with float values. Raises
    `OpenAIError` if the call fails, the response can't be parsed,
    or no recognised keys came back (which would mean the model
    ignored the schema instructions entirely — in which case we'd
    rather fail loudly than fill the form with garbage).
    """
    client = OpenAIClient()
    try:
        raw = await client.completion(
            _build_prompt(currency),
            temperature=0.4,            # a bit of variance is fine — these are estimates
            max_tokens=400,             # 13 keys × ~10 tokens = ample headroom
            response_format={"type": "json_object"},
        )
    finally:
        await client.aclose()

    parsed = _extract_json_dict(raw)
    cleaned = _coerce_to_floats(parsed)
    if not cleaned:
        raise OpenAIError(
            f"AI did not return any of the expected crop keys "
            f"(got: {list(parsed)})"
        )
    return cleaned


def _extract_json_dict(raw_response: dict[str, Any]) -> dict[str, Any]:
    """Pull the assistant message out of the chat-completion response
    and parse it. With `response_format=json_object` the content is
    guaranteed to be a JSON string, but we still guard against the
    edge case where OpenAI returns no choices (e.g. all-empty
    content-filter trip)."""
    choices = raw_response.get("choices") or []
    if not choices:
        raise OpenAIError("OpenAI response had no choices")
    content = choices[0].get("message", {}).get("content") or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OpenAIError(
            f"AI response is not valid JSON despite json_object mode: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise OpenAIError("AI response is not a JSON object")
    return parsed


def _coerce_to_floats(parsed: dict[str, Any]) -> dict[str, float]:
    """Validate / sanitise the LLM's response.

    - Whitelist to `EXPECTED_KEYS` — drop hallucinated keys.
    - Coerce each value to float; skip anything non-numeric.
    - Reject obviously-broken values (negative or > 100k) so a model
      glitch can't put `1000000` UAH/t into the user's form.
    """
    out: dict[str, float] = {}
    for key in EXPECTED_KEYS:
        if key not in parsed:
            continue
        v = parsed[key]
        # Accept ints, floats, and stringified numbers; reject everything else.
        try:
            f = float(v) if not isinstance(v, bool) else None  # bool is int subclass
            if f is None:
                continue
        except (TypeError, ValueError):
            continue
        if not (0 < f < 100_000):
            log.warning("AI returned implausible price for %s: %s — dropping", key, v)
            continue
        # Round to one decimal for a clean form value.
        out[key] = round(f, 1)
    return out
