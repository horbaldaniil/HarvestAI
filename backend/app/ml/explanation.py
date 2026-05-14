"""LLM-backed plain-language explanation for yield predictions.

Generates 2-3 Ukrainian sentences explaining WHY the model produced a
particular yield estimate, given the top-5 SHAP feature contributions
plus optional context (oblast baseline, quantile band). Cached on the
Prediction row by the predict-yield worker, so page-views don't
re-call OpenAI — the LLM round-trip happens exactly once per
"Перерахувати" click.

Mirrors the design of `app/services/ai_crop_prices.py`:
- gpt-4o-mini via `OpenAIClient.completion()` with
  `response_format={"type": "json_object"}` so we always get a parseable
  `{"summary": "..."}` shape back.
- Best-effort — every failure mode (no API key, OpenAI down, parse
  error, off-spec length) returns `None` so the prediction still
  persists without a narrative.
- Pure helpers / async function. No DB writes, no global state.

Why not just deterministic Python text? We tried that route in the
plan-mode discussion; the user picked LLM because the prose is much
more natural ("Високий пік NDVI у червні (+0.4 т/га) і достатня
волога ґрунту тримають траєкторію") than a template stitch. The cost
is ≈200 tokens/call — negligible at single-user demo scale.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.integrations.openai.client import OpenAIClient, OpenAIError

log = logging.getLogger(__name__)

# Maximum length of the summary we'll accept from the model. Anything
# longer is treated as a prompt-drift artifact and we fall back to None
# rather than show a 1000-character wall of text in the small card.
_MAX_SUMMARY_CHARS = 600


# Human-readable Ukrainian feature labels mirroring the
# `prediction.featureNames.*` i18n keys on the frontend. Used in the
# prompt so gpt-4o-mini can refer to features by name a human would
# understand ("пік NDVI у червні") instead of the raw slug
# (`ndvi_mean_june`).
_FEATURE_LABELS_UK: dict[str, str] = {
    "ndvi_peak": "пік NDVI у сезоні",
    "ndvi_peak_week": "тиждень піку NDVI",
    "ndvi_mean_may": "середній NDVI у травні",
    "ndvi_mean_june": "середній NDVI у червні",
    "ndvi_mean_july": "середній NDVI у липні",
    "ndvi_mean_august": "середній NDVI у серпні",
    "ndvi_integral": "інтеграл NDVI",
    "ndvi_std": "розкид NDVI",
    "evi_peak": "пік EVI",
    "ndwi_min": "мінімальний NDWI (волога)",
    "savi_peak": "пік SAVI",
    "precip_sum_apr_jul": "сума опадів квітень-липень",
    "temp_mean_apr_jul": "середня температура квітень-липень",
    "heat_stress_days": "дні теплового стресу (T_max > 30°C)",
    "drought_dryspells": "найдовша посушлива серія днів",
    "centroid_lat": "географічна широта поля",
    "centroid_lon": "географічна довгота поля",
    "bdod": "щільність ґрунту",
    "cec": "ємність катіонного обміну ґрунту",
    "clay": "вміст глини в ґрунті",
    "phh2o": "pH ґрунту",
    "sand": "вміст піску в ґрунті",
    "silt": "вміст мулу в ґрунті",
    "soc": "органічний вуглець ґрунту",
    "crop_season_overlap_aprjul": "перекриття вегетації з квітнем-липнем",
    "gdd_proxy": "сума активних температур",
    "precip_crop_weighted": "опади зважено по сезону культури",
    "heat_stress_crop_weighted": "тепловий стрес зважено по сезону",
    "drought_crop_weighted": "посуха зважено по сезону",
    "growing_season_length_months": "тривалість вегетаційного сезону",
}


# Human-readable Ukrainian crop labels (mirrors `CropType.display_uk`).
# Kept inline so this module has no dependency on the SQLAlchemy enum.
_CROP_LABELS_UK: dict[str, str] = {
    "wheat": "пшениці", "corn": "кукурудзи", "sunflower": "соняшнику",
    "soybean": "сої", "rapeseed": "ріпаку",
    "barley": "ячменю", "rye": "жита", "oats": "вівса", "buckwheat": "гречки",
    "peas": "гороху",
    "sugar_beet": "цукрового буряку", "potato": "картоплі",
    "corn_silage": "кукурудзи на силос",
}


def _format_shap_for_prompt(top_shap: list[dict]) -> str:
    """Render the SHAP top-5 list as a compact Ukrainian bulleted block
    that the LLM can consume. Each line: `- <human-name> внесок
    +X.XX т/га (значення фічі = Y.Y)`."""
    if not top_shap:
        return "(SHAP не доступний для цього прогнозу)"
    lines: list[str] = []
    for item in top_shap:
        name = item.get("name", "?")
        label = _FEATURE_LABELS_UK.get(name, name)
        contrib = item.get("contribution")
        val = item.get("value")
        sign = "+" if contrib is not None and contrib >= 0 else ""
        contrib_str = f"{sign}{contrib:.2f}" if contrib is not None else "?"
        val_str = f"{val:.2f}" if isinstance(val, (int, float)) else "—"
        lines.append(f"- {label}: внесок {contrib_str} т/га (значення = {val_str})")
    return "\n".join(lines)


def _build_messages(
    *,
    crop: str,
    value_tha: float,
    oblast_baseline: float | None,
    oblast_yield_tha: float | None,
    q_low: float | None,
    q_high: float | None,
    top_shap: list[dict],
) -> list[dict[str, str]]:
    """Two-message prompt mirroring `ai_crop_prices.py` — system locks
    down the shape + tone, user supplies the per-prediction inputs.

    Two oblast baselines are passed in: `oblast_baseline` is the NDVI
    proxy (kept for compatibility, low-signal), `oblast_yield_tha` is
    the Держстат-published mean yield for the same (oblast, crop)
    (high-signal, agronomically meaningful). The latter drives a
    targeted "gap" directive in the system prompt when the prediction
    diverges by > 20 % from regional reality — surfacing the same
    paradox the UI's OblastComparisonTable surfaces, so the user reads
    one coherent narrative instead of two cards in tension.
    """
    crop_label = _CROP_LABELS_UK.get(crop, crop)

    context_lines = [f"Культура: {crop_label}", f"Прогноз: {value_tha:.2f} т/га"]
    if oblast_baseline is not None and oblast_baseline > 0:
        delta_pct = (value_tha - oblast_baseline) / oblast_baseline * 100
        context_lines.append(
            f"Середнє по області (NDVI-проксі): {oblast_baseline:.2f} — "
            f"різниця {delta_pct:+.1f}%"
        )
    yield_gap_directive = ""
    if oblast_yield_tha is not None and oblast_yield_tha > 0:
        yield_gap_pct = (value_tha - oblast_yield_tha) / oblast_yield_tha * 100
        context_lines.append(
            f"Середня врожайність по області (Держстат): "
            f"{oblast_yield_tha:.2f} т/га — прогноз {yield_gap_pct:+.1f}% "
            f"відносно цього базелайну."
        )
        # Trigger the gap-address directive only when the divergence is
        # large enough to mislead the user (UI shows a red badge in this
        # range). Below 20 % the OblastComparisonTable will likely grey
        # the row out, and the LLM stays on its default narrative.
        if abs(yield_gap_pct) > 20:
            direction = "нижче" if yield_gap_pct < 0 else "вище"
            yield_gap_directive = (
                f" Прогноз на {abs(yield_gap_pct):.0f}% {direction} за "
                f"обласне середнє ({oblast_yield_tha:.1f} т/га). У 1 "
                "реченні поясни цей розрив чесно — спирайся на SHAP-"
                "фактори, якщо вони підтримують напрям. Якщо причин з "
                "SHAP не видно, прямо скажи, що модель не повністю "
                "враховує регіональну специфіку поля (ґрунт, історія "
                "сівозміни, агротехніка). Не обіцяй причинно-наслідкових "
                "зв'язків, яких не підтверджують дані."
            )
    if q_low is not None and q_high is not None:
        context_lines.append(
            f"90% довірчий інтервал: {q_low:.2f}–{q_high:.2f} т/га"
        )
    context_lines.append("Топ-5 факторів моделі (за абсолютним внеском):")
    context_lines.append(_format_shap_for_prompt(top_shap))

    system = (
        "Ти агрономічний аналітик, що пояснює фермеру результат "
        "ML-моделі прогнозу врожайності. Напиши коротке (2–3 речення, "
        "до 280 символів разом) пояснення українською: чому модель "
        "видала саме таку оцінку. Спирайся на топ-фактори SHAP — "
        "переклади їх у природні агрономічні твердження "
        "(не цитуй сирі назви фічей). Якщо є порівняння з областю — "
        "згадай. Якщо є довірчий інтервал — згадай як орієнтир "
        "невизначеності. Тон: спокійний, по суті, без зайвих "
        "лозунгів. Відповідай ТІЛЬКИ JSON-об'єктом виду "
        "{\"summary\": \"...\"} БЕЗ зайвих ключів."
        + yield_gap_directive
    )
    user = "Контекст прогнозу:\n" + "\n".join(context_lines)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


async def generate_explanation_via_llm(
    *,
    crop: str,
    value_tha: float,
    oblast_baseline: float | None = None,
    oblast_yield_tha: float | None = None,
    q_low: float | None = None,
    q_high: float | None = None,
    top_shap: list[dict] | None = None,
) -> str | None:
    """Return a 2-3 sentence Ukrainian narrative explaining the prediction.

    Returns `None` on any failure (no API key, OpenAI error, malformed
    JSON, empty / too-long summary). The prediction still persists
    without it — the UI hides the summary block when this is None.

    When `oblast_yield_tha` is provided AND the prediction diverges from
    it by more than 20 %, the prompt explicitly directs the LLM to
    address the gap in the narrative so the user reads one coherent
    explanation instead of having to reconcile PredictionCard (model
    says good) with OblastComparisonTable (gap is large).
    """
    top_shap = top_shap or []
    messages = _build_messages(
        crop=crop,
        value_tha=value_tha,
        oblast_baseline=oblast_baseline,
        oblast_yield_tha=oblast_yield_tha,
        q_low=q_low,
        q_high=q_high,
        top_shap=top_shap,
    )

    client = OpenAIClient()
    try:
        raw = await client.completion(
            messages,
            temperature=0.5,            # slight variance is welcome for prose
            max_tokens=300,             # 280-char target with breathing room
            response_format={"type": "json_object"},
        )
    except OpenAIError as exc:
        log.info("Skipping LLM explanation for %s: %s", crop, exc)
        return None
    except Exception as exc:  # noqa: BLE001 — defensive: network / parse blows
        log.warning("LLM explanation call failed for %s: %s", crop, exc)
        return None
    finally:
        await client.aclose()

    return _extract_summary(raw)


def _extract_summary(raw_response: dict[str, Any]) -> str | None:
    """Pull `summary` out of the chat-completion response with strict
    shape checks; return None on any deviation."""
    choices = raw_response.get("choices") or []
    if not choices:
        return None
    content = choices[0].get("message", {}).get("content") or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        log.warning("LLM explanation response was not valid JSON: %r", content[:200])
        return None
    if not isinstance(parsed, dict):
        return None
    summary = parsed.get("summary")
    if not isinstance(summary, str):
        return None
    summary = summary.strip()
    if not summary:
        return None
    if len(summary) > _MAX_SUMMARY_CHARS:
        log.warning(
            "LLM explanation exceeded %d chars (got %d) — dropping",
            _MAX_SUMMARY_CHARS, len(summary),
        )
        return None
    return summary


__all__ = ["generate_explanation_via_llm"]
