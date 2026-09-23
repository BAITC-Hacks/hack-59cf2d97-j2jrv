from __future__ import annotations

from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - dependency may be absent in some environments
    OpenAI = None

from app.config import OPENAI_API_KEY


def _fallback_summary(recommendations: list[dict[str, Any]]) -> str:
    if not recommendations:
        return "Нет рекомендаций для анализа."

    prioritized = sorted(recommendations, key=lambda item: float(item.get("recommended_qty", 0) or 0), reverse=True)
    lines = []
    for item in prioritized[:3]:
        article = item.get("article", "unknown")
        supplier = item.get("supplier", "unknown")
        qty = item.get("recommended_qty", 0)
        lines.append(f"{article} ({supplier}) — приоритетная закупка: {qty} ед.")

    if len(prioritized) > 3:
        lines.append(f"Ещё {len(prioritized) - 3} SKU требуют проверки по запасам и срокам поставки.")

    return " ".join(lines)


def _has_valid_openai_key() -> bool:
    value = str(OPENAI_API_KEY or "").strip()
    return bool(value) and value.startswith("sk-")


def build_ai_summary(recommendations: list[dict[str, Any]], model: str = "gpt-4o-mini") -> str:
    if not recommendations:
        return "Нет рекомендаций для ИИ-анализа."

    if not _has_valid_openai_key() or OpenAI is None:
        return _fallback_summary(recommendations)

    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=10.0, max_retries=0)
        prompt = "\n".join(
            [
                "Проанализируй эти рекомендации по закупкам и кратко объясни, какое решение наиболее важно для бизнеса. "
                "Учитывай, что это регулярная потребность, а разовые крупные заказы должны быть исключены из расчёта.",
                *[
                    (
                        f"- {item.get('article', 'unknown')}: supplier={item.get('supplier', 'unknown')}, "
                        f"recommended_qty={item.get('recommended_qty', 0)}, estimated_demand={item.get('estimated_demand', 0)}"
                    )
                    for item in recommendations[:8]
                ],
            ]
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты консультант по закупкам и запасам. Пиши коротко, понятно и по делу."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=220,
            timeout=10.0,
        )

        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip()
    except Exception:
        pass

    return _fallback_summary(recommendations)
