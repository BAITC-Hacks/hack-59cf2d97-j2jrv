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

    lines = []
    for item in recommendations[:5]:
        article = item.get("article", "unknown")
        supplier = item.get("supplier", "unknown")
        qty = item.get("recommended_qty", 0)
        lines.append(f"{article}: {supplier}, рекомендовано {qty} единиц.")

    if len(recommendations) > 5:
        lines.append(f"И ещё {len(recommendations) - 5} позиций требуют проверки.")

    return " ".join(lines)


def build_ai_summary(recommendations: list[dict[str, Any]], model: str = "gpt-4o-mini") -> str:
    if not recommendations:
        return "Нет рекомендаций для ИИ-анализа."

    if not OPENAI_API_KEY or OpenAI is None:
        return _fallback_summary(recommendations)

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
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
        )

        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip()
    except Exception:
        pass

    return _fallback_summary(recommendations)
