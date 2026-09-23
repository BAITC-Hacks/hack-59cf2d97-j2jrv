from __future__ import annotations

from typing import Any


def render_ascii_dashboard(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No recommendations\n"

    cols = [
        ("article", 12),
        ("supplier", 16),
        ("recommended_qty", 16),
        ("reason", 40),
    ]

    lines: list[str] = []
    header = " | ".join(f"{name:<{width}}" for name, width in cols)
    divider = "-+-".join("-" * width for _, width in cols)
    lines.append(header)
    lines.append(divider)

    for item in rows:
        values = [
            str(item.get("article", "-"))[: cols[0][1] - 1],
            str(item.get("supplier", "-"))[: cols[1][1] - 1],
            str(item.get("recommended_qty", "0"))[: cols[2][1] - 1],
            str(item.get("reason", "-"))[: cols[3][1] - 1],
        ]
        lines.append(" | ".join(f"{v:<{width}}" for v, (_, width) in zip(values, cols)))

    return "\n".join(lines) + "\n"
