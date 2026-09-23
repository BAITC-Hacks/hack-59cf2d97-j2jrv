from __future__ import annotations

import pandas as pd

from app.ai_service import build_ai_summary
from app.visuals import render_ascii_dashboard
from src.reorder_optimizer import build_recommendation_table, clean_sales_for_forecast, detect_large_orders


def test_large_order_is_filtered_from_regular_demand():
    sales = pd.DataFrame(
        [
            {"date": "2025-01-01", "article": "EL-100", "client_id": "C001", "qty": 12, "price": 100},
            {"date": "2025-01-02", "article": "EL-100", "client_id": "C001", "qty": 14, "price": 100},
            {"date": "2025-01-03", "article": "EL-100", "client_id": "C002", "qty": 10, "price": 100},
            {"date": "2025-01-04", "article": "EL-100", "client_id": "C001", "qty": 250, "price": 100},
        ]
    )
    sales["date"] = pd.to_datetime(sales["date"])

    flagged, threshold = detect_large_orders(sales)
    assert not flagged.empty
    assert threshold > 0

    cleaned = clean_sales_for_forecast(sales, flagged)
    assert len(cleaned) < len(sales)
    assert cleaned["qty"].sum() < sales["qty"].sum()


def test_stockout_increases_recommendation():
    sales = pd.DataFrame(
        [
            {"date": "2025-01-01", "article": "EL-200", "client_id": "C001", "qty": 10, "price": 80},
            {"date": "2025-01-02", "article": "EL-200", "client_id": "C001", "qty": 10, "price": 80},
            {"date": "2025-01-03", "article": "EL-200", "client_id": "C002", "qty": 9, "price": 80},
            {"date": "2025-01-04", "article": "EL-200", "client_id": "C002", "qty": 11, "price": 80},
        ]
    )
    sales["date"] = pd.to_datetime(sales["date"])

    inventory = pd.DataFrame([{"article": "EL-200", "quantity_on_hand": 5}])
    inbound = pd.DataFrame([{"article": "EL-200", "expected_qty": 0}])
    stockouts = pd.DataFrame([{"article": "EL-200", "stockout_days": 7, "lost_sales_units": 30}])
    suppliers = pd.DataFrame([{"article": "EL-200", "supplier": "MegaParts", "lead_time_days": 10, "min_order_qty": 0}])

    recommendations = build_recommendation_table(sales, inventory, inbound, stockouts, suppliers)
    assert not recommendations.empty
    assert recommendations.loc[0, "recommended_qty"] > 0
    assert recommendations.loc[0, "supplier"] == "MegaParts"


def test_ai_summary_works_without_openai_key(monkeypatch):
    monkeypatch.setattr("app.ai_service.OPENAI_API_KEY", "", raising=False)
    summary = build_ai_summary([
        {"article": "EL-100", "supplier": "AlphaParts", "recommended_qty": 100, "estimated_demand": 80},
        {"article": "EL-200", "supplier": "MegaParts", "recommended_qty": 70, "estimated_demand": 60},
    ])
    assert "EL-100" in summary
    assert "AlphaParts" in summary


def test_ascii_dashboard_renders_table():
    rows = [
        {"article": "EL-100", "supplier": "AlphaParts", "recommended_qty": 120.0, "reason": "avg_demand=12"},
        {"article": "EL-200", "supplier": "MegaParts", "recommended_qty": 80.0, "reason": "avg_demand=9"},
    ]
    dashboard = render_ascii_dashboard(rows)
    assert "EL-100" in dashboard
    assert "AlphaParts" in dashboard
    assert "recommended_qty" in dashboard.lower()
