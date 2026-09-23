from __future__ import annotations

import asyncio

import pandas as pd

from app.ai_service import build_ai_summary
from app.visuals import render_ascii_dashboard
from src.data_split import split_by_time
from src.reorder_optimizer import (
    build_recommendation_table,
    clean_sales_for_forecast,
    detect_large_orders,
    evaluate_forecast_quality,
)


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


def test_time_split_and_validation_metrics():
    sales = pd.DataFrame(
        [
            {"date": "2025-01-01", "article": "EL-100", "client_id": "C001", "qty": 10},
            {"date": "2025-01-02", "article": "EL-100", "client_id": "C001", "qty": 9},
            {"date": "2025-01-03", "article": "EL-100", "client_id": "C001", "qty": 11},
            {"date": "2025-01-04", "article": "EL-100", "client_id": "C001", "qty": 10},
            {"date": "2025-01-05", "article": "EL-100", "client_id": "C001", "qty": 10},
            {"date": "2025-01-06", "article": "EL-100", "client_id": "C001", "qty": 12},
            {"date": "2025-01-07", "article": "EL-100", "client_id": "C001", "qty": 11},
            {"date": "2025-01-08", "article": "EL-100", "client_id": "C001", "qty": 13},
            {"date": "2025-01-09", "article": "EL-100", "client_id": "C001", "qty": 14},
            {"date": "2025-01-10", "article": "EL-100", "client_id": "C001", "qty": 12},
            {"date": "2025-01-11", "article": "EL-100", "client_id": "C001", "qty": 8},
            {"date": "2025-01-12", "article": "EL-100", "client_id": "C001", "qty": 9},
            {"date": "2025-01-13", "article": "EL-100", "client_id": "C001", "qty": 10},
            {"date": "2025-01-14", "article": "EL-100", "client_id": "C001", "qty": 11},
            {"date": "2025-01-15", "article": "EL-100", "client_id": "C001", "qty": 10},
            {"date": "2025-01-16", "article": "EL-100", "client_id": "C001", "qty": 12},
        ]
    )
    sales["date"] = pd.to_datetime(sales["date"])
    train, valid = split_by_time(sales, date_col="date", train_ratio=0.75)
    assert len(train) > 0 and len(valid) > 0

    inventory = pd.DataFrame([{"article": "EL-100", "quantity_on_hand": 30}])
    inbound = pd.DataFrame([{"article": "EL-100", "expected_qty": 0}])
    stockouts = pd.DataFrame([{"article": "EL-100", "stockout_days": 0, "lost_sales_units": 0}])
    suppliers = pd.DataFrame([{"article": "EL-100", "supplier": "AlphaParts", "lead_time_days": 7, "min_order_qty": 0}])

    metrics = evaluate_forecast_quality(train, valid, inventory, inbound, stockouts, suppliers)
    assert metrics["validation_rows"] == len(valid)
    assert metrics["actual_total_qty"] == valid["qty"].sum()
    assert metrics["mae"] >= 0
    assert metrics["mape"] >= 0


def test_app_startup_does_not_compute_recommendations_immediately(monkeypatch):
    from app.main import app

    called = {"value": False}

    def fake_get_cached_recommendations():
        called["value"] = True
        raise AssertionError("startup should not load recommendations eagerly")

    monkeypatch.setattr("app.main.get_cached_recommendations", fake_get_cached_recommendations)
    monkeypatch.setattr("app.main.warm_recommendations_cache", lambda: None)

    async def run_lifespan():
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(run_lifespan())
    assert called["value"] is False


def test_quick_preview_mode_skips_heavy_validation(monkeypatch):
    import app.main as app_main

    calls = {"validate": 0}

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def calculate(self):
            return [{"article": "EL-100", "supplier": "AlphaParts", "recommended_qty": 12}]

        def validate(self):
            calls["validate"] += 1
            return {"mae": 0.0, "mape": 0.0}

    monkeypatch.setenv("APP_PREVIEW_MODE", "1")
    monkeypatch.setattr(app_main, "ReorderAgent", FakeAgent)

    recommendations, validation_summary = app_main.get_cached_recommendations()

    assert recommendations
    assert validation_summary.get("mode") == "preview"
    assert calls["validate"] == 0


def test_ai_summary_skips_invalid_openai_key(monkeypatch):
    monkeypatch.setattr("app.ai_service.OPENAI_API_KEY", "placeholder-key", raising=False)
    called = {"value": False}

    class FakeClient:
        @property
        def chat(self):
            called["value"] = True
            raise AssertionError("OpenAI should not be called for placeholder keys")

    monkeypatch.setattr("app.ai_service.OpenAI", lambda *args, **kwargs: FakeClient(), raising=False)

    summary = build_ai_summary([
        {"article": "EL-100", "supplier": "AlphaParts", "recommended_qty": 100, "estimated_demand": 80},
        {"article": "EL-200", "supplier": "MegaParts", "recommended_qty": 70, "estimated_demand": 60},
    ])
    assert not called["value"]
    assert "EL-100" in summary
    assert "AlphaParts" in summary


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


def test_agent_prefers_external_data_when_available(monkeypatch, tmp_path):
    local_csv = tmp_path / "sample_sales.csv"
    local_csv.write_text("date,article,qty\n2025-01-01,LOCAL_ONLY,5\n", encoding="utf-8")

    import pandas as pd

    from app.agent import ReorderAgent

    expected = pd.DataFrame([{"date": "2025-01-01", "article": "EXCEL_ONLY", "qty": 7}])

    monkeypatch.setattr(ReorderAgent, "_has_external_sources", lambda self: True)
    monkeypatch.setattr("app.agent.load_external_sales", lambda: expected)

    agent = ReorderAgent(data_dir=tmp_path)
    data = agent._read_csv(agent.sales_path)

    assert data.equals(expected)
    assert set(data["article"]) == {"EXCEL_ONLY"}


def test_agent_limits_csv_rows_for_quick_checks(tmp_path):
    sales = tmp_path / "sample_sales.csv"
    sales.write_text(
        "date,article,client_id,qty,price\n"
        "2025-01-01,EL-100,C001,10,100\n"
        "2025-01-02,EL-100,C001,11,100\n"
        "2025-01-03,EL-100,C001,12,100\n"
        "2025-01-04,EL-100,C001,13,100\n",
        encoding="utf-8",
    )

    from app.agent import ReorderAgent

    agent = ReorderAgent(data_dir=tmp_path, max_rows=2)
    frame = agent._read_csv(agent.sales_path)

    assert len(frame) == 2
