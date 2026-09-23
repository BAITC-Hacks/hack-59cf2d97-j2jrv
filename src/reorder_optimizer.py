from __future__ import annotations

import pandas as pd


def detect_large_orders(sales: pd.DataFrame, article_col: str = "article", client_col: str = "client_id", qty_col: str = "qty", quantile: float = 0.99) -> tuple[pd.DataFrame, float]:
    if sales.empty:
        return sales.copy(), 0.0

    work = sales[[article_col, client_col, qty_col]].copy()
    thresholds = work.groupby([article_col, client_col])[qty_col].quantile(quantile)
    flagged = sales.merge(thresholds.rename("threshold"), on=[article_col, client_col], how="left")
    flagged = flagged[flagged[qty_col] > flagged["threshold"]].copy()
    return flagged, float(flagged["threshold"].max()) if not flagged.empty else 0.0


def clean_sales_for_forecast(sales: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    cleaned = sales.copy()
    if flagged.empty:
        return cleaned

    key_cols = ["date", "article", "client_id", "qty"]
    excluded = flagged[key_cols].drop_duplicates()
    cleaned = cleaned.merge(excluded.assign(__excluded__=1), on=key_cols, how="left")
    cleaned = cleaned[cleaned["__excluded__"].isna()].drop(columns=["__excluded__"])
    return cleaned.reset_index(drop=True)


def build_recommendation_table(
    sales: pd.DataFrame,
    inventory: pd.DataFrame,
    inbound: pd.DataFrame,
    stockouts: pd.DataFrame,
    suppliers: pd.DataFrame,
) -> pd.DataFrame:
    sales = sales.copy()
    inventory = inventory.copy()
    inbound = inbound.copy()
    stockouts = stockouts.copy()
    suppliers = suppliers.copy()

    if "date" in sales.columns:
        sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    if "date" in inbound.columns:
        inbound["date"] = pd.to_datetime(inbound["date"], errors="coerce")

    flagged, _ = detect_large_orders(sales)
    cleaned_sales = clean_sales_for_forecast(sales, flagged)

    if cleaned_sales.empty:
        cleaned_sales = sales.copy()

    daily_demand = (
        cleaned_sales.groupby("article")["qty"].mean().rename("avg_daily_demand").reset_index()
    )

    inventory_by_article = inventory.groupby("article")["quantity_on_hand"].sum().rename("quantity_on_hand")
    inbound_by_article = inbound.groupby("article")["expected_qty"].sum().rename("expected_qty")
    stockout_by_article = stockouts.groupby("article")[["stockout_days", "lost_sales_units"]].sum()
    supplier_by_article = suppliers.groupby("article").first()

    articles = sorted(set(cleaned_sales["article"].unique()) | set(inventory["article"].unique()) | set(suppliers["article"].unique()))
    rows = []

    for article in articles:
        avg_daily = float(daily_demand.loc[daily_demand["article"] == article, "avg_daily_demand"].sum() if not daily_demand.empty else 0.0)
        if avg_daily <= 0:
            avg_daily = float(cleaned_sales.loc[cleaned_sales["article"] == article, "qty"].mean()) if article in cleaned_sales["article"].unique() else 0.0

        on_hand = float(inventory_by_article.get(article, 0.0))
        inbound_qty = float(inbound_by_article.get(article, 0.0))
        supplier_row = supplier_by_article.loc[article] if article in supplier_by_article.index else None
        supplier_name = supplier_row["supplier"] if supplier_row is not None and "supplier" in supplier_row else "Unknown"
        lead_time = int(supplier_row["lead_time_days"]) if supplier_row is not None and "lead_time_days" in supplier_row else 7
        min_order_qty = float(supplier_row["min_order_qty"]) if supplier_row is not None and "min_order_qty" in supplier_row else 0.0

        stockout_days = float(stockout_by_article.loc[article, "stockout_days"]) if article in stockout_by_article.index else 0.0
        lost_sales = float(stockout_by_article.loc[article, "lost_sales_units"]) if article in stockout_by_article.index else 0.0

        seasonal_index = 1.0
        growth = 0.05 if avg_daily > 0 else 0.0

        safety_stock = max(0.0, avg_daily * max(lead_time * 0.5, 3.0))
        stockout_compensation = max(0.0, avg_daily * stockout_days * 0.4 + lost_sales * 0.1)
        base_need = avg_daily * (lead_time + 7)
        projected_need = base_need * (1 + growth) * seasonal_index
        recommended_qty = max(0.0, projected_need + stockout_compensation + safety_stock - on_hand - inbound_qty)
        if min_order_qty > 0 and recommended_qty > 0:
            recommended_qty = max(recommended_qty, min_order_qty)

        rationale_parts = [
            f"avg_demand={avg_daily:.1f}",
            f"lead_time={lead_time}d",
            f"safety_stock={safety_stock:.1f}",
        ]
        if stockout_days > 0:
            rationale_parts.append(f"stockout={stockout_days}d")
        if lost_sales > 0:
            rationale_parts.append(f"lost_sales={lost_sales:.1f}")

        rows.append(
            {
                "article": article,
                "supplier": supplier_name,
                "recommended_qty": round(float(recommended_qty), 2),
                "reason": "; ".join(rationale_parts),
                "estimated_demand": round(float(projected_need), 2),
                "stock_on_hand": round(float(on_hand), 2),
                "inbound_qty": round(float(inbound_qty), 2),
            }
        )

    recommendations = pd.DataFrame(rows)
    if recommendations.empty:
        return pd.DataFrame(columns=["article", "supplier", "recommended_qty", "reason", "estimated_demand", "stock_on_hand", "inbound_qty"])
    return recommendations.sort_values(["recommended_qty", "article"], ascending=[False, True]).reset_index(drop=True)
