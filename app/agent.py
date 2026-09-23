from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_split import split_by_time
from src.external_data_loader import (
    load_external_inbound,
    load_external_inventory,
    load_external_sales,
    load_external_stockouts,
    load_external_suppliers,
)
from src.reorder_optimizer import build_recommendation_table, evaluate_forecast_quality


class ReorderAgent:
    def __init__(
        self,
        data_dir: str | Path = "data",
        sales_path: str | Path | None = None,
        inventory_path: str | Path | None = None,
        inbound_path: str | Path | None = None,
        stockouts_path: str | Path | None = None,
        suppliers_path: str | Path | None = None,
        max_rows: int | None = None,
    ) -> None:
        base_dir = Path(data_dir)
        if str(data_dir) == "data" and (Path("data") / "real").exists():
            base_dir = Path("data") / "real"
        self.data_dir = base_dir
        self.sales_path = Path(sales_path) if sales_path else self.data_dir / "sample_sales.csv"
        self.inventory_path = Path(inventory_path) if inventory_path else self.data_dir / "inventory.csv"
        self.inbound_path = Path(inbound_path) if inbound_path else self.data_dir / "inbound.csv"
        self.stockouts_path = Path(stockouts_path) if stockouts_path else self.data_dir / "stockouts.csv"
        self.suppliers_path = Path(suppliers_path) if suppliers_path else self.data_dir / "suppliers.csv"
        self.training_summary: dict[str, object] = {}
        self.validation_summary: dict[str, object] = {}
        self.max_rows = max_rows if max_rows is not None else int(__import__("os").getenv("APP_MAX_ROWS", "200"))

    def _has_external_sources(self) -> bool:
        external_roots = [Path(r'd:\download\Systeme electric'), Path(r'd:\download\IEK')]
        return any(root.exists() for root in external_roots)

    def _read_csv(self, path: Path) -> pd.DataFrame:
        external_roots = [Path(r'd:\download\Systeme electric'), Path(r'd:\download\IEK')]
        needs = {
            self.sales_path.name: load_external_sales,
            self.inventory_path.name: load_external_inventory,
            self.inbound_path.name: load_external_inbound,
            self.stockouts_path.name: load_external_stockouts,
            self.suppliers_path.name: load_external_suppliers,
        }

        loader = needs.get(path.name)
        if self._has_external_sources() and loader is not None:
            try:
                df = loader()
                if self.max_rows and len(df) > self.max_rows:
                    return df.head(self.max_rows).copy()
                return df
            except (FileNotFoundError, MemoryError, OSError, ValueError):
                pass

        if path.exists():
            if self.max_rows:
                return pd.read_csv(path, nrows=self.max_rows)
            return pd.read_csv(path)

        if loader is not None:
            for root in external_roots:
                if root.exists():
                    try:
                        df = loader()
                        if self.max_rows and len(df) > self.max_rows:
                            return df.head(self.max_rows).copy()
                        return df
                    except (FileNotFoundError, MemoryError, OSError, ValueError):
                        continue
        raise FileNotFoundError(f"Expected data file does not exist: {path}")

    def fit(self) -> dict[str, object]:
        sales = self._read_csv(self.sales_path)
        inventory = self._read_csv(self.inventory_path)
        inbound = self._read_csv(self.inbound_path)
        stockouts = self._read_csv(self.stockouts_path)
        suppliers = self._read_csv(self.suppliers_path)

        self.training_summary = {
            "sales_rows": int(len(sales)),
            "inventory_rows": int(len(inventory)),
            "inbound_rows": int(len(inbound)),
            "stockout_rows": int(len(stockouts)),
            "supplier_rows": int(len(suppliers)),
            "articles": sorted(set(sales["article"].tolist() + inventory["article"].tolist() + suppliers["article"].tolist())),
        }
        return self.training_summary

    def validate(self, train_ratio: float = 0.8) -> dict[str, object]:
        sales = self._read_csv(self.sales_path)
        inventory = self._read_csv(self.inventory_path)
        inbound = self._read_csv(self.inbound_path)
        stockouts = self._read_csv(self.stockouts_path)
        suppliers = self._read_csv(self.suppliers_path)

        if "date" not in sales.columns:
            raise ValueError("sales data must contain a 'date' column for train/validation split")

        train, valid = split_by_time(sales, date_col="date", train_ratio=train_ratio)
        metrics = evaluate_forecast_quality(train, valid, inventory, inbound, stockouts, suppliers)
        self.validation_summary = {
            "train_rows": int(len(train)),
            "validation_rows": int(metrics["validation_rows"]),
            "train_ratio": float(train_ratio),
            "actual_total_qty": float(metrics["actual_total_qty"]),
            "forecast_total_qty": float(metrics["forecast_total_qty"]),
            "mae": float(metrics["mae"]),
            "mape": float(metrics["mape"]),
        }
        return self.validation_summary

    def calculate(self) -> list[dict[str, object]]:
        sales = self._read_csv(self.sales_path)
        inventory = self._read_csv(self.inventory_path)
        inbound = self._read_csv(self.inbound_path)
        stockouts = self._read_csv(self.stockouts_path)
        suppliers = self._read_csv(self.suppliers_path)

        recommendations = build_recommendation_table(sales, inventory, inbound, stockouts, suppliers)
        return recommendations.to_dict(orient="records")
