from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reorder_optimizer import build_recommendation_table


class ReorderAgent:
    def __init__(
        self,
        data_dir: str | Path = "data",
        sales_path: str | Path | None = None,
        inventory_path: str | Path | None = None,
        inbound_path: str | Path | None = None,
        stockouts_path: str | Path | None = None,
        suppliers_path: str | Path | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sales_path = Path(sales_path) if sales_path else self.data_dir / "sample_sales.csv"
        self.inventory_path = Path(inventory_path) if inventory_path else self.data_dir / "inventory.csv"
        self.inbound_path = Path(inbound_path) if inbound_path else self.data_dir / "inbound.csv"
        self.stockouts_path = Path(stockouts_path) if stockouts_path else self.data_dir / "stockouts.csv"
        self.suppliers_path = Path(suppliers_path) if suppliers_path else self.data_dir / "suppliers.csv"
        self.training_summary: dict[str, object] = {}

    def _read_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Expected data file does not exist: {path}")
        return pd.read_csv(path)

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

    def calculate(self) -> list[dict[str, object]]:
        sales = self._read_csv(self.sales_path)
        inventory = self._read_csv(self.inventory_path)
        inbound = self._read_csv(self.inbound_path)
        stockouts = self._read_csv(self.stockouts_path)
        suppliers = self._read_csv(self.suppliers_path)

        recommendations = build_recommendation_table(sales, inventory, inbound, stockouts, suppliers)
        return recommendations.to_dict(orient="records")
