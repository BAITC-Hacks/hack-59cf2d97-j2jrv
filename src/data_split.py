from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATE_COLUMNS = [
    "date",
    "dt",
    "datetime",
    "sales_date",
    "created_at",
    "timestamp",
    "order_date",
]


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    raise ValueError(f"Unsupported file type: {p.name}. Use .csv, .xlsx or .xls.")


def choose_date_column(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Date column '{explicit}' not found in columns: {list(df.columns)}")
        return explicit

    for candidate in DEFAULT_DATE_COLUMNS:
        if candidate in df.columns:
            return candidate

    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col

    raise ValueError(
        "Could not find a date column. Please provide --date-col manually. "
        f"Available columns: {list(df.columns)}"
    )


def split_by_time(
    df: pd.DataFrame,
    date_col: str,
    train_ratio: float = 0.8,
    min_rows: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    if len(work) < min_rows:
        raise ValueError(f"Not enough rows for split: got {len(work)}, need at least {min_rows}.")

    split_idx = max(1, min(len(work) - 1, int(len(work) * train_ratio)))
    train = work.iloc[:split_idx].copy()
    valid = work.iloc[split_idx:].copy()
    return train, valid


def save_split(train: pd.DataFrame, valid: pd.DataFrame, input_path: str | Path, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = Path(input_path).stem
    train_path = output_path / f"train_{stem}.csv"
    valid_path = output_path / f"valid_{stem}.csv"

    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)

    print(f"Saved train split: {train_path} ({len(train)} rows)")
    print(f"Saved validation split: {valid_path} ({len(valid)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split time-series data into train and validation chunks.")
    parser.add_argument("files", nargs="+", help="CSV, XLSX or XLS files to split.")
    parser.add_argument("--date-col", default=None, help="Explicit date column name, if automatic detection is insufficient.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Share of data used for training. Default: 0.8")
    parser.add_argument("--output-dir", default="data/split", help="Directory for train/validation outputs. Default: data/split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for file_path in args.files:
        df = read_table(file_path)
        date_col = choose_date_column(df, args.date_col)
        train, valid = split_by_time(df, date_col=date_col, train_ratio=args.train_ratio)
        save_split(train, valid, file_path, args.output_dir)


if __name__ == "__main__":
    main()
