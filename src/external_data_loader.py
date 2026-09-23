from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MONTH_MAP = {
    'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'май': 5, 'июн': 6,
    'июл': 7, 'авг': 8, 'сен': 9, 'окт': 10, 'ноя': 11, 'дек': 12,
}


def _normalize_article(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none'}:
        return ""
    text = text.replace('"', '').replace("'", '')
    return re.sub(r'\s+', ' ', text).strip()


def _is_real_article(value: object) -> bool:
    text = str(value).strip()
    if not text or text.lower() in {'nan', 'none', 'itogo', 'итого', 'total', 'totals'}:
        return False
    return True


def _load_excel_file(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def _iter_external_roots() -> list[Path]:
    roots = [
        Path(r'd:\download\Systeme electric'),
        Path(r'd:\download\IEK'),
    ]
    return [root for root in roots if root.exists()]


def _choose_files_by_keyword(keyword_fragments: list[str]) -> list[Path]:
    matches: list[Path] = []
    for root in _iter_external_roots():
        for path in root.glob('*.xlsx'):
            name = path.name.lower()
            if all(fragment.lower() in name for fragment in keyword_fragments):
                matches.append(path)
    return matches


def _extract_month_columns(columns: list[object]) -> list[object]:
    month_cols: list[object] = []
    for col in columns:
        text = str(col).lower()
        if any(token in text for token in MONTH_MAP):
            month_cols.append(col)
    return month_cols


def load_external_sales() -> pd.DataFrame:
    files = _choose_files_by_keyword(['продаж']) + _choose_files_by_keyword(['sales'])
    if not files:
        raise FileNotFoundError('No external sales files found')

    frames: list[pd.DataFrame] = []
    for file in files:
        df = _load_excel_file(file)
        if df.empty:
            continue
        normalized = df.copy()
        normalized.columns = [str(col) for col in normalized.columns]
        article_col = next((c for c in normalized.columns if 'номенклатура' in c.lower() or 'наименование' in c.lower() or 'товар' in c.lower()), None)
        if article_col is None:
            continue
        month_cols = _extract_month_columns(list(normalized.columns))
        if not month_cols:
            continue
        work = normalized[[article_col] + month_cols].copy()
        work.columns = ['article'] + [str(col) for col in month_cols]
        work = work.dropna(subset=['article'])
        melted = work.melt(id_vars=['article'], var_name='month', value_name='qty')
        melted = melted[melted['qty'].notna()].copy()
        melted['article'] = melted['article'].map(_normalize_article)
        melted = melted[melted['article'].map(_is_real_article)]
        if melted.empty:
            continue

        def parse_month_label(label: str) -> pd.Timestamp:
            text = label.lower()
            for token, month_no in MONTH_MAP.items():
                if token in text:
                    year_match = re.search(r'(20\d{2})', label)
                    year = int(year_match.group(1)) if year_match else 2024
                    return pd.Timestamp(year=year, month=month_no, day=1)
            return pd.Timestamp('2000-01-01')

        melted['date'] = melted['month'].map(parse_month_label)
        melted['qty'] = pd.to_numeric(melted['qty'], errors='coerce').abs()
        melted['client_id'] = 'external'
        melted = melted.dropna(subset=['date', 'qty'])
        frames.append(melted[['date', 'article', 'client_id', 'qty']])

    if not frames:
        return pd.DataFrame(columns=['date', 'article', 'client_id', 'qty'])
    return pd.concat(frames, ignore_index=True)


def load_external_inventory() -> pd.DataFrame:
    files = _choose_files_by_keyword(['остатк'])
    if not files:
        raise FileNotFoundError('No external inventory files found')

    frames: list[pd.DataFrame] = []
    for file in files:
        df = _load_excel_file(file)
        if df.empty:
            continue
        normalized = df.copy()
        normalized.columns = [str(col) for col in normalized.columns]
        article_col = next((c for c in normalized.columns if 'номенклатура' in c.lower() or 'наименование' in c.lower()), None)
        if article_col is None:
            continue
        month_cols = _extract_month_columns(list(normalized.columns))
        if not month_cols:
            continue
        work = normalized[[article_col] + month_cols].copy()
        work.columns = ['article'] + [str(col) for col in month_cols]
        work = work.dropna(subset=['article'])
        melted = work.melt(id_vars=['article'], var_name='month', value_name='quantity_on_hand')
        melted = melted[melted['quantity_on_hand'].notna()].copy()
        melted['article'] = melted['article'].map(_normalize_article)
        melted = melted[melted['article'].map(_is_real_article)]
        melted['quantity_on_hand'] = pd.to_numeric(melted['quantity_on_hand'], errors='coerce')
        melted = melted.dropna(subset=['quantity_on_hand'])
        if melted.empty:
            continue
        def month_sort_key(label: str) -> tuple[int, int]:
            text = label.lower()
            for token, month_no in MONTH_MAP.items():
                if token in text:
                    year_match = re.search(r'(20\d{2})', label)
                    year = int(year_match.group(1)) if year_match else 2024
                    return (year, month_no)
            return (0, 0)

        melted['__order'] = melted['month'].map(month_sort_key)
        melted = melted.sort_values('__order').drop(columns=['__order'])
        latest_by_article = melted.drop_duplicates(subset=['article'], keep='last')
        frames.append(latest_by_article[['article', 'quantity_on_hand']])

    if not frames:
        return pd.DataFrame(columns=['article', 'quantity_on_hand'])
    return pd.concat(frames, ignore_index=True).groupby('article', as_index=False)['quantity_on_hand'].last()


def load_external_inbound() -> pd.DataFrame:
    files = _choose_files_by_keyword(['путь']) + _choose_files_by_keyword(['inbound'])
    if not files:
        return pd.DataFrame(columns=['article', 'expected_qty'])

    frames: list[pd.DataFrame] = []
    for file in files:
        df = _load_excel_file(file)
        if df.empty:
            continue
        normalized = df.copy()
        normalized.columns = [str(col) for col in normalized.columns]
        article_col = next((c for c in normalized.columns if 'артикул' in c.lower() or 'код' in c.lower() or 'код 1с' in c.lower()), None)
        if article_col is None:
            continue
        number_cols = [
            c for c in normalized.columns
            if c not in {article_col, next((candidate for candidate in normalized.columns if 'наименование' in candidate.lower()), '')}
            and pd.api.types.is_numeric_dtype(normalized[c])
        ]
        if not number_cols:
            number_cols = [c for c in normalized.columns if c not in {article_col} and normalized[c].map(lambda v: pd.notna(v) and str(v).replace('.', '').replace(',', '').replace(' ', '').isdigit()).all()]
        work = normalized[[article_col] + number_cols].copy()
        work = work.rename(columns={article_col: 'article'})
        work['article'] = work['article'].map(_normalize_article)
        work = work[work['article'].map(_is_real_article)]
        if work.empty:
            continue
        values = work.drop(columns=['article']).apply(pd.to_numeric, errors='coerce')
        work['expected_qty'] = values.sum(axis=1, numeric_only=True).fillna(0.0)
        frames.append(work[['article', 'expected_qty']])

    if not frames:
        return pd.DataFrame(columns=['article', 'expected_qty'])
    result = pd.concat(frames, ignore_index=True).groupby('article', as_index=False)['expected_qty'].sum()
    result['expected_qty'] = pd.to_numeric(result['expected_qty'], errors='coerce').fillna(0.0)
    return result


def load_external_suppliers() -> pd.DataFrame:
    files = _choose_files_by_keyword(['moq'])
    if not files:
        return pd.DataFrame(columns=['article', 'supplier', 'lead_time_days', 'min_order_qty'])

    rows: list[dict] = []
    for file in files:
        df = _load_excel_file(file)
        if df.empty:
            continue
        normalized = df.copy()
        normalized.columns = [str(col) for col in normalized.columns]
        article_col = next((c for c in normalized.columns if 'артикул' in c.lower() or 'код' in c.lower() or 'код 1с' in c.lower()), None)
        min_order_col = next((c for c in normalized.columns if 'мин' in c.lower() or 'кратность' in c.lower() or 'отгр' in c.lower()), None)
        if article_col is None or min_order_col is None:
            continue
        work = normalized[[article_col, min_order_col]].copy()
        work.columns = ['article', 'min_order_qty']
        work['article'] = work['article'].map(_normalize_article)
        work['min_order_qty'] = pd.to_numeric(work['min_order_qty'], errors='coerce').fillna(1)
        work = work[work['article'].map(_is_real_article)]
        supplier_name = file.parent.name
        rows.extend({
            'article': row['article'],
            'supplier': supplier_name,
            'lead_time_days': 21,
            'min_order_qty': float(row['min_order_qty']),
        } for _, row in work.iterrows())

    if not rows:
        return pd.DataFrame(columns=['article', 'supplier', 'lead_time_days', 'min_order_qty'])
    return pd.DataFrame(rows).drop_duplicates(subset=['article', 'supplier']).reset_index(drop=True)


def load_external_stockouts() -> pd.DataFrame:
    return pd.DataFrame(columns=['article', 'stockout_days', 'lost_sales_units'])
