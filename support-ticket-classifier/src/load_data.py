"""Load a ticket dataset and return a standard 3-column frame.

Returned columns: raw_text, category, priority (priority is None if the dataset has none).
Known Kaggle schemas are auto-detected; for any other CSV pass the column names explicitly.
"""
from typing import List, Optional

import pandas as pd

KNOWN_SCHEMAS = [
    {   # kaggle: suraj520/customer-support-ticket-dataset
        "name": "Kaggle - Customer Support Ticket Dataset",
        "needs": ["Ticket Description", "Ticket Type"],
        "text": ["Ticket Subject", "Ticket Description"],
        "category": "Ticket Type",
        "priority": "Ticket Priority",
    },
    {   # kaggle: adisongoh/it-service-ticket-classification-dataset
        "name": "Kaggle - IT Service Ticket Classification",
        "needs": ["Document", "Topic_group"],
        "text": ["Document"],
        "category": "Topic_group",
        "priority": None,
    },
]


def load_dataset(path: str, text_cols: Optional[List[str]] = None,
                 category_col: Optional[str] = None, priority_col: Optional[str] = None):
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:  # some exported CSVs are not UTF-8
        df = pd.read_csv(path, encoding="latin-1")
    schema_name = "custom"

    if not (text_cols and category_col):
        for s in KNOWN_SCHEMAS:
            if all(c in df.columns for c in s["needs"]):
                text_cols = [c for c in s["text"] if c in df.columns]
                category_col = s["category"]
                priority_col = priority_col or (s["priority"] if s["priority"] in df.columns else None)
                schema_name = s["name"]
                break
        else:
            raise ValueError(
                "Could not auto-detect columns. Pass --text-cols and --category-col.\n"
                f"Columns found: {list(df.columns)}"
            )

    missing = [c for c in list(text_cols) + [category_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) {missing} not found. Columns in file: {list(df.columns)}")

    def _clean_label(series):
        """Strip labels and turn blanks / 'nan' / 'none' into real missing values (NaN)."""
        def fix(v):
            if pd.isna(v):
                return float("nan")
            v = str(v).strip()
            return float("nan") if v.lower() in ("", "nan", "none", "null") else v
        return series.map(fix).astype("object")

    out = pd.DataFrame({
        "raw_text": df[text_cols].fillna("").astype(str).agg(" . ".join, axis=1),
        "category": _clean_label(df[category_col]),
        "priority": _clean_label(df[priority_col]) if priority_col and priority_col in df.columns else None,
    })
    out = out[out["raw_text"].str.strip() != ""]
    out = out[out["category"].notna()]
    meta = {"schema": schema_name, "text_cols": text_cols, "category_col": category_col,
            "priority_col": priority_col if priority_col in df.columns else None,
            "rows_loaded": int(len(df))}
    return out.reset_index(drop=True), meta
