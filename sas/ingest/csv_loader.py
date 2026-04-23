"""CSV → pandas DataFrame with column normalization and type coercion.

Column name normalization: spaces and non-alphanumerics → underscores, lowercase.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.csv as pv


_BOOL_COLS = [
    "public_exploit",
    "package_in_use",
    "risk_accepted",
    "cisa_kev_known_ransomware",
    "fix_available",
]

_DATE_COLS = [
    "disclosure_date",
    "fix_available_date",
    "cisa_kev_publish_date",
    "cisa_kev_due_date",
]


def _normalize(col: str) -> str:
    s = col.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def load_csv(path: Path) -> pd.DataFrame:
    """Read a Sysdig vuln CSV into a DataFrame with normalized columns and types."""
    # pyarrow is much faster than pandas for large CSVs; convert to pandas after.
    table = pv.read_csv(str(path))
    df = table.to_pandas()
    df.columns = [_normalize(c) for c in df.columns]

    for c in _BOOL_COLS:
        if c in df.columns:
            df[c] = df[c].map(_coerce_bool)

    for c in _DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    return df


def _coerce_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    # Handle pandas NA / NaN / None
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "t")
