"""Unified output schema for the Clin-JEPA data pipeline (Steps 02-03).

Defines the canonical per-event column schema used by observation extraction,
action extraction, and timeline building. Every per-stay parquet file must
conform to this schema.
"""

import pandas as pd

# Canonical column order for per-stay parquet files
OUTPUT_COLUMNS = [
    "stay_id",
    "timestamp",
    "timestamp_hours",
    "domain",
    "source_table",
    "category",
    "variable_name",
    "numeric_value",
    "text_value",
    "unit",
    "end_timestamp",
    "end_timestamp_hours",
    "is_continuous",
]

OUTPUT_DTYPES = {
    "stay_id": "int64",
    "timestamp": "datetime64[ns]",
    "timestamp_hours": "float64",
    "domain": "object",
    "source_table": "object",
    "category": "object",
    "variable_name": "object",
    "numeric_value": "float64",
    "text_value": "object",
    "unit": "object",
    "end_timestamp": "datetime64[ns]",
    "end_timestamp_hours": "float64",
    "is_continuous": "bool",
}


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that a DataFrame matches the output schema.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If columns don't match the expected schema.
    """
    missing = set(OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    extra = set(df.columns) - set(OUTPUT_COLUMNS)
    if extra:
        raise ValueError(f"Unexpected columns: {extra}")


def empty_events_df() -> pd.DataFrame:
    """Return an empty DataFrame with the correct output schema."""
    return pd.DataFrame({col: pd.Series(dtype=OUTPUT_DTYPES[col]) for col in OUTPUT_COLUMNS})
