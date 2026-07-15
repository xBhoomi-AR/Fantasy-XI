import pandas as pd


def validate_dataframe(df: pd.DataFrame):
    if df is None:
        raise ValueError("DataFrame is None")

    if df.empty:
        raise ValueError("DataFrame is empty")


def validate_columns(df: pd.DataFrame, required_columns: list):
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def validate_nulls(df: pd.DataFrame, required_columns: list):
    null_columns = [
        column
        for column in required_columns
        if df[column].isnull().any()
    ]

    if null_columns:
        raise ValueError(
            f"Null values found in: {null_columns}"
        )


def validate_duplicates(df: pd.DataFrame, subset: list):
    if df.duplicated(subset=subset).any():
        raise ValueError(
            "Duplicate records found."
        )