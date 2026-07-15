import pandas as pd


def snake_case_columns(df: pd.DataFrame):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )

    return df


def convert_boolean(value):
    if value in [True, 1, "True", "true"]:
        return True

    if value in [False, 0, "False", "false"]:
        return False

    return None


def safe_int(value):
    if pd.isna(value):
        return None

    return int(value)


def safe_float(value):
    if pd.isna(value):
        return None

    return float(value)