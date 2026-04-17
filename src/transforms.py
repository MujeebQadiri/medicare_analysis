"""
transforms.py
Reshaping and transformation utilities for Part D wide-format data.
"""

import pandas as pd
from src.data_loader import YEARS, apply_outlier_filter


def melt_metric(df: pd.DataFrame, metric_prefix: str, value_name: str) -> pd.DataFrame:
    """
    Pivot a wide-format Part D dataframe to long format for a single metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (wide format with columns like Tot_Spndng_2019 ... 2023).
    metric_prefix : str
        Column prefix to melt, e.g. 'Tot_Spndng', 'Tot_Clms', 'Tot_Benes'.
    value_name : str
        Name for the resulting value column.

    Returns
    -------
    pd.DataFrame with columns: Brnd_Name, Mftr_Name, Year, <value_name>
    """
    id_cols = ["Brnd_Name", "Gnrc_Name", "Mftr_Name"]
    metric_cols = [f"{metric_prefix}_{yr}" for yr in YEARS]
    available = [c for c in metric_cols if c in df.columns]

    melted = df[id_cols + available].melt(
        id_vars=id_cols,
        value_vars=available,
        var_name="Year",
        value_name=value_name,
    )
    melted["Year"] = melted["Year"].str.extract(r"(\d{4})").astype(int)
    return melted.dropna(subset=[value_name])


def build_time_series(
    df: pd.DataFrame,
    drug: str,
    metrics: list[str],
    exclude_outliers: bool = True,
) -> pd.DataFrame:
    """
    Build a year-indexed time series for a single drug across multiple metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Overall-level dataframe.
    drug : str
        Brand name to filter on.
    metrics : list[str]
        Column prefixes to include, e.g. ['Tot_Spndng', 'Tot_Clms', 'Tot_Benes'].
    exclude_outliers : bool

    Returns
    -------
    pd.DataFrame indexed by Year with one column per metric.
    """
    drug_df = df[df["Brnd_Name"] == drug]
    rows = []
    for yr in YEARS:
        sub = apply_outlier_filter(drug_df, yr, exclude_outliers)
        if sub.empty:
            continue
        row = {"Year": yr}
        for m in metrics:
            col = f"{m}_{yr}"
            row[m] = sub[col].sum() if col in sub.columns else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("Year")


def top_n_by_metric(
    df: pd.DataFrame,
    metric_col: str,
    n: int = 20,
    exclude_outliers: bool = True,
    year: int = 2023,
) -> pd.DataFrame:
    """
    Return top N drugs by a given 2023 metric, with outlier filtering.
    """
    filtered = apply_outlier_filter(df.copy(), year, exclude_outliers)
    return (
        filtered.dropna(subset=[metric_col])
        .nlargest(n, metric_col)[["Brnd_Name", "Gnrc_Name", "Mftr_Name", metric_col]]
        .reset_index(drop=True)
    )
