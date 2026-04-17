"""
metrics.py
Reusable metric calculations for Part D spending analysis.
"""

import pandas as pd
import numpy as np
from src.data_loader import YEARS, apply_outlier_filter


def fills_per_beneficiary(df: pd.DataFrame, year: int = 2023) -> pd.Series:
    """
    Calculate fills per beneficiary for a given year.
    Higher values indicate chronic/maintenance drugs; lower values suggest acute use.
    """
    clms = df[f"Tot_Clms_{year}"]
    benes = df[f"Tot_Benes_{year}"]
    return (clms / benes).replace([np.inf, -np.inf], np.nan)


def yoy_spend_change(df: pd.DataFrame, year_from: int, year_to: int) -> pd.Series:
    """
    Calculate year-over-year percent change in total spending between two years.
    """
    prior = df[f"Tot_Spndng_{year_from}"]
    current = df[f"Tot_Spndng_{year_to}"]
    return ((current - prior) / prior).replace([np.inf, -np.inf], np.nan)


def total_program_spend(df_overall: pd.DataFrame, exclude_outliers: bool = True) -> pd.DataFrame:
    """
    Aggregate total Medicare Part D program spend per year across all drugs.

    Returns
    -------
    pd.DataFrame with columns: Year, Total_Spending
    """
    rows = []
    for yr in YEARS:
        sub = apply_outlier_filter(df_overall, yr, exclude_outliers)
        col = f"Tot_Spndng_{yr}"
        if col in sub.columns:
            rows.append({"Year": yr, "Total_Spending": sub[col].sum()})
    return pd.DataFrame(rows)


def manufacturer_market_share(df_mftr: pd.DataFrame, drug: str, year: int = 2023) -> pd.DataFrame:
    """
    Calculate each manufacturer's share of total spending for a given drug and year.

    Returns
    -------
    pd.DataFrame with columns: Mftr_Name, Spending, Share_Pct
    """
    sub = df_mftr[df_mftr["Brnd_Name"] == drug][["Mftr_Name", f"Tot_Spndng_{year}"]].copy()
    sub = sub.rename(columns={f"Tot_Spndng_{year}": "Spending"}).dropna()
    total = sub["Spending"].sum()
    sub["Share_Pct"] = (sub["Spending"] / total * 100).round(1)
    return sub.sort_values("Spending", ascending=False).reset_index(drop=True)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize outlier flag counts per year across the dataset.

    Useful for understanding how many records are excluded when filtering.
    """
    rows = []
    for yr in YEARS:
        flag_col = f"Outlier_Flag_{yr}"
        if flag_col in df.columns:
            total = df[flag_col].notna().sum()
            flagged = (df[flag_col] == 1).sum()
            rows.append({
                "Year": yr,
                "Total_Records": total,
                "Outlier_Records": flagged,
                "Outlier_Pct": round(flagged / total * 100, 2) if total else 0,
            })
    return pd.DataFrame(rows)
