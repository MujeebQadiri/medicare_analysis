"""
data_loader.py
Handles fetching, caching, and splitting of CMS Medicare Part D data.
"""

import pathlib
import requests
import pandas as pd

CMS_ENDPOINT = "https://data.cms.gov/data-api/v1/dataset/7e0b4365-fd63-4a29-8f5e-e0ac9f66a81b/data"

# Anchor to repo root (one level up from src/) so the cache path is always
# <repo_root>/data/medicare_part_d_spending.csv regardless of where the
# calling script or notebook is run from.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CACHE = _REPO_ROOT / "data" / "medicare_part_d_spending.csv"

STRING_COLS = ["Brnd_Name", "Gnrc_Name", "Mftr_Name"]
YEARS = [2019, 2020, 2021, 2022, 2023]


def fetch_partd_data(cache_path: pathlib.Path = DEFAULT_CACHE, page_size: int = 1000) -> pd.DataFrame:
    """
    Load CMS Part D data from local cache if available, otherwise fetch from API.

    Parameters
    ----------
    cache_path : Path
        Location to read/write the cached CSV.
    page_size : int
        Number of records per API request (max 1000).

    Returns
    -------
    pd.DataFrame
        Raw dataframe with all records (both Overall and manufacturer-level).
    """
    if cache_path.exists():
        df = pd.read_csv(cache_path)
    else:
        all_data, offset = [], 0
        while True:
            batch = requests.get(CMS_ENDPOINT, params={"size": page_size, "offset": offset}).json()
            if not batch:
                break
            all_data.extend(batch)
            offset += page_size
            print(f"Fetched {len(all_data)} records...")
        df = pd.DataFrame(all_data)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)

    return _cast_types(df)


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast all non-string columns to numeric."""
    num_cols = df.columns.difference(STRING_COLS)
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df


def split_overall_mftr(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into Overall (aggregated) and manufacturer-level records.

    Use df_overall for most analyses to avoid double-counting.
    Use df_mftr for manufacturer market share comparisons.

    Returns
    -------
    df_overall : pd.DataFrame
    df_mftr : pd.DataFrame
    """
    df_overall = df[df["Mftr_Name"] == "Overall"].copy()
    df_mftr = df[df["Mftr_Name"] != "Overall"].copy()
    return df_overall, df_mftr


def apply_outlier_filter(df: pd.DataFrame, year: int, exclude: bool = True) -> pd.DataFrame:
    """
    Optionally remove records where Outlier_Flag_{year} == 1.

    Parameters
    ----------
    df : pd.DataFrame
    year : int
        The data year to check the outlier flag for.
    exclude : bool
        If True, drops flagged records.
    """
    if not exclude:
        return df
    flag_col = f"Outlier_Flag_{year}"
    if flag_col in df.columns:
        return df[df[flag_col] != 1]
    return df

def _get_repo_root() -> pathlib.Path:
    """
    Resolve repo root reliably from both .py scripts and Jupyter notebooks.
    Walks up from cwd until it finds pyproject.toml or .git as a root marker.
    """
    current = pathlib.Path().resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current  # fallback to cwd if no marker found

def get_rxcui(brand_name: str) -> str | None:
    """
    Resolve a drug brand name to its RxNorm Concept Unique Identifier (RxCUI).

    Uses the NLM RxNorm API (https://rxnav.nlm.nih.gov). No API key required.

    Parameters
    ----------
    brand_name : str
        Brand or generic drug name as it appears in the CMS dataset
        (e.g. "Ozempic", "Victoza 2-Pak"). Packaging suffixes like
        "2-Pak" may reduce match rate — strip them if coverage is low.

    Returns
    -------
    str or None
        The first RxCUI returned by the API, or None if no match found.
    """
    r = requests.get(
        "https://rxnav.nlm.nih.gov/REST/rxcui.json",
        params={"name": brand_name, "search": 1},
    )
    ids = r.json().get("idGroup", {}).get("rxnormId", [])
    return ids[0] if ids else None


def get_drug_classes(rxcui: str, rela_source: str = "ATC") -> list[dict]:
    """
    Return therapeutic class memberships for a drug given its RxCUI.

    Uses the NLM RxClass API (https://rxnav.nlm.nih.gov/RxClassIntro.html).
    No API key required.

    Parameters
    ----------
    rxcui : str
        RxNorm Concept Unique Identifier, obtained via get_rxcui().
    rela_source : str, optional
        Classification system to query. Default is "ATC" (Anatomical
        Therapeutic Chemical). Other options include:
        - "MESH"     : MeSH pharmacological actions
        - "FMTSME"   : FDA mechanism of action / physiologic effect
        - "VA"       : VA National Drug File therapeutic categories
        - "MEDRT"    : Medication Reference Terminology (DoD/VA)

    Returns
    -------
    list of dict
        Each dict contains:
        - "class_name" : human-readable class label (e.g. "GLP-1 receptor agonists")
        - "class_id"   : classification code (e.g. ATC code "A10BJ")
        - "class_type" : classification type string (e.g. "ATC1-4")
        Sorted by specificity (longest class_id = most specific level first).
        Returns empty list if no classes found.
    """
    r = requests.get(
        "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json",
        params={"rxcui": rxcui, "relaSource": rela_source},
    )
    concepts = r.json().get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
    classes = [
        {
            "class_name": c["rxclassMinConceptItem"]["className"],
            "class_id":   c["rxclassMinConceptItem"]["classId"],
            "class_type": c["rxclassMinConceptItem"]["classType"],
        }
        for c in concepts
    ]
    return sorted(classes, key=lambda x: len(x["class_id"]), reverse=True)