"""
scripts/enrich_drug_classes.py
──────────────────────────────
Enriches the CMS Part D drug list with therapeutic classifications from the
NLM RxNorm and RxClass APIs (free, no API key required).

Produces: data/drug_classes.csv with columns:
    Brnd_Name   — original CMS brand name
    Gnrc_Name   — original CMS generic name
    RxCUI       — RxNorm concept unique identifier
    ATC_Code    — most specific ATC code (e.g. "A10BJ")
    ATC_Class   — ATC class label (e.g. "Glucagon-like peptide-1 receptor agonists")
    ATC_L1      — ATC level-1 label (e.g. "Alimentary tract and metabolism")
    MESH_Class  — MeSH pharmacologic action / disease category
    Match_Type  — how the RxCUI was resolved ("exact", "approx", or "none")

Usage
-----
    python scripts/enrich_drug_classes.py

Re-run any time you want to refresh the classifications. Already-resolved
drugs are skipped if drug_classes.csv already exists (incremental mode).

Rate limiting
-------------
The NLM API asks for no more than 20 requests/second. This script sleeps
0.1s between calls which keeps it comfortably under that limit.
~1,400 unique drugs takes roughly 10–15 minutes to enrich.
"""

import sys
import time
import pathlib
import requests
import pandas as pd
from requests.exceptions import Timeout, ConnectionError as ReqConnError

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_loader import fetch_partd_data, split_overall_mftr, _get_repo_root

RXNORM_BASE  = "https://rxnav.nlm.nih.gov/REST"
RXCLASS_BASE = "https://rxnav.nlm.nih.gov/REST/rxclass"
SLEEP_S      = 0.1   # seconds between API calls
OUTPUT_PATH  = _get_repo_root() / "data" / "drug_classes.csv"

# ATC level-1 code → human readable label
ATC_L1_MAP = {
    "A": "Alimentary tract and metabolism",
    "B": "Blood and blood forming organs",
    "C": "Cardiovascular system",
    "D": "Dermatologicals",
    "G": "Genito-urinary system and sex hormones",
    "H": "Systemic hormonal preparations",
    "J": "Anti-infectives for systemic use",
    "L": "Antineoplastic and immunomodulating agents",
    "M": "Musculo-skeletal system",
    "N": "Nervous system",
    "P": "Antiparasitic products",
    "R": "Respiratory system",
    "S": "Sensory organs",
    "V": "Various",
}


# ── API helpers ───────────────────────────────────────────────────────────────

def _get(url: str, params: dict, retries: int = 4, base_wait: float = 2.0):
    """GET with exponential-backoff retry on timeout or connection errors."""
    for attempt in range(retries):
        try:
            return requests.get(url, params=params, timeout=15)
        except (Timeout, ReqConnError) as e:
            if attempt == retries - 1:
                raise
            wait = base_wait * (2 ** attempt)
            print(f"\n  [retry {attempt+1}/{retries-1} after {wait:.0f}s — {e.__class__.__name__}]",
                  end=" ", flush=True)
            time.sleep(wait)


def get_ingredient_rxcui(rxcui: str, gnrc_name: str = "") -> str:
    """
    Return the ingredient-level RxCUI for a branded or clinical drug RxCUI.

    MEDRT may_treat relationships are indexed on ingredient concepts (tty=IN),
    not on branded products (tty=BN/SBD/GPCK). Resolving to the ingredient
    before querying MEDRT significantly improves hit rate.

    Strategy:
    1. Try /rxcui/{rxcui}/related.json?tty=IN — works for well-formed RxNorm RxCUIs.
    2. If that fails (MMSL-source RxCUIs lack the full relationship graph),
       fall back to a direct normalized search on the generic name.

    Returns the original rxcui unchanged if both attempts fail.
    """
    r = _get(f"{RXNORM_BASE}/rxcui/{rxcui}/related.json", {"tty": "IN"})
    concepts = (r.json()
                 .get("relatedGroup", {})
                 .get("conceptGroup", []))
    for group in concepts:
        props = group.get("conceptProperties", [])
        if props:
            return props[0]["rxcui"]

    # Fallback: search by generic name (normalized, ingredients only)
    if gnrc_name:
        r2 = _get(f"{RXNORM_BASE}/rxcui.json", {"name": gnrc_name, "search": 0})
        ids = r2.json().get("idGroup", {}).get("rxnormId", [])
        if ids:
            return ids[0]

    return rxcui


def get_rxcui(brand_name: str) -> tuple[str | None, str]:
    """
    Resolve a drug brand name to its RxNorm Concept Unique Identifier (RxCUI).

    Uses the NLM RxNorm API (https://rxnav.nlm.nih.gov). No API key required.

    Parameters
    ----------
    brand_name : str
        Brand or generic drug name as it appears in the CMS dataset
        (e.g. "Ozempic", "Victoza 2-Pak"). Packaging suffixes like
        "2-Pak" may reduce match rate — stripped automatically below.

    Returns
    -------
    tuple of (rxcui, match_type)
        rxcui      : RxCUI string, or None if no match found
        match_type : "exact", "approx", or "none"

    Examples
    --------
    >>> get_rxcui("Ozempic")
    ('2200280', 'exact')
    >>> get_rxcui("NotADrug")
    (None, 'none')
    """
    # Try exact match first
    r = _get(f"{RXNORM_BASE}/rxcui.json", {"name": brand_name, "search": 1})
    ids = r.json().get("idGroup", {}).get("rxnormId", [])
    if ids:
        return ids[0], "exact"

    # Strip common CMS packaging suffixes and retry
    cleaned = (brand_name
               .replace(" 2-Pak", "").replace(" 3-Pak", "")
               .replace(" 2-pak", "").replace(" 3-pak", "")
               .strip())
    if cleaned != brand_name:
        r2 = _get(f"{RXNORM_BASE}/rxcui.json", {"name": cleaned, "search": 1})
        ids2 = r2.json().get("idGroup", {}).get("rxnormId", [])
        if ids2:
            return ids2[0], "approx"

    # Fall back to approximate search
    r3 = _get(f"{RXNORM_BASE}/approximateTerm.json", {"term": cleaned, "maxEntries": 1})
    candidates = r3.json().get("approximateGroup", {}).get("candidate", [])
    if candidates:
        return candidates[0]["rxcui"], "approx"

    return None, "none"


def get_drug_classes(rxcui: str, rela_source: str = "ATC", relas: str = None) -> list[dict]:
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
        Therapeutic Chemical). Supported options:
        - "ATC"    : WHO Anatomical Therapeutic Chemical — mechanism-based
                     hierarchy (organ system → pharmacologic subgroup)
        - "MEDRT"  : VA/DoD Medication Reference Terminology — use with
                     relas="may_treat" to get disease/condition labels
        - "EPC"    : FDA Established Pharmacologic Classes — concise
                     mechanism-of-action labels (e.g. "Opioid Agonist")
        - "VA"     : VA National Drug File therapeutic categories
        - "FMTSME" : FDA mechanism of action / physiologic effect
    relas : str, optional
        Relationship filter. Only used with certain sources:
        - "may_treat" with MEDRT → returns conditions the drug treats
          (e.g. "Diabetes Mellitus", "Hypertension")
        - Leave None for ATC, EPC, VA (they don't use relas filtering)

    Returns
    -------
    list of dict
        Each dict contains:
        - "class_name" : human-readable class label
        - "class_id"   : classification code (e.g. ATC code "A10BJ")
        - "class_type" : classification type string
        Sorted by specificity — longest class_id = most specific level first.
        Returns empty list if no classes found.

    Examples
    --------
    >>> get_drug_classes("2200280")
    [{"class_name": "Glucagon-like peptide-1 receptor agonists",
      "class_id": "A10BJ", "class_type": "ATC1-4"}, ...]

    >>> get_drug_classes("2200280", rela_source="MEDRT", relas="may_treat")
    [{"class_name": "Diabetes Mellitus, Type 2",
      "class_id": "D003924", "class_type": "DISEASE"}, ...]

    >>> get_drug_classes("2200280", rela_source="EPC")
    [{"class_name": "Glucagon-Like Peptide 1 Agonist",
      "class_id": "N0000175745", "class_type": "EPC"}, ...]
    """
    params = {"rxcui": rxcui, "relaSource": rela_source}
    if relas:
        params["relas"] = relas
    r = _get(f"{RXCLASS_BASE}/class/byRxcui.json", params)
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


# ── Enrichment pipeline ───────────────────────────────────────────────────────

def enrich(df_overall: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the full CMS drug list with ATC and MeSH classifications.

    Loads existing drug_classes.csv if present and only processes
    new/missing drugs (incremental mode).
    """
    # Get unique drugs
    drugs = (df_overall[["Brnd_Name", "Gnrc_Name"]]
             .drop_duplicates(subset=["Brnd_Name"])
             .reset_index(drop=True))

    # Load existing cache if available
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)

        # Rows that already have MESH_Class filled in need no further work
        done_with_mesh = set(existing.loc[existing["MESH_Class"].notna(), "Brnd_Name"].values)
        # Rows that have a resolved RxCUI but no MESH_Class need only the MEDRT call
        needs_medrt = existing[
            existing["MESH_Class"].isna() & existing["RxCUI"].notna()
        ].copy()
        # Rows with no RxCUI at all need full re-enrichment (shouldn't happen often)
        fully_unresolved = set(existing.loc[existing["RxCUI"].isna(), "Brnd_Name"].values)

        already_fully_done = done_with_mesh - fully_unresolved
        drugs_new = drugs[~drugs["Brnd_Name"].isin(existing["Brnd_Name"])].reset_index(drop=True)

        print(
            f"Resuming — {len(already_fully_done)} complete, "
            f"{len(needs_medrt)} need MEDRT backfill, "
            f"{len(drugs_new)} new drugs"
        )
        # Start results from the already-complete rows only; we'll patch needs_medrt below
        results_by_name = {r["Brnd_Name"]: r for r in existing.to_dict("records")}
    else:
        needs_medrt = pd.DataFrame()
        drugs_new = drugs.reset_index(drop=True)
        results_by_name = {}
        print(f"Fresh run — {len(drugs_new)} drugs to enrich")

    # ── MEDRT backfill for rows that have RxCUI but no MESH_Class ────────────
    medrt_total = len(needs_medrt)
    for j, mrow in enumerate(needs_medrt.itertuples(), 1):
        rxcui = str(mrow.RxCUI).split(".")[0]  # stored as float in CSV → strip ".0"
        print(f"[MEDRT {j}/{medrt_total}] {mrow.Brnd_Name}", end=" ... ", flush=True)
        ing_rxcui = get_ingredient_rxcui(rxcui, gnrc_name=mrow.Gnrc_Name)
        time.sleep(SLEEP_S)
        medrt_classes = get_drug_classes(ing_rxcui, rela_source="MEDRT", relas="may_treat")
        time.sleep(SLEEP_S)
        if medrt_classes:
            mesh = sorted(medrt_classes, key=lambda x: len(x["class_name"]))[0]["class_name"]
            results_by_name[mrow.Brnd_Name]["MESH_Class"] = mesh
            print(mesh)
        else:
            print("n/a")

        if j % 50 == 0:
            pd.DataFrame(list(results_by_name.values())).to_csv(OUTPUT_PATH, index=False)
            print(f"  ↳ Progress saved (MEDRT backfill {j}/{medrt_total})")

    # ── Full enrichment for new drugs ─────────────────────────────────────────
    total = len(drugs_new)
    for i, row in drugs_new.iterrows():
        brand = row["Brnd_Name"]
        generic = row["Gnrc_Name"]
        print(f"[{i+1}/{total}] {brand}", end=" ... ", flush=True)

        record = {
            "Brnd_Name":  brand,
            "Gnrc_Name":  generic,
            "RxCUI":      None,
            "ATC_Code":   None,
            "ATC_Class":  None,
            "ATC_L1":     None,
            "MESH_Class": None,
            "Match_Type": "none",
        }

        # Step 1: resolve RxCUI
        rxcui, match_type = get_rxcui(brand)
        record["Match_Type"] = match_type
        time.sleep(SLEEP_S)

        if rxcui:
            record["RxCUI"] = rxcui

            # Step 2a: ATC classification
            atc_classes = get_drug_classes(rxcui, rela_source="ATC")
            time.sleep(SLEEP_S)
            if atc_classes:
                top_atc = atc_classes[0]
                record["ATC_Code"]  = top_atc["class_id"]
                record["ATC_Class"] = top_atc["class_name"]
                record["ATC_L1"]    = ATC_L1_MAP.get(top_atc["class_id"][0], "Other")

            # Step 2b: Disease/condition classification via MEDRT may_treat
            # Resolve to ingredient RxCUI first — MEDRT indexes on ingredients, not brands
            ing_rxcui = get_ingredient_rxcui(rxcui, gnrc_name=generic)
            time.sleep(SLEEP_S)
            medrt_classes = get_drug_classes(ing_rxcui, rela_source="MEDRT", relas="may_treat")
            time.sleep(SLEEP_S)
            if medrt_classes:
                record["MESH_Class"] = sorted(
                    medrt_classes, key=lambda x: len(x["class_name"])
                )[0]["class_name"]

        print(f"{match_type} → ATC: {record['ATC_Class'] or 'n/a'} | MeSH: {record['MESH_Class'] or 'n/a'}")
        results_by_name[brand] = record

        # Save incrementally every 50 drugs
        if (i + 1) % 50 == 0:
            pd.DataFrame(list(results_by_name.values())).to_csv(OUTPUT_PATH, index=False)
            print(f"  ↳ Progress saved ({i+1}/{total})")

    results = list(results_by_name.values())

    # Final save
    out_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✓ Done. {len(out_df)} drugs enriched → {OUTPUT_PATH}")
    print(f"  ATC coverage:  {out_df['ATC_Class'].notna().mean():.1%}")
    print(f"  MeSH coverage: {out_df['MESH_Class'].notna().mean():.1%}")
    print(f"  RxCUI resolved: {(out_df['Match_Type'] != 'none').mean():.1%}")
    return out_df


if __name__ == "__main__":
    print("Loading CMS Part D data...")
    df = fetch_partd_data()
    df_overall, _ = split_overall_mftr(df)
    enrich(df_overall)
