# Medicare Part D Drug Spending Dashboard

An interactive dashboard for exploring U.S. Medicare Part D prescription drug spending data published by the Centers for Medicare & Medicaid Services (CMS), covering **2019–2023**.

Built with **Streamlit** and **Plotly**, the dashboard is aimed at anyone curious about how the U.S. government spends money on prescription drugs — no healthcare background required.

---

## What is Medicare Part D?

Medicare is the U.S. federal health insurance program for people aged 65 and older, as well as some younger people with disabilities. **Part D** is the prescription drug benefit — it covers the cost of medications for enrolled beneficiaries. The government pays a significant portion of these costs, and CMS publishes detailed spending data annually.

---

## Running the Dashboard

```bash
# Install dependencies
pip install -e .

# Launch the app
streamlit run dashboard/app.py
```

---

## Dashboard Pages

### 📊 Spending Dashboard

The main analytical view. Use the sidebar to select a drug, adjust filters, and explore the market.

| Section | What it shows |
|---|---|
| **KPI Cards** | Total spending, claims, beneficiaries, and average spend per claim for the selected drug in 2023, with year-over-year change vs. 2022 |
| **Year-over-Year Trend** | Line charts showing how total spending, claims, and beneficiaries evolved for the selected drug from 2019–2023 |
| **Spending Trend by Manufacturer** | Compares spending across different manufacturers of the same drug over time |
| **Claims vs. Beneficiaries Bubble Chart** | Scatter plot of the top 60 drugs — bubble size represents spending, color represents how many prescriptions each beneficiary fills per year |
| **Price Change % (2022 → 2023)** | Side-by-side bar charts showing which drugs saw the largest percentage price increases and decreases per dosage unit |
| **Spending Change $ (2022 → 2023)** | Same comparison but in raw dollar terms — useful for seeing which drugs had the biggest absolute impact on program costs |
| **5-Year CAGR vs. 2023 Spending Quadrant** | Scatter plot dividing drugs into four quadrants: high/low spending crossed with high/low growth rate |
| **Total Part D Program Spend** | Bar chart of total program spending across all drugs, by year |
| **Top N Drugs Ranking** | Horizontal bar chart of the top drugs ranked by spending, claims, or beneficiaries |

**Sidebar controls:**
- **Exclude Outlier Records** — CMS flags certain records as statistical outliers; toggle this to include or exclude them
- **Drug Selection** — choose any drug in the dataset to focus all single-drug charts on it
- **Manufacturer View** — filter which manufacturers appear in the manufacturer trend chart
- **Top N / Rank By** — adjust the leaderboard at the bottom of the page

---

### 🧬 Browse by Therapy

Explore drugs grouped by their therapeutic category rather than by individual drug name. Useful for understanding which disease areas drive the most spending.

- **ATC (Mechanism)** — groups drugs by how they work in the body, organized by organ system (e.g., cardiovascular, nervous system)
- **MeSH (Disease / Condition)** — groups drugs by the condition they treat (e.g., diabetes, cancer)

For each selection you get KPI cards, a spending breakdown by class, and individual drug cards.

---

### 💊 Drug Information

A searchable reference of every drug in the dataset, showing brand name, generic name, and a plain-language description of what the drug is used for. Search by drug name or condition.

---

## Glossary of Terms

| Term | Definition |
|---|---|
| **Medicare Part D** | The part of the U.S. Medicare program that covers prescription drug costs for enrolled seniors and qualifying individuals |
| **CMS** | Centers for Medicare & Medicaid Services — the federal agency that administers Medicare and publishes the spending data used here |
| **Beneficiary** | A person enrolled in Medicare who is eligible to receive prescription drug benefits |
| **Claim** | A single prescription fill submitted to Medicare for reimbursement; one beneficiary can have multiple claims per year |
| **Total Spending** | The total amount Medicare Part D paid for a drug across all claims in a given year |
| **Avg Spend Per Claim** | Total spending divided by total claims — roughly the cost Medicare pays each time a prescription is filled |
| **Fills per Beneficiary** | Total claims divided by total beneficiaries — how many times, on average, each patient filled a prescription for that drug in a year |
| **CAGR** | **Compound Annual Growth Rate** — the average yearly growth rate of a value over multiple years, expressed as a percentage. A CAGR of 10% means the value grew by about 10% per year on average, compounding. Formula: `(End Value / Start Value)^(1/n) − 1`, where n is the number of years |
| **5-Year CAGR (2019–2023)** | The annualized growth rate of the average spend per dosage unit from 2019 to 2023 |
| **Dosage Unit** | A single unit of a drug as defined by its form — one tablet, one milliliter of liquid, one patch, etc. Prices are often compared per dosage unit to make different package sizes comparable |
| **YoY (Year-over-Year)** | A comparison of a metric in one year versus the same metric in the prior year, expressed as a percentage change |
| **Outlier Flag** | A CMS-assigned flag marking records that may distort analysis — for example, a drug with very few claims in a given year. The dashboard lets you toggle these out |
| **Brand Name** | The commercial name a pharmaceutical company gives to a drug (e.g., Ozempic, Humira) |
| **Generic Name** | The non-proprietary scientific name for the active ingredient (e.g., semaglutide, adalimumab). Generic drugs contain the same active ingredient as the brand-name version |
| **Manufacturer** | The pharmaceutical company that produces and sells a given drug |
| **Market Share** | A manufacturer's portion of total spending for a drug, expressed as a percentage |
| **ATC Classification** | **Anatomical Therapeutic Chemical** — an international system that classifies drugs by the organ system they affect and their mechanism of action |
| **MeSH** | **Medical Subject Headings** — a controlled vocabulary used by the U.S. National Library of Medicine to classify biomedical topics, including diseases and drug uses |
| **RxNorm / RxClass** | Standardized drug naming and classification systems maintained by the U.S. National Library of Medicine, used here to group drugs into therapeutic categories |

---

## Data Source

All spending data comes from the **CMS Medicare Part D Drug Spending Dashboard** public dataset, covering calendar years 2019–2023. Drug information and therapeutic classifications are enriched from RxNorm/RxClass via the National Library of Medicine.
