from pathlib import Path
import re
import pandas as pd

ROOT = Path.cwd()
INPUT = ROOT / "data" / "processed" / "total-color-data-final.csv"
OUTPUT = ROOT / "data" / "processed" / "02_dye_presence_targets.csv"
PREVALENCE = ROOT / "reports" / "modeling" / "02_dye_prevalence.csv"
REPORT = ROOT / "reports" / "modeling" / "02_dye_target_audit.txt"

for path in (INPUT,):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

df = pd.read_csv(INPUT)
required = {"Sample_ID", "Target_Label", "Adulterant_Type"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Required columns missing: {sorted(missing)}")

# Use only adulterated samples with a known, non-empty composition.
labels = df["Target_Label"].astype(str).str.strip().str.casefold()
types = df["Adulterant_Type"].fillna("").astype(str).str.strip()

mask = (
    labels.eq("adulterated")
    & types.ne("")
    & ~types.str.casefold().isin({"unknown", "nan", "none"})
)
work = df.loc[mask].copy()

if work.empty:
    raise ValueError("No eligible adulterated samples found.")

def parse_dyes(value):
    parts = [p.strip() for p in value.split("+")]
    return {
        re.sub(r"\s+", " ", p).casefold(): re.sub(r"\s+", " ", p)
        for p in parts
        if p and p.casefold() not in {"saffron", "unknown", "nan", "none"}
    }

parsed = work["Adulterant_Type"].map(parse_dyes)

# Preserve readable dye names and create stable column names.
display_names = {}
for dye_map in parsed:
    for key, display in dye_map.items():
        display_names.setdefault(key, display)

if not display_names:
    raise ValueError("No dye names could be extracted from Adulterant_Type.")

def safe_column_name(name):
    name = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return name or "Unknown_Dye"

column_map = {}
used = set()
for key in sorted(display_names):
    base = safe_column_name(display_names[key])
    col = base
    suffix = 2
    while col in used:
        col = f"{base}_{suffix}"
        suffix += 1
    used.add(col)
    column_map[key] = col

target_columns = []
for key, col in column_map.items():
    work[col] = parsed.map(lambda dyes: int(key in dyes))
    target_columns.append(col)

# Keep sample metadata, spectral predictors, and binary target columns together.
metadata = ["Sample_ID", "Target_Label", "Adulterant_Type"]
if "Dataset_Role" in work.columns:
    metadata.append("Dataset_Role")
spectral_columns = [
    c for c in work.columns
    if c not in set(metadata) and c not in set(target_columns)
]
output_columns = metadata + spectral_columns + target_columns
result = work[output_columns].copy()

ROOT.joinpath("data", "processed").mkdir(parents=True, exist_ok=True)
ROOT.joinpath("reports", "modeling").mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

prevalence_rows = []
for key, col in column_map.items():
    positives = int(result[col].sum())
    prevalence_rows.append({
        "Dye": display_names[key],
        "Target_Column": col,
        "Positive_Samples": positives,
        "Negative_Samples": len(result) - positives,
        "Prevalence_Percent": round(100 * positives / len(result), 2),
    })

prevalence = pd.DataFrame(prevalence_rows).sort_values(
    ["Positive_Samples", "Dye"], ascending=[False, True]
)
prevalence.to_csv(PREVALENCE, index=False, encoding="utf-8-sig")

report_lines = [
    "SAFFRON DYE-PRESENCE TARGET AUDIT",
    "=" * 45,
    f"Input: {INPUT}",
    f"Output: {OUTPUT}",
    f"Eligible adulterated samples: {len(result)}",
    f"Spectral predictor columns: {len(spectral_columns)}",
    f"Number of dye targets: {len(target_columns)}",
    "",
    "Dye target counts (1 = present, 0 = absent):",
    prevalence.to_string(index=False),
    "",
    "Notes:",
    "- Targets were parsed from Adulterant_Type; 'Saffron' is not a dye target.",
    "- This file contains eligible adulterated samples only.",
    "- A recorded composition is treated as the source of the labels; labels are not independently chemically verified here.",
    "- Many rare labels may be unsuitable for reliable model validation.",
    "- Fit preprocessing and feature selection inside each cross-validation training fold.",
]
REPORT.write_text("\n".join(report_lines), encoding="utf-8")

print("Target preparation completed.")
print(f"Eligible adulterated samples: {len(result)}")
print(f"Spectral predictors: {len(spectral_columns)}")
print(f"Dye targets: {len(target_columns)}")
print("\nDye prevalence:")
print(prevalence.to_string(index=False))
print("\nCreated:")
print(OUTPUT)
print(PREVALENCE)
print(REPORT)
