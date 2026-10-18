from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")
SOURCE = ROOT / "data" / "processed" / "total-color-data-final.csv"
OUT = ROOT / "reports" / "modeling"

OUT.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {SOURCE.resolve()}\n"
        "First create total-color-data-final.csv."
    )

df = pd.read_csv(SOURCE)

required = {"Sample_ID", "Target_Label", "Adulterant_Type"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Required columns missing: {sorted(missing)}")

# First four columns are metadata; remaining columns are NMR spectral variables.
metadata = ["Sample_ID", "Target_Label", "Adulterant_Type", "Dataset_Role"]
spectral_cols = [c for c in df.columns if c not in metadata]

if len(spectral_cols) == 0:
    raise ValueError("No spectral columns were found.")

X = df[spectral_cols].apply(pd.to_numeric, errors="coerce")
y_label = df["Target_Label"].fillna("").astype(str).str.strip()
adulterant = df["Adulterant_Type"].fillna("").astype(str).str.strip()

# Main target: classify the adulterant composition among adulterated samples.
is_adulterated = y_label.str.casefold().eq("adulterated")
is_genuine = y_label.str.casefold().eq("genuine")
is_unknown = adulterant.str.casefold().eq("unknown")

# Do not silently discard rows from the source; report what will be eligible
# for the first adulterant-type classification experiment.
eligible = is_adulterated & ~is_unknown & adulterant.ne("")

class_counts = adulterant[eligible].value_counts().rename_axis(
    "Adulterant_Type"
).reset_index(name="Sample_Count")

class_counts["Percent_of_Eligible"] = (
    100 * class_counts["Sample_Count"] / max(int(eligible.sum()), 1)
).round(2)

class_counts.to_csv(
    OUT / "01_adulterant_class_counts.csv",
    index=False,
    encoding="utf-8-sig"
)

numeric_X = X.to_numpy(dtype=float)
all_missing_cols = X.columns[X.isna().all()].tolist()
constant_cols = [
    c for c in X.columns
    if X[c].nunique(dropna=True) <= 1
]
duplicate_ids = df["Sample_ID"].duplicated(keep=False)
duplicate_spectra = X.duplicated(keep=False)

spectral_min = float(np.nanmin(numeric_X))
spectral_max = float(np.nanmax(numeric_X))
row_max = np.nanmax(numeric_X, axis=1)
row_min = np.nanmin(numeric_X, axis=1)
row_norm = np.sqrt(np.nansum(numeric_X ** 2, axis=1))

report = []
report.append("SAFFRON ARTIFICIAL-COLOUR CLASSIFICATION — DATA AUDIT")
report.append("=" * 62)
report.append(f"Source: {SOURCE.resolve()}")
report.append(f"Rows: {len(df)}")
report.append(f"Metadata columns recognized: {len([c for c in metadata if c in df.columns])}")
report.append(f"Spectral variables: {len(spectral_cols)}")
report.append(f"Duplicate Sample_ID rows: {int(duplicate_ids.sum())}")
report.append(f"Rows with duplicate spectral vectors: {int(duplicate_spectra.sum())}")
report.append(f"Spectral cells missing/non-numeric: {int(X.isna().sum().sum())}")
report.append(f"All-missing spectral columns: {len(all_missing_cols)}")
report.append(f"Constant/all-single-value spectral columns: {len(constant_cols)}")
report.append(f"Global minimum spectral intensity: {spectral_min:.6f}")
report.append(f"Global maximum spectral intensity: {spectral_max:.6f}")
report.append("")
report.append("Target_Label counts:")
report.append(y_label.value_counts(dropna=False).to_string())
report.append("")
report.append("Adulterant_Type counts (all rows):")
report.append(adulterant.replace("", "[MISSING]").value_counts(dropna=False).to_string())
report.append("")
report.append("Rows eligible for first adulterant-type classification:")
report.append(f"Adulterated rows: {int(is_adulterated.sum())}")
report.append(f"Genuine rows: {int(is_genuine.sum())}")
report.append(f"Unknown adulterant rows: {int((is_adulterated & is_unknown).sum())}")
report.append(f"Eligible labelled adulterated rows: {int(eligible.sum())}")
report.append(f"Number of distinct adulterant/composition classes: {int(adulterant[eligible].nunique())}")
report.append("")
report.append("Class counts:")
report.append(class_counts.to_string(index=False) if len(class_counts) else "No eligible classes.")
report.append("")
report.append("Largest 10 row maxima (intensity audit only; not automatic exclusions):")

max_table = df[["Sample_ID", "Target_Label", "Adulterant_Type"]].copy()
max_table["Max_Raw_Intensity"] = row_max
max_table["Min_Raw_Intensity"] = row_min
max_table["L2_Norm"] = row_norm
report.append(
    max_table.sort_values("Max_Raw_Intensity", ascending=False)
    .head(10).to_string(index=False)
)

report.append("")
report.append("IMPORTANT INTERPRETATION:")
report.append("- The first classifier predicts the exact Adulterant_Type composition label.")
report.append("- Mixtures such as two-dye combinations are separate classes in this first experiment.")
report.append("- Very small classes may not support reliable cross-validation.")
report.append("- Raw intensity extremes are flagged for inspection, not automatically removed.")
report.append("- Genuine samples are not used to train the adulterant-type classifier.")
report.append("- Later preprocessing and feature selection must be fitted inside each CV fold.")

report_text = "\n".join(report)
(OUT / "01_dataset_audit.txt").write_text(report_text, encoding="utf-8")

print(report_text)
print("\nCreated:")
print(f"  {OUT / '01_dataset_audit.txt'}")
print(f"  {OUT / '01_adulterant_class_counts.csv'}")
