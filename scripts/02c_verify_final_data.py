from pathlib import Path
import pandas as pd

files = [
    Path("data/processed/total-color-data-final.csv"),
    Path("data/processed/02_dye_presence_targets.csv"),
]

for path in files:
    print(f"\n--- {path} ---")

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)
    print("Rows:", len(df))

    if "Sample_ID" in df.columns:
        ids = df["Sample_ID"].astype(str).str.strip()
        print("Sample 54 present:", ids.eq("54").any())

    if "Target_Label" in df.columns:
        print("Target counts:")
        print(df["Target_Label"].value_counts(dropna=False).to_string())
