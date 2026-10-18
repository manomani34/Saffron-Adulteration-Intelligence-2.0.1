from pathlib import Path
import pandas as pd

source = Path("data/processed/total-color-data-final.csv")
targets = Path("data/processed/02_dye_presence_targets.csv")

df = pd.read_csv(source)
before = len(df)

print("Sample 54 before removal:")
print(df.loc[df["Sample_ID"].eq(54),
             ["Sample_ID", "Target_Label", "Adulterant_Type"]].to_string(index=False))

df = df.loc[~df["Sample_ID"].eq(54)].copy()
df.to_csv(source, index=False, encoding="utf-8-sig")

if targets.exists():
    t = pd.read_csv(targets)
    t = t.loc[~t["Sample_ID"].eq(54)].copy()
    t.to_csv(targets, index=False, encoding="utf-8-sig")
    print(f"Updated: {targets}")

print(f"\nRemoved rows: {before - len(df)}")
print(f"Final dataset rows: {len(df)}")
print(f"Sample 54 still present: {df['Sample_ID'].eq(54).any()}")
print(f"Updated: {source}")
