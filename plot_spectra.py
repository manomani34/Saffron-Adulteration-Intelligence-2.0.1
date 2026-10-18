from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
CSV = BASE / "data" / "raw" / "total-color-data.csv"
OUT = BASE / "reports" / "spectra"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV, encoding="utf-8-sig")

metadata = {"Sample_ID", "Target_Label", "Adulterant_Type", "Dataset_Role"}
spectral_cols = [c for c in df.columns if c not in metadata]

# Convert chemical-shift column names and intensities to numeric values.
ppm = np.array([float(c) for c in spectral_cols])
order = np.argsort(ppm)
ppm = ppm[order]
spectral_cols = [spectral_cols[i] for i in order]
spectra = df[spectral_cols].apply(pd.to_numeric, errors="coerce").to_numpy()

if np.isnan(spectra).any():
    raise ValueError("Spectral data contains missing/non-numeric values.")

ids = df["Sample_ID"].astype(str).to_numpy()
labels = df["Target_Label"].astype(str).to_numpy()

def plot_spectra(indices, title, output, mode):
    fig, ax = plt.subplots(figsize=(16, 8), dpi=180)

    if mode == "all":
        # Different color for every sample.
        cmap = plt.get_cmap("turbo", len(indices))
        for j, i in enumerate(indices):
            ax.plot(
                ppm, spectra[i],
                color=cmap(j), linewidth=0.85, alpha=0.85,
                label=f"ID {ids[i]} | {labels[i]}"
            )
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1),
            fontsize=6.5, frameon=False
        )
        fig.subplots_adjust(right=0.78)
    else:
        # One color per spectrum; IDs in the legend.
        cmap = plt.get_cmap("turbo", max(len(indices), 1))
        for j, i in enumerate(indices):
            ax.plot(
                ppm, spectra[i],
                color=cmap(j), linewidth=0.9, alpha=0.85,
                label=f"ID {ids[i]}"
            )
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.01, 1),
            fontsize=8, frameon=False
        )
        fig.subplots_adjust(right=0.82)

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Chemical shift (ppm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.invert_xaxis()  # Conventional NMR axis
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created: {output}")

# 1. All samples, including references and unknowns.
plot_spectra(
    list(range(len(df))),
    f"All NMR spectra ({len(df)} samples)",
    OUT / "01_all_spectra.png",
    "all"
)

# 2. Samples explicitly labeled as adulterated.
idx = np.flatnonzero(labels == "Adulterated")
plot_spectra(
    idx,
    f"Adulterated samples ({len(idx)})",
    OUT / "02_adulterated_spectra.png",
    "subset"
)

# 3. Samples explicitly labeled as genuine.
idx = np.flatnonzero(labels == "Genuine")
plot_spectra(
    idx,
    f"Genuine samples ({len(idx)})",
    OUT / "03_genuine_spectra.png",
    "subset"
)

print("\nFinished. PNG files are in:", OUT)
