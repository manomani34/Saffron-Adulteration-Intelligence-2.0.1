from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "processed" / "02_dye_presence_targets.csv"
REPORT_DIR = ROOT / "reports" / "svm"
OUTPUT_DIR = ROOT / "output" / "svm"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(f"Input dataset not found: {INPUT}")

df = pd.read_csv(INPUT)

target_candidates = [
    "Tartrazine",
    "Orange_2",
    "Quinoline_Yellow",
    "Erythrosine",
    "Azorubine",
    "Orange_GGN",
    "Ponceau_4R",
    "Amaranth",
    "Sunset_Yellow",
]

targets = [c for c in target_candidates if c in df.columns]

metadata = {
    "Sample_ID",
    "Target_Label",
    "Adulterant_Type",
    "Dataset_Role",
}

spectral_columns = [
    c for c in df.columns
    if c not in metadata and c not in targets
]

if not targets:
    raise ValueError("No dye target columns found.")

X = df[spectral_columns].apply(pd.to_numeric, errors="coerce")

if X.isna().any().any():
    raise ValueError("Spectral data contains missing or non-numeric values.")

if len(df) != 28:
    print(f"WARNING: Expected 28 samples; found {len(df)}.")

N_SPLITS = 2
N_REPEATS = 10
TOP_K = min(50, len(spectral_columns))
RANDOM_STATE = 42

def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(
            score_func=lambda x, y: mutual_info_classif(
                x, y, random_state=RANDOM_STATE
            ),
            k=TOP_K,
        )),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])

cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)

fold_results = []
prediction_results = []
target_summary = []

warnings.filterwarnings("ignore", category=ConvergenceWarning)

for target in targets:
    y = pd.to_numeric(df[target], errors="raise").astype(int)

    positives = int(y.sum())
    negatives = int(len(y) - positives)

    target_summary.append({
        "Target": target,
        "Samples": len(y),
        "Positive": positives,
        "Negative": negatives,
        "Status": (
            "Exploratory only: fewer than 5 positives"
            if positives < 5 else
            "Exploratory: small dataset"
        ),
    })

    print(
        f"\nSVM | Target={target} | "
        f"Positive={positives} | Negative={negatives}"
    )

    if positives < N_SPLITS or negatives < N_SPLITS:
        print("Skipped: insufficient class examples.")
        continue

    for fold_number, (train_idx, test_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() != 2 or y_test.nunique() != 2:
            continue

        model = clone(make_model())
        model.fit(X_train, y_train)

        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_test, predicted, labels=[0, 1]
        ).ravel()

        sensitivity = (
            tp / (tp + fn) if (tp + fn) > 0 else np.nan
        )
        specificity = (
            tn / (tn + fp) if (tn + fp) > 0 else np.nan
        )

        fold_results.append({
            "Target": target,
            "Fold": fold_number,
            "Train_Size": len(train_idx),
            "Test_Size": len(test_idx),
            "Train_Positive": int(y_train.sum()),
            "Test_Positive": int(y_test.sum()),
            "True_Positive": int(tp),
            "False_Positive": int(fp),
            "True_Negative": int(tn),
            "False_Negative": int(fn),
            "Balanced_Accuracy": balanced_accuracy_score(
                y_test, predicted
            ),
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Precision": precision_score(
                y_test, predicted, zero_division=0
            ),
            "F1": f1_score(
                y_test, predicted, zero_division=0
            ),
            "Average_Precision": average_precision_score(
                y_test, probabilities
            ),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
        })

        for position, sample_index in enumerate(test_idx):
            prediction_results.append({
                "Target": target,
                "Fold": fold_number,
                "Sample_ID": df.iloc[sample_index]["Sample_ID"],
                "Actual": int(y_test.iloc[position]),
                "Predicted": int(predicted[position]),
                "Probability": float(probabilities[position]),
            })

fold_df = pd.DataFrame(fold_results)
pred_df = pd.DataFrame(prediction_results)
target_df = pd.DataFrame(target_summary)

metrics = [
    "Balanced_Accuracy",
    "Sensitivity",
    "Specificity",
    "Precision",
    "F1",
    "Average_Precision",
    "ROC_AUC",
]

summary_rows = []

for target in targets:
    subset = fold_df[fold_df["Target"] == target]

    if subset.empty:
        continue

    row = {
        "Target": target,
        "Positive_Samples": int(df[target].sum()),
        "Negative_Samples": int(len(df) - df[target].sum()),
        "Completed_Folds": len(subset),
    }

    for metric in metrics:
        values = subset[metric].astype(float)
        row[f"{metric}_Mean"] = values.mean()
        row[f"{metric}_SD"] = values.std(ddof=1)

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

fold_path = OUTPUT_DIR / "svm_cv_folds.csv"
pred_path = OUTPUT_DIR / "svm_cv_predictions.csv"
summary_path = REPORT_DIR / "svm_metrics_summary.csv"
target_path = REPORT_DIR / "svm_target_audit.csv"
report_path = REPORT_DIR / "svm_evaluation_report.txt"

fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
target_df.to_csv(target_path, index=False, encoding="utf-8-sig")

report_lines = [
    "SVM MODEL — DYE PRESENCE CLASSIFICATION",
    "=" * 55,
    f"Input: {INPUT}",
    f"Samples: {len(df)}",
    f"Spectral features before selection: {len(spectral_columns)}",
    f"Selected features per fold: {TOP_K}",
    "Kernel: RBF",
    "C: 1.0",
    "Gamma: scale",
    "Class weights: balanced",
    f"Cross-validation: {N_SPLITS} folds x {N_REPEATS} repeats",
    "Random state: 42",
    "",
    "Methodology:",
    "- Scaling and feature selection are fitted within each training fold.",
    "- Metrics are exploratory due to the small dataset.",
    "- Repeated cross-validation folds are correlated, not independent experiments.",
    "- Rare targets must not be interpreted as reliably validated classifiers.",
    "",
    "Target audit:",
    target_df.to_string(index=False),
    "",
    "Cross-validation summary:",
    (
        summary_df.round(4).to_string(index=False)
        if not summary_df.empty else
        "No target had sufficient examples to complete evaluation."
    ),
    "",
    "Output files:",
    str(fold_path),
    str(pred_path),
    str(summary_path),
    str(target_path),
]

report_path.write_text("\n".join(report_lines), encoding="utf-8")

print("\nTARGET AUDIT")
print(target_df.to_string(index=False))

print("\nSVM SUMMARY")
if summary_df.empty:
    print("No results produced.")
else:
    print(
        summary_df[
            [
                "Target",
                "Positive_Samples",
                "Completed_Folds",
                "Balanced_Accuracy_Mean",
                "Sensitivity_Mean",
                "Specificity_Mean",
                "F1_Mean",
                "Average_Precision_Mean",
                "ROC_AUC_Mean",
            ]
        ].round(3).to_string(index=False)
    )

print("\nCreated files:")
for path in [fold_path, pred_path, summary_path, target_path, report_path]:
    print(path)
