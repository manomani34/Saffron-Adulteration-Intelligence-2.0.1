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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "processed" / "02_dye_presence_targets.csv"
REPORT_DIR = ROOT / "reports" / "logistic_regression"
OUTPUT_DIR = ROOT / "output" / "logistic_regression"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(f"Input dataset not found: {INPUT}")

df = pd.read_csv(INPUT)

targets = [
    c for c in [
        "Tartrazine", "Orange_2", "Quinoline_Yellow",
        "Erythrosine", "Azorubine", "Orange_GGN",
        "Ponceau_4R", "Amaranth", "Sunset_Yellow"
    ]
    if c in df.columns
]

metadata = {
    "Sample_ID", "Target_Label", "Adulterant_Type", "Dataset_Role"
}
spectral_columns = [
    c for c in df.columns
    if c not in metadata and c not in targets
]

if not targets:
    raise ValueError("No dye target columns found.")

X = df[spectral_columns].apply(pd.to_numeric, errors="coerce")
if X.isna().any().any():
    raise ValueError("Spectral predictors contain missing/non-numeric values.")

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
        ("classifier", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
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
target_audit = []

warnings.filterwarnings("ignore", category=ConvergenceWarning)

for target in targets:
    y = pd.to_numeric(df[target], errors="raise").astype(int)
    positives = int(y.sum())
    negatives = len(y) - positives

    target_audit.append({
        "Target": target,
        "Samples": len(y),
        "Positive": positives,
        "Negative": negatives,
        "Status": (
            "Exploratory only: fewer than 5 positives"
            if positives < 5 else "Exploratory: small dataset"
        ),
    })

    print(f"\nLogistic Regression | {target}: +{positives}, -{negatives}")

    if positives < N_SPLITS or negatives < N_SPLITS:
        print("Skipped: insufficient class examples.")
        continue

    for fold_number, (train_idx, test_idx) in enumerate(
        cv.split(X, y), start=1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_train.nunique() != 2 or y_test.nunique() != 2:
            continue

        model = clone(make_model())
        model.fit(X_train, y_train)

        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(
            y_test, predicted, labels=[0, 1]
        ).ravel()

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
            "Balanced_Accuracy": balanced_accuracy_score(y_test, predicted),
            "Sensitivity": recall_score(y_test, predicted, zero_division=0),
            "Specificity": tn / (tn + fp) if tn + fp else np.nan,
            "Precision": precision_score(y_test, predicted, zero_division=0),
            "F1": f1_score(y_test, predicted, zero_division=0),
            "Average_Precision": average_precision_score(
                y_test, probabilities
            ),
            "ROC_AUC": roc_auc_score(y_test, probabilities),
        })

        for pos, sample_index in enumerate(test_idx):
            prediction_results.append({
                "Target": target,
                "Fold": fold_number,
                "Sample_ID": df.iloc[sample_index]["Sample_ID"],
                "Actual": int(y_test.iloc[pos]),
                "Predicted": int(predicted[pos]),
                "Probability": float(probabilities[pos]),
            })

fold_df = pd.DataFrame(fold_results)
pred_df = pd.DataFrame(prediction_results)
audit_df = pd.DataFrame(target_audit)

metrics = [
    "Balanced_Accuracy", "Sensitivity", "Specificity",
    "Precision", "F1", "Average_Precision", "ROC_AUC"
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

files = {
    "folds": OUTPUT_DIR / "logistic_regression_cv_folds.csv",
    "predictions": OUTPUT_DIR / "logistic_regression_cv_predictions.csv",
    "summary": REPORT_DIR / "logistic_regression_metrics_summary.csv",
    "audit": REPORT_DIR / "logistic_regression_target_audit.csv",
    "report": REPORT_DIR / "logistic_regression_evaluation_report.txt",
}

fold_df.to_csv(files["folds"], index=False, encoding="utf-8-sig")
pred_df.to_csv(files["predictions"], index=False, encoding="utf-8-sig")
summary_df.to_csv(files["summary"], index=False, encoding="utf-8-sig")
audit_df.to_csv(files["audit"], index=False, encoding="utf-8-sig")

report_lines = [
    "LOGISTIC REGRESSION — DYE PRESENCE CLASSIFICATION",
    "=" * 58,
    f"Input: {INPUT}",
    f"Samples: {len(df)}",
    f"Spectral predictors before selection: {len(spectral_columns)}",
    f"Selected features per fold: {TOP_K}",
    "Model: Logistic Regression",
    "Regularization: L2 (scikit-learn default)",
    "C: 1.0",
    "Class weights: balanced",
    f"Cross-validation: {N_SPLITS} folds x {N_REPEATS} repeats",
    "Random state: 42",
    "",
    "Methodology:",
    "- Scaling and feature selection occur inside each training fold.",
    "- Results are exploratory because the dataset is small.",
    "- Repeated folds are correlated and are not independent experiments.",
    "- Targets with fewer than five positives are particularly unstable.",
    "",
    "Target audit:",
    audit_df.to_string(index=False),
    "",
    "Cross-validation summary:",
    summary_df.round(4).to_string(index=False)
        if not summary_df.empty else "No models completed.",
    "",
    "Generated files:",
    *[str(path) for path in files.values()],
]

files["report"].write_text("\n".join(report_lines), encoding="utf-8")

print("\nTARGET AUDIT")
print(audit_df.to_string(index=False))

print("\nLOGISTIC REGRESSION SUMMARY")
if summary_df.empty:
    print("No results produced.")
else:
    columns = [
        "Target", "Positive_Samples", "Completed_Folds",
        "Balanced_Accuracy_Mean", "Sensitivity_Mean",
        "Specificity_Mean", "F1_Mean",
        "Average_Precision_Mean", "ROC_AUC_Mean"
    ]
    print(summary_df[columns].round(3).to_string(index=False))

print("\nCreated files:")
for path in files.values():
    print(path)
