from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


BASE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_PATH = BASE_DIR / "CodaBench" / "Test Data" / "test_reference.csv"
MODELS = ["Becca", "Claire", "Leo", "Tim"]
OUTPUT_DIR = BASE_DIR / "confusion_matrices"


def normalize_sentence(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text


def main() -> None:
    ref_df = pd.read_csv(REFERENCE_PATH)
    ref_df = ref_df[["sentence", "label"]].copy()
    ref_df["norm_sentence"] = ref_df["sentence"].map(normalize_sentence)
    ref_df = ref_df.rename(columns={"label": "label_true"})
    ref_exact = ref_df[["sentence", "label_true"]].drop_duplicates(subset=["sentence"])
    ref_norm = ref_df[["norm_sentence", "label_true"]].drop_duplicates(subset=["norm_sentence"])

    labels = sorted(ref_df["label_true"].dropna().unique().tolist())
    summary_rows = []

    for model_name in MODELS:
        model_pred_path = BASE_DIR / "final_models" / model_name / "predictions.csv"
        model_out_dir = OUTPUT_DIR / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        if not model_pred_path.exists():
            summary_rows.append(
                {
                    "model": model_name,
                    "prediction_rows": 0,
                    "matched_rows": 0,
                    "coverage_pct": 0.0,
                    "status": f"missing file: {model_pred_path.name}",
                }
            )
            continue

        pred_df = pd.read_csv(model_pred_path)
        pred_df = pred_df[["sentence", "label"]].copy()
        pred_df["norm_sentence"] = pred_df["sentence"].map(normalize_sentence)
        pred_df = pred_df.rename(columns={"label": "label_pred"})

        matched_df = pred_df.merge(ref_exact, on="sentence", how="inner")

        # Fallback only when exact sentence matching misses rows.
        if len(matched_df) < len(pred_df):
            pred_norm = pred_df[["norm_sentence", "label_pred"]].drop_duplicates(subset=["norm_sentence"])
            matched_df = pred_norm.merge(ref_norm, on="norm_sentence", how="inner")

        matched_rows = len(matched_df)
        pred_rows = len(pred_df)
        coverage_pct = (matched_rows / pred_rows * 100.0) if pred_rows else 0.0

        summary_rows.append(
            {
                "model": model_name,
                "prediction_rows": pred_rows,
                "matched_rows": matched_rows,
                "coverage_pct": round(coverage_pct, 3),
                "status": "ok" if matched_rows else "no overlap",
            }
        )

        matched_df.to_csv(model_out_dir / "matched_eval_rows.csv", index=False)

        if matched_rows == 0:
            continue

        cm = confusion_matrix(
            matched_df["label_true"],
            matched_df["label_pred"],
            labels=labels,
        )

        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv(model_out_dir / "confusion_matrix_counts.csv")

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
        plt.title(
            f"{model_name} Confusion Matrix\\n"
            f"Matched rows: {matched_rows}/{pred_rows} ({coverage_pct:.2f}%)"
        )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(model_out_dir / "confusion_matrix.png", dpi=200)
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "coverage_summary.csv", index=False)


if __name__ == "__main__":
    main()
