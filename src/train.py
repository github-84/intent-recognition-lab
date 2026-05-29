from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "intents.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "intent_model.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"

#数据
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"text", "intent"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.dropna(subset=["text", "intent"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]

    if df.empty:
        raise ValueError("Dataset is empty after removing blank rows.")

    class_counts = df["intent"].value_counts()
    too_small = class_counts[class_counts < 2]
    if not too_small.empty:
        labels = ", ".join(too_small.index.tolist())
        raise ValueError(f"Each intent needs at least 2 samples for stratified split: {labels}")

    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 3),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def save_reports(
    y_test: pd.Series,
    y_pred: list[str],
    labels: list[str],
    accuracy: float,
    train_size: int,
    test_size: int,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = classification_report(y_test, y_pred, labels=labels, digits=4, zero_division=0)
    (REPORT_DIR / "classification_report.txt").write_text(
        "\n".join(
            [
                "Intent Recognition Classification Report",
                f"Accuracy: {accuracy:.4f}",
                f"Train samples: {train_size}",
                f"Test samples: {test_size}",
                "",
                report,
            ]
        ),
        encoding="utf-8",
    )

    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_df.to_csv(REPORT_DIR / "confusion_matrix.csv", encoding="utf-8-sig")


def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> float:
    df = load_dataset(data_path)
    labels = sorted(df["intent"].unique().tolist())

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["intent"],
        test_size=0.2,
        random_state=42,
        stratify=df["intent"],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "labels": labels,
            "data_path": str(data_path),
        },
        model_path,
    )

    save_reports(y_test, y_pred, labels, accuracy, len(x_train), len(x_test))
    return accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Chinese intent recognition model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to intents.csv")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to save model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score = train(args.data, args.model)
    print(f"Training finished. Accuracy: {score:.4f}")
    print(f"Model saved to: {args.model}")
    print(f"Reports saved to: {REPORT_DIR}")
