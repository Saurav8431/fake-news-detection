from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
ENGLISH_DATA_PATH = RAW_DIR / "welfake.parquet"
HINDI_FAKE_PATH = RAW_DIR / "hindi_fake.xlsx"
HINDI_REAL_PATH = RAW_DIR / "hindi_authentic.xlsx"
MODEL_DIR = ROOT / "models"
ENGLISH_MODEL_PATH = MODEL_DIR / "fake_news_pipeline.joblib"
ENGLISH_METRICS_PATH = MODEL_DIR / "metrics.json"
HINDI_MODEL_PATH = MODEL_DIR / "hindi_fake_news_pipeline.joblib"
HINDI_METRICS_PATH = MODEL_DIR / "hindi_metrics.json"
ENGLISH_DATASET_URL = (
    "https://huggingface.co/datasets/davanstrien/WELFake/resolve/main/"
    "data/train-00000-of-00001-290868f0a36350c5.parquet?download=true"
)
HINDI_DATASET_URLS = {
    HINDI_FAKE_PATH: (
        "https://raw.githubusercontent.com/BSAkash/"
        "Hindi-Fake-News-Detection-Dataset/master/fake.xlsx"
    ),
    HINDI_REAL_PATH: (
        "https://raw.githubusercontent.com/BSAkash/"
        "Hindi-Fake-News-Detection-Dataset/master/authentic.xlsx"
    ),
}


def ensure_download(path: Path, url: str, description: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {description}...")
        try:
            urlretrieve(url, path)
        except Exception:
            path.unlink(missing_ok=True)
            raise


def clean_frame(frame: pd.DataFrame) -> tuple[list[str], pd.Series]:
    title = frame["title"].fillna("").astype(str).str.strip()
    body = frame["text"].fillna("").astype(str).str.strip()
    frame = pd.DataFrame(
        {
            "content": (title + "\n\n" + body).str.strip(),
            "label": pd.to_numeric(frame["label"], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["label"])
    frame = frame[frame["content"].str.len() > 0]
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame["label"].isin([0, 1])]
    frame = frame.drop_duplicates(subset=["content"], keep="first")

    if frame["label"].nunique() != 2:
        raise ValueError("Dataset must contain both fake (0) and real (1) labels.")
    if len(frame) < 10:
        raise ValueError("Dataset contains too few valid articles to train.")

    return frame["content"].tolist(), frame["label"]


def load_english_dataset() -> tuple[list[str], pd.Series]:
    ensure_download(
        ENGLISH_DATA_PATH,
        ENGLISH_DATASET_URL,
        "WELFake dataset (about 152 MB)",
    )
    frame = pd.read_parquet(ENGLISH_DATA_PATH)
    required = {"title", "text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    return clean_frame(frame)


def load_hindi_dataset() -> tuple[list[str], pd.Series]:
    frames = []
    for path, url in HINDI_DATASET_URLS.items():
        kind = "fake" if path == HINDI_FAKE_PATH else "authentic"
        ensure_download(path, url, f"Hindi {kind} news data")
        source = pd.read_excel(path)
        required = {"Title", "Summary"}
        missing = required.difference(source.columns)
        if missing:
            raise ValueError(
                f"Hindi dataset file {path.name} is missing columns: "
                f"{sorted(missing)}"
            )
        frames.append(
            pd.DataFrame(
                {
                    "title": source["Title"],
                    "text": source["Summary"],
                    "label": 0 if kind == "fake" else 1,
                }
            )
        )

    return clean_frame(pd.concat(frames, ignore_index=True))


def build_pipeline(language: str) -> Pipeline:
    if language == "hindi":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            max_features=150_000,
            min_df=2,
            sublinear_tf=True,
        )
    else:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=100_000,
            min_df=2,
            sublinear_tf=True,
        )

    return Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_evaluate(
    language: str,
    dataset: str,
    texts: list[str],
    labels: pd.Series,
    model_path: Path,
    metrics_path: Path,
) -> None:
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels.to_numpy(),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    pipeline = build_pipeline(language)
    print(
        f"Training {language.title()} model on {len(x_train):,} articles; "
        f"evaluating on {len(x_test):,}..."
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "language": language,
        "dataset": dataset,
        "dataset_rows_used": len(texts),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "label_mapping": {"0": "fake", "1": "real"},
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_fake": float(
            precision_score(y_test, predictions, pos_label=0, zero_division=0)
        ),
        "recall_fake": float(
            recall_score(y_test, predictions, pos_label=0, zero_division=0)
        ),
        "f1_fake": float(f1_score(y_test, predictions, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix_labels": ["fake", "real"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


def main() -> None:
    english_texts, english_labels = load_english_dataset()
    hindi_texts, hindi_labels = load_hindi_dataset()

    train_and_evaluate(
        "english",
        "WELFake (davanstrien/WELFake)",
        english_texts,
        english_labels,
        ENGLISH_MODEL_PATH,
        ENGLISH_METRICS_PATH,
    )
    train_and_evaluate(
        "hindi",
        "Hindi Fake News Detection Dataset (BSAkash)",
        hindi_texts,
        hindi_labels,
        HINDI_MODEL_PATH,
        HINDI_METRICS_PATH,
    )


if __name__ == "__main__":
    main()
