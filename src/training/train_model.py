import os
import re
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_and_evaluate_models(dataset_path="data/generated/reviews_dataset.csv", models_dir="models"):
    print("=" * 60)
    print("STARTING ML TRAINING PIPELINE (REVIO AI ENGINE)")
    print("=" * 60)

    os.makedirs(models_dir, exist_ok=True)

    # 1. Load Dataset
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # 2. Schema Validation
    required_cols = ["review_id", "review_text", "rating", "sentiment_label"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns for training: {missing_cols}")

    # Remove invalid rows if any
    df = df.dropna(subset=["review_text", "sentiment_label"])
    print(f"Loaded {len(df)} valid records for training.")

    # 3. Preprocess Text
    df["clean_text"] = df["review_text"].apply(preprocess_text)

    X = df["clean_text"]
    y = df["sentiment_label"]

    # 4. Train / Validation / Test Split (70% Train, 15% Val, 15% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.17647, random_state=42, stratify=y_train_val
    ) # 0.17647 of 85% is ~15% of total dataset

    print(f"Dataset split: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")

    # 5. Define Baseline Model Pipelines
    candidate_models = {
        "TFIDF_LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]),
        "TFIDF_LinearSVM": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", LinearSVC(C=1.0, random_state=42, max_iter=2000))
        ]),
        "TFIDF_MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", MultinomialNB(alpha=0.5))
        ])
    }

    results = {}
    best_model_name = None
    best_val_f1 = -1.0
    best_pipeline = None

    print("\n--- Training and Comparing Candidate Models ---")

    for model_name, pipeline in candidate_models.items():
        print(f"\nTraining model: {model_name}...")
        # Fit ONLY on training data to prevent data leakage
        pipeline.fit(X_train, y_train)

        # Evaluate on Validation set
        val_preds = pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds)
        p, r, f1, _ = precision_recall_fscore_support(y_val, val_preds, average="macro")

        print(f"[{model_name}] Val Accuracy: {val_acc:.4f} | Val Macro Precision: {p:.4f} | Val Macro Recall: {r:.4f} | Val Macro F1: {f1:.4f}")

        results[model_name] = {
            "val_accuracy": float(val_acc),
            "val_macro_precision": float(p),
            "val_macro_recall": float(r),
            "val_macro_f1": float(f1)
        }

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model_name = model_name
            best_pipeline = pipeline

    print(f"\nBest Model Selected based on Validation Macro F1: {best_model_name} (F1: {best_val_f1:.4f})")

    # 6. Evaluate Best Model on Untouched Test Set
    print("\n--- Final Evaluation on Untouched Test Set ---")
    test_preds = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test, test_preds, average="macro")
    class_report = classification_report(y_test, test_preds, output_dict=True)
    labels = sorted(list(y.unique()))
    cm = confusion_matrix(y_test, test_preds, labels=labels).tolist()

    print(f"Test Set Accuracy: {test_acc:.4f}")
    print(f"Test Set Macro F1: {test_f1:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds))

    # 7. Save Best Model
    model_path = os.path.join(models_dir, "best_review_model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"Saved trained model pipeline to {model_path}")

    # 8. Save Model Metadata & Evaluation Results
    metadata = {
        "model_version": "v1.0.0",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model_architecture": best_model_name,
        "classes": labels,
        "dataset_split": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test)
        },
        "validation_comparison": results,
        "test_performance": {
            "accuracy": float(test_acc),
            "macro_precision": float(test_p),
            "macro_recall": float(test_r),
            "macro_f1": float(test_f1),
            "classification_report": class_report,
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm
            }
        },
        "limitations_disclaimer": (
            "Model trained and evaluated on synthetic customer review data. "
            "Synthetic data metrics do not guarantee real-world out-of-domain performance. "
            "Re-training on real domain customer feedback is recommended prior to production deployment."
        )
    }

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to {metadata_path}")
    print("=" * 60)
    return metadata

if __name__ == "__main__":
    train_and_evaluate_models()
