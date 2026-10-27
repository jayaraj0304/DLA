"""
Quora Question-Pair Similarity using NLP/ML
Feature extraction: TF-IDF + pairwise lexical features
Run: python main.py
Optional: python main.py --data data/quora_sample.csv
"""
import argparse
import os
import re
import sys
import subprocess

def ensure_dependencies():
    """Try importing required packages; offer a helpful install command if missing."""
    missing = []
    for module, package in [("numpy", "numpy"), ("pandas", "pandas"),
                            ("sklearn", "scikit-learn"), ("scipy", "scipy")]:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install them with: python -m pip install -r requirements.txt")
        sys.exit(1)

ensure_dependencies()

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def clean_text(text):
    """Normalize missing values and whitespace; vectorizer handles tokenization."""
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text).lower()).strip()

def pairwise_features(q1, q2):
    """Create numeric similarity features for each pair."""
    features = []
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on",
        "for", "and", "or", "what", "how", "why", "when", "where", "who", "do",
        "does", "did", "i", "you", "we", "it", "with", "can", "could"
    }
    for left, right in zip(q1, q2):
        words1 = re.findall(r"\b\w+\b", clean_text(left))
        words2 = re.findall(r"\b\w+\b", clean_text(right))
        set1, set2 = set(words1), set(words2)
        content1 = set(words1) - stop_words
        content2 = set(words2) - stop_words
        union = set1 | set2
        content_union = content1 | content2
        common = set1 & set2
        content_common = content1 & content2
        features.append([
            len(words1), len(words2),                       # question lengths
            abs(len(words1) - len(words2)),                 # length difference
            len(common),                                    # common unique words
            len(common) / (len(union) + 1e-9),              # Jaccard similarity
            len(content_common),                            # common content words
            len(content_common) / (len(content_union) + 1e-9),
            len(common) / (min(len(set1), len(set2)) + 1e-9),
            float(clean_text(left) == clean_text(right)),   # exact match
            abs(len(left) - len(right)),                   # character-length difference
        ])
    return np.asarray(features, dtype=float)

def build_features(q1, q2, vectorizer=None, fit=False):
    """Combine TF-IDF pair representations with engineered pairwise features."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2),
            max_features=20000, sublinear_tf=True
        )
    if fit:
        vectorizer.fit(list(q1) + list(q2))
    v1 = vectorizer.transform(q1)
    v2 = vectorizer.transform(q2)
    # Pair representations: absolute TF-IDF difference and elementwise product.
    tfidf_difference = abs(v1 - v2)
    tfidf_product = v1.multiply(v2)
    engineered = csr_matrix(pairwise_features(q1, q2))
    combined = hstack([tfidf_difference, tfidf_product, engineered], format="csr")
    return combined, vectorizer

def main():
    parser = argparse.ArgumentParser(description="Quora question-pair similarity")
    parser.add_argument("--data", default="data/quora_sample.csv",
                        help="CSV with question1, question2, is_duplicate columns")
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Dataset not found: {args.data}")
        print("Run from the repository folder, or pass --data path/to/file.csv")
        sys.exit(1)

    df = pd.read_csv(args.data)
    required = {"question1", "question2", "is_duplicate"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
    df = df.dropna(subset=["question1", "question2", "is_duplicate"]).copy()
    df["question1"] = df["question1"].map(clean_text)
    df["question2"] = df["question2"].map(clean_text)
    df["is_duplicate"] = df["is_duplicate"].astype(int)

    if len(df) < 8 or df["is_duplicate"].nunique() < 2:
        raise ValueError("Dataset needs at least 8 rows and both labels (0 and 1).")

    q1 = df["question1"].tolist()
    q2 = df["question2"].tolist()
    y = df["is_duplicate"].to_numpy()

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=42, stratify=y
    )
    q1_train = [q1[i] for i in train_idx]
    q2_train = [q2[i] for i in train_idx]
    q1_test = [q1[i] for i in test_idx]
    q2_test = [q2[i] for i in test_idx]

    vectorizer = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        max_features=20000, sublinear_tf=True
    )
    # Fit TF-IDF only on training questions to avoid test-set leakage.
    X_train, vectorizer = build_features(q1_train, q2_train, vectorizer, fit=True)
    X_test, _ = build_features(q1_test, q2_test, vectorizer, fit=False)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y[train_idx])
    predictions = model.predict(X_test)

    print("\n=== Quora Question-Pair Similarity ===")
    print(f"Rows: {len(df)} | Train: {len(train_idx)} | Test: {len(test_idx)}")
    print("\nFeature extraction:")
    print("  1. Word and bigram TF-IDF")
    print("  2. Absolute TF-IDF difference |v1 - v2|")
    print("  3. Elementwise TF-IDF product v1 * v2")
    print("  4. Pairwise features: lengths, common words, Jaccard, overlap, exact match")
    print(f"\nCombined feature matrix: train={X_train.shape}, test={X_test.shape}")
    print(f"\nAccuracy: {accuracy_score(y[test_idx], predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y[test_idx], predictions, zero_division=0))
    print("Confusion matrix [rows=true, columns=predicted]:")
    print(confusion_matrix(y[test_idx], predictions))

    examples = [
        ("How can I learn Python?", "What is the best way to learn Python?"),
        ("What is the capital of India?", "How do I make chocolate cake?")
    ]
    print("\nSample predictions (1 = likely duplicate, 0 = not duplicate):")
    for left, right in examples:
        x, _ = build_features([clean_text(left)], [clean_text(right)], vectorizer)
        pred = int(model.predict(x)[0])
        probability = float(model.predict_proba(x)[0][1])
        print(f"\nQ1: {left}\nQ2: {right}\nPrediction: {pred} (duplicate probability: {probability:.2f})")

if __name__ == "__main__":
    main()
