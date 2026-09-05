"""
phishing_detector.py
---------------------
Phishing Email Detection Model built with scikit-learn.

Pipeline:
1. Load labeled email dataset (subject, body, label)
2. Extract features:
     - TF-IDF text features from subject + body
     - Engineered features: URL count, suspicious-keyword count,
       exclamation marks, uppercase-word ratio, digit ratio, link-to-text ratio
3. Train a classifier (Random Forest) to label emails as "phishing" or "safe"
4. Evaluate with accuracy, confusion matrix, and classification report
5. Provide a predict_email() function to classify new/unseen emails
"""

import re
import pandas as pd
import numpy as np
import matplotlib  # type: ignore[reportMissingImports]
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[reportMissingImports]  # pyright: ignore[reportMissingImports]

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Feature engineering helpers
# ---------------------------------------------------------------------
URL_REGEX = re.compile(r"https?://\S+|www\.\S+")

SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "suspend", "immediately", "click here", "confirm",
    "password", "account", "limited", "act now", "winner", "won", "claim",
    "security alert", "update your", "restricted", "unusual activity",
    "final notice", "expire", "login", "credentials", "bank",
]


def extract_engineered_features(texts):
    """Return a DataFrame of hand-crafted features for a list of raw email texts."""
    feats = []
    for text in texts:
        lower = text.lower()
        url_count = len(URL_REGEX.findall(text))
        keyword_count = sum(lower.count(kw) for kw in SUSPICIOUS_KEYWORDS)
        exclamations = text.count("!")
        digit_count = sum(c.isdigit() for c in text)
        words = text.split()
        upper_words = sum(1 for w in words if w.isupper() and len(w) > 1)
        length = max(len(text), 1)

        feats.append({
            "url_count": url_count,
            "suspicious_keyword_count": keyword_count,
            "exclamation_count": exclamations,
            "uppercase_word_count": upper_words,
            "digit_ratio": digit_count / length,
            "text_length": length,
        })
    return pd.DataFrame(feats)


# ---------------------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------------------
def load_dataset(path="data/emails_dataset.csv"):
    df = pd.read_csv(path)
    df["full_text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    return df


# ---------------------------------------------------------------------
# 3. Build features (TF-IDF + engineered), train, evaluate
# ---------------------------------------------------------------------
def main():
    df = load_dataset()
    X_text = df["full_text"].values
    y = df["label"].values

    # Train/test split (stratified so both classes are balanced in each split)
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # --- TF-IDF text features ---
    vectorizer = TfidfVectorizer(
        max_features=1500,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    # --- Engineered features ---
    train_eng = extract_engineered_features(X_train_text)
    test_eng = extract_engineered_features(X_test_text)

    # Combine TF-IDF (sparse) with engineered features (dense -> sparse)
    X_train = hstack([X_train_tfidf, csr_matrix(train_eng.values)])
    X_test = hstack([X_test_tfidf, csr_matrix(test_eng.values)])

    # --- Train classifier ---
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels)

    print("=" * 60)
    print(f"Accuracy: {acc:.4f}")
    print("=" * 60)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"Labels order: {labels}")
    print(cm)
    print("=" * 60)
    print("Classification Report:")
    print(report)

    # --- Save confusion matrix plot ---
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Phishing Detector — Confusion Matrix\nAccuracy: {acc:.2%}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nSaved confusion matrix plot -> confusion_matrix.png")

    # --- Top predictive features (from TF-IDF importances) ---
    feature_names = list(vectorizer.get_feature_names_out()) + list(train_eng.columns)
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\nTop 15 most predictive features:")
    for i in top_idx:
        print(f"  {feature_names[i]:<30s} {importances[i]:.4f}")

    return vectorizer, clf, train_eng.columns.tolist()


# ---------------------------------------------------------------------
# 4. Predict on new, unseen emails
# ---------------------------------------------------------------------
def predict_email(text, vectorizer, clf, eng_cols):
    tfidf_feat = vectorizer.transform([text])
    eng_feat = extract_engineered_features([text])[eng_cols]
    X = hstack([tfidf_feat, csr_matrix(eng_feat.values)])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    classes = clf.classes_
    proba_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}
    return pred, proba_dict


if __name__ == "__main__":
    vectorizer, clf, eng_cols = main()

    # Demo predictions on brand-new example emails
    print("\n" + "=" * 60)
    print("Demo predictions on new, unseen emails:")
    print("=" * 60)

    examples = [
        "URGENT: Your account will be suspended! Click http://verify-now-secure.com "
        "immediately to confirm your password and avoid permanent closure.",
        "Hi Sarah, attached is the agenda for tomorrow's 10am standup. Let me know if "
        "you'd like to add anything.",
    ]
    for ex in examples:
        pred, proba = predict_email(ex, vectorizer, clf, eng_cols)
        print(f"\nEmail: {ex[:70]}...")
        print(f"  -> Prediction: {pred}  |  Probabilities: {proba}")
