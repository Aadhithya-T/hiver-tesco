"""Intent classification baselines: Majority Class and TF-IDF + Logistic Regression."""

from collections import Counter
from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class MajorityClassIntentBaseline:
    """Trivial intent baseline that always predicts the most frequent intent from the training set."""

    def __init__(self):
        self.majority_class: Optional[str] = None
        self.class_distribution: Dict[str, int] = {}

    def fit(self, X_texts: List[str], y_labels: List[str]) -> "MajorityClassIntentBaseline":
        if not y_labels:
            raise ValueError("Cannot fit MajorityClassIntentBaseline on empty labels.")
        counts = Counter(y_labels)
        self.class_distribution = dict(counts)
        self.majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, X_texts: List[str]) -> List[str]:
        if self.majority_class is None:
            raise RuntimeError("Model must be fitted before predict.")
        return [self.majority_class] * len(X_texts)


class TfidfLogRegIntentBaseline:
    """Intent classifier using character/word TF-IDF n-grams and Logistic Regression.

    Trained strictly on leakage-safe query_text from the training split.
    """

    def __init__(
        self,
        ngram_range=(1, 2),
        max_features=1000,
        sublinear_tf=True,
        C=1.0,
        random_state=42,
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            min_df=1,
            strip_accents="unicode",
            lowercase=True,
        )
        self.classifier = LogisticRegression(
            C=C,
            class_weight="balanced",
            random_state=random_state,
            max_iter=1000,
        )
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X_texts: List[str], y_labels: List[str]) -> "TfidfLogRegIntentBaseline":
        X_tfidf = self.vectorizer.fit_transform(X_texts)
        self.classifier.fit(X_tfidf, y_labels)
        self.classes_ = self.classifier.classes_
        return self

    def predict(self, X_texts: List[str]) -> List[str]:
        X_tfidf = self.vectorizer.transform(X_texts)
        preds = self.classifier.predict(X_tfidf)
        return list(preds)

    def predict_proba(self, X_texts: List[str]) -> np.ndarray:
        X_tfidf = self.vectorizer.transform(X_texts)
        return self.classifier.predict_proba(X_tfidf)
