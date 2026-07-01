"""
classifier.py - Adult Income classifier (production wrapper).

Wraps the Decision Tree model (adult_income_decision_tree.joblib)
behind the BaseModel interface.

Predicts 1 = income >50K, 0 = income <=50K.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from core_ml.base_model import BaseModel

_MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "adult_income_decision_tree.joblib"
)


class AdultClassifier(BaseModel):
    """Decision Tree classifier for adult census income prediction.

    Usage
    -----
    >>> clf = AdultClassifier.load()
    >>> predictions = clf.predict(X)   # array of 0/1
    """

    def __init__(self):
        self._model = None
        self.meta: dict = {}  # (#8) populated by load() / save()

    # BaseModel interface

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdultClassifier":
        from sklearn.tree import DecisionTreeClassifier
        # Best hyperparameters from notebook GridSearchCV 
        self._model = DecisionTreeClassifier(
            min_samples_split=200,
            min_samples_leaf=10,
            max_features=None,
            max_depth=None,
            random_state=42,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_proba(X)

    def save(self, path: Path | None = None) -> None:
        self._check_fitted()
        dest = Path(path or _MODEL_PATH)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, dest)
        # (#8) Write version metadata
        meta = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model_type": type(self._model).__name__,
            "model_path": str(dest),
        }
        dest.with_name(dest.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
        self.meta = meta

    @classmethod
    def load(cls, path: Path | None = None) -> "AdultClassifier":
        instance = cls()
        model_path = Path(path or _MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No saved model at {model_path}. Run fit() + save() first."
            )
        instance._model = joblib.load(model_path)
        # (#8) Load metadata if it exists
        meta_path = model_path.with_name(model_path.stem + "_meta.json")
        instance.meta = json.loads(meta_path.read_text()) if meta_path.exists() \
            else {"note": "No metadata - model predates versioning."}
        return instance

    # Helpers

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "unfitted"
        return f"<AdultClassifier status={status}>"
