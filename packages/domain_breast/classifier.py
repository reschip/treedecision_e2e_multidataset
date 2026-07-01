"""
classifier.py - Breast Cancer classifier (production wrapper).

Wraps the pre-trained Decision Tree model (wdbc_model.joblib) behind
the BaseModel interface.

Predicts 1 = Malignant, 0 = Benign.

The saved model was trained with:
    criterion='entropy', max_depth=8, min_samples_split=15,
    min_samples_leaf=20, class_weight='balanced', splitter='best'
    (selected by GridSearchCV optimizing recall -  of the notebook)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from core_ml.base_model import BaseModel

_MODEL_PATH = Path(__file__).resolve().parent / "models" / "wdbc_model.joblib"
_META_PATH  = Path(__file__).resolve().parent / "models" / "wdbc_model_meta.json"


class BreastClassifier(BaseModel):
    """Production wrapper around the WDBC cancer detection model.

    Predicts 1 = Malignant, 0 = Benign.

    Usage
    -----
    >>> clf = BreastClassifier.load()
    >>> predictions = clf.predict(X_scaled)   # array of 0/1
    >>> proba = clf.predict_proba(X_scaled)   # shape (n, 2)
    >>> print(clf.meta)                        # model version info
    """

    def __init__(self):
        self._model = None
        self.meta: dict = {}  # (#8) populated by load() / save()

    # BaseModel interface

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BreastClassifier":
        from sklearn.tree import DecisionTreeClassifier
        # Best hyperparameters from GridSearchCV (notebook scoring='recall')
        self._model = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=8,
            min_samples_split=15,
            min_samples_leaf=20,
            class_weight="balanced",
            splitter="best",
            random_state=42,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        if not hasattr(self._model, "predict_proba"):
            raise RuntimeError("Loaded model was not trained with probability=True.")
        return self._model.predict_proba(X)

    def save(self, path: Path | None = None) -> None:
        self._check_fitted()
        dest = Path(path or _MODEL_PATH)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, dest)
        # (#8) Write version metadata alongside the model
        meta = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model_type": type(self._model).__name__,
            "model_path": str(dest),
            "n_classes": int(len(self._model.classes_)) if hasattr(self._model, "classes_") else None,
        }
        meta_path = dest.with_name(dest.stem + "_meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))
        self.meta = meta

    @classmethod
    def load(cls, path: Path | None = None) -> "BreastClassifier":
        instance = cls()
        model_path = Path(path or _MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No saved model at {model_path}. Run fit() + save() first."
            )
        instance._model = joblib.load(model_path)
        # (#8) Load version metadata if it exists
        meta_path = model_path.with_name(model_path.stem + "_meta.json")
        if meta_path.exists():
            instance.meta = json.loads(meta_path.read_text())
        else:
            instance.meta = {"note": "No metadata found - model predates versioning."}
        return instance

    # Helpers

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() or load() first.")

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "unfitted"
        return f"<BreastClassifier status={status}>"
