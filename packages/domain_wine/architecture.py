"""
architecture.py - Wine quality classifier (production wrapper).

Wraps the pre-trained joblib models (vino_blanco, vino_tinto) behind
the BaseModel interface so the API layer never touches joblib directly.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from core_ml.base_model import BaseModel

_MODEL_DIR = Path(__file__).resolve().parent / "models"

# Map human-readable names to saved artefacts
_MODEL_FILES = {
    "white": _MODEL_DIR / "vino_blanco.joblib",
    "red": _MODEL_DIR / "vino_tinto.joblib",
}


class WineClassifier(BaseModel):
    """Production-ready wrapper around the wine quality Random Forest models.

    Usage
    -----
    >>> clf = WineClassifier.load("white")
    >>> predictions = clf.predict(X_scaled)
    """

    def __init__(self, kind: str = "white"):
        if kind not in ("white", "red"):
            raise ValueError(f"kind must be 'white' or 'red', got '{kind}'")
        self.kind = kind
        self._model = None
        self.meta: dict = {}  # (#8) populated by load() / save()

    # BaseModel interface

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WineClassifier":
        """Re-train with the best hyperparameters found in the notebook (GridSearchCV)."""
        from sklearn.tree import DecisionTreeClassifier
        # Best params from notebook (SMOTE + GridSearch)
        self._model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            max_features="log2",
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
        dest = Path(path or _MODEL_FILES[self.kind])
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, dest)
        # (#8) Write version metadata
        meta = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model_type": type(self._model).__name__,
            "wine_kind": self.kind,
            "model_path": str(dest),
        }
        dest.with_name(dest.stem + "_meta.json").write_text(json.dumps(meta, indent=2))
        self.meta = meta

    @classmethod
    def load(cls, kind: str = "white") -> "WineClassifier":
        """Load a pre-trained model from disk."""
        instance = cls(kind=kind)
        model_path = _MODEL_FILES[kind]
        if not model_path.exists():
            raise FileNotFoundError(
                f"No saved model found at {model_path}. "
                "Run fit() and save() first."
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
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "unfitted"
        return f"<WineClassifier kind={self.kind!r} status={status}>"
