"""
base_model.py - Abstract base class that all domain models must implement.

Every domain package (domain_wine, domain_breast, domain_adultincome)
must subclass BaseModel to guarantee a consistent interface for the API layer.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseModel(ABC):
    """
    Abstract interface shared by every production ML model in this monorepo.

    Subclasses must implement:
        - fit(X, y)          - train on labelled data
        - predict(X)         - return hard predictions
        - predict_proba(X)   - return class probabilities (if applicable)
        - save(path)         - persist the fitted model to disk
        - load(path)         - restore a persisted model from disk
    """

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseModel":
        """Train the model. Returns self for method chaining."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Return hard class predictions for input X."""

    def predict_proba(self, X: Any) -> Any:
        """Return class probabilities. Override in models that support it."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_proba."
        )

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model to *path* using joblib or equivalent."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseModel":
        """Restore a model previously saved with save()."""

    # Convenience helpers
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
