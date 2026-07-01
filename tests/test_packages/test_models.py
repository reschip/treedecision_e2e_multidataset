"""
test_models.py - Unit tests for domain model wrappers.

Tests the predict() / predict_proba() interfaces using
the pre-trained .joblib files that ship with the repository.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "packages"))


# -- Wine ---------------------------------------------------------------------

class TestWineClassifier:
    """Tests against the saved vino_blanco / vino_tinto models."""

    @pytest.fixture(scope="class")
    def white_clf(self):
        from domain_wine.architecture import WineClassifier
        return WineClassifier.load("white")

    @pytest.fixture(scope="class")
    def red_clf(self):
        from domain_wine.architecture import WineClassifier
        return WineClassifier.load("red")

    def test_white_predict_shape(self, white_clf):
        X = np.random.rand(5, 11)
        preds = white_clf.predict(X)
        assert preds.shape == (5,)

    def test_white_proba_shape(self, white_clf):
        X = np.random.rand(5, 11)
        proba = white_clf.predict_proba(X)
        assert proba.shape[0] == 5

    def test_red_predict_returns_int_scores(self, red_clf):
        X = np.random.rand(3, 11)
        preds = red_clf.predict(X)
        assert all(isinstance(int(p), int) for p in preds)

    def test_unfitted_raises(self):
        from domain_wine.architecture import WineClassifier
        clf = WineClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(np.zeros((1, 11)))


# -- Breast --------------------------------------------------------------------

class TestBreastClassifier:
    @pytest.fixture(scope="class")
    def clf(self):
        from domain_breast.classifier import BreastClassifier
        return BreastClassifier.load()

    def test_predict_binary(self, clf):
        X = np.random.rand(4, 30)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_proba_sums_to_one(self, clf):
        X = np.random.rand(3, 30)
        proba = clf.predict_proba(X)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)


# -- Adult Income --------------------------------------------------------------

class TestAdultClassifier:
    @pytest.fixture(scope="class")
    def clf(self):
        from domain_adultincome.classifier import AdultClassifier
        return AdultClassifier.load()

    def test_predict_binary(self, clf):
        X = np.random.rand(4, 14)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_proba_shape(self, clf):
        X = np.random.rand(4, 14)
        proba = clf.predict_proba(X)
        assert proba.shape == (4, 2)
