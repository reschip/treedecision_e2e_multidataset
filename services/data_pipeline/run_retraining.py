"""
run_retraining.py - Overnight batch retraining script.

This script:
  1. Loads the latest processed data for each domain.
  2. Re-trains each model from scratch.
  3. Saves updated .joblib artifacts, overwriting the previous version.

Intended to run as a scheduled job (Airflow / Prefect / cron).
Trigger manually with: python run_retraining.py

Future work:
  - Integrate MLflow for experiment tracking.
  - Add drift detection before deciding to retrain.
  - Send Slack / email notification on failure.
"""
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "packages"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def retrain_wine() -> None:
    log.info("Wine Quality: starting retraining")
    from domain_wine.dataset import WineDataset
    from domain_wine.architecture import WineClassifier

    for kind in ("white", "red"):
        log.info("  Training %s wine model", kind)
        ds = WineDataset(kind=kind, processed=True)
        X_train, _, y_train, _ = ds.get_splits(scale=True)
        clf = WineClassifier(kind=kind).fit(X_train, y_train)
        clf.save()
        log.info("  %s wine model saved.", kind)


def retrain_breast() -> None:
    log.info("Breast Cancer: starting retraining")
    from domain_breast.features import BreastFeatures
    from domain_breast.classifier import BreastClassifier

    ds = BreastFeatures(processed=True)
    X_train, _, y_train, _ = ds.get_splits(scale=True)
    clf = BreastClassifier().fit(X_train, y_train)
    clf.save()
    log.info("Breast cancer model saved.")


def retrain_adult() -> None:
    log.info("Adult Income: starting retraining")
    from domain_adultincome.dataset import AdultDataset
    from domain_adultincome.classifier import AdultClassifier

    ds = AdultDataset(processed=True)
    X_train, _, y_train, _ = ds.get_splits(scale=False)
    clf = AdultClassifier().fit(X_train, y_train)
    clf.save()
    log.info("Adult income model saved.")


if __name__ == "__main__":
    log.info("ML Monorepo - Batch Retraining starting")

    errors = []
    for name, fn in [("wine", retrain_wine), ("breast", retrain_breast), ("adult", retrain_adult)]:
        try:
            fn()
        except Exception as exc:
            log.error("%s retraining FAILED: %s", name, exc)
            errors.append(name)

    if errors:
        log.error("Retraining completed with errors in: %s", errors)
        sys.exit(1)
    else:
        log.info("All models retrained successfully.")
