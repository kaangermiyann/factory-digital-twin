"""Metrikler, aciklanabilirlik ve sizinti kontrolu."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "r2": float("nan"), "n": 0}
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def recall_at_precision(y_true: np.ndarray, proba: np.ndarray, target_precision: float = 0.5) -> float:
    """Operasyonel metrik: 'her 2 alarmdan 1'i gercek' kosulunda kac olayi yakaliyoruz?

    PR-AUC guzel bir ozet ama operatore anlatilamaz. Bu anlatilabilir.
    """
    precision, recall, _ = precision_recall_curve(y_true, proba)
    feasible = recall[precision >= target_precision]
    return float(feasible.max()) if len(feasible) else 0.0


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"pr_auc": float("nan"), "roc_auc": float("nan"), "brier": float("nan"),
                "recall_at_p50": float("nan"), "positive_rate": float(y_true.mean()), "n": int(len(y_true))}
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "recall_at_p50": recall_at_precision(y_true, proba, 0.5),
        "positive_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }


def score(kind: str, y_true, y_pred) -> Dict[str, float]:
    return classification_metrics(y_true, y_pred) if kind == "classification" else regression_metrics(y_true, y_pred)


def primary_metric(kind: str) -> Tuple[str, bool]:
    """(metrik adi, buyuk_olan_iyi_mi)"""
    return ("pr_auc", True) if kind == "classification" else ("mae", False)


# --------------------------------------------------------------------------- #
def leakage_check(kind: str, metrics: Dict[str, float], threshold: float = 0.98) -> Optional[str]:
    """Fazla iyi sonuc = neredeyse kesinlikle sizinti.

    Proses verisinde R2 > 0.98 gercekci degildir; hedefin bir turevi ozellik
    listesine sizmistir. Sessizce gecmek yerine yuksek sesle uyariyoruz.
    """
    value = metrics.get("roc_auc") if kind == "classification" else metrics.get("r2")
    if value is not None and np.isfinite(value) and value > threshold:
        return (
            f"SIZINTI SUPHESI: {'ROC-AUC' if kind == 'classification' else 'R2'} = {value:.4f} "
            f"(> {threshold}). Hedefin bir turevi ozellik listesine sizmis olabilir. "
            f"profile.yaml -> targets.<hedef>.exclude_features listesini gozden gecirin."
        )
    return None


# --------------------------------------------------------------------------- #
def top_features(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    kind: str,
    n: int = 20,
    n_repeats: int = 3,
    random_state: int = 42,
    max_rows: int = 4000,
) -> pd.DataFrame:
    """Permutation importance -- proses muhendisini ikna eden ciktinin kendisi.

    Agacin kendi `feature_importances_` degeri yuksek kardinaliteli kolonlara
    meyillidir; permutation, holdout uzerinde olculdugu icin daha durustur.
    """
    if len(X) > max_rows:
        idx = np.linspace(0, len(X) - 1, max_rows).astype(int)
        X, y = X.iloc[idx], y.iloc[idx]
    scoring = "average_precision" if kind == "classification" else "neg_mean_absolute_error"
    # n_jobs=1 BILINCLI: RandomForest ve HistGradientBoosting zaten kendi
    # icinde tum cekirdekleri kullanir. Buraya da -1 vermek IC ICE PARALELLIK
    # yaratir (N surec x N is parcacigi) ve hizlandirmak yerine yavaslatir --
    # 8 cekirdekli bir makinede fark dakikalarla olculur.
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring, n_jobs=1
    )
    frame = pd.DataFrame(
        {"feature": X.columns, "importance": result.importances_mean, "std": result.importances_std}
    )
    return frame.sort_values("importance", ascending=False).head(n).reset_index(drop=True)


def partial_dependence_curve(
    model: Any, X: pd.DataFrame, feature: str, kind: str, grid: int = 25, max_rows: int = 1500
) -> pd.DataFrame:
    """Tek degiskenli PDP: 'hiz 900'u gecince enerji tuketimi tirmaniyor'.

    Aksiyona donusebilen tek aciklanabilirlik ciktisi budur.
    """
    sample = X.sample(min(len(X), max_rows), random_state=0)
    lo, hi = np.percentile(X[feature], [2, 98])
    values = np.linspace(lo, hi, grid)
    means = []
    for value in values:
        probe = sample.copy()
        probe[feature] = value
        pred = model.predict_proba(probe)[:, 1] if kind == "classification" else model.predict(probe)
        means.append(float(np.mean(pred)))
    return pd.DataFrame({feature: values, "prediction": means})


def format_comparison(rows: List[Dict[str, Any]], kind: str) -> pd.DataFrame:
    """Model karsilastirma tablosu -- sunumun en ikna edici tek gorseli."""
    frame = pd.DataFrame(rows)
    metric, higher_better = primary_metric(kind)
    if metric in frame.columns:
        frame = frame.sort_values(metric, ascending=not higher_better)
    cols = ["model"] + [c for c in ("mae", "rmse", "mape", "r2", "pr_auc", "roc_auc",
                                    "recall_at_p50", "brier") if c in frame.columns]
    return frame[cols].reset_index(drop=True)
