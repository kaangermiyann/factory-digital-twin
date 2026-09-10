"""Model egitimi.

    python -m twin.models.train                    # profildeki tum hedefler
    python -m twin.models.train --targets sec_total_kwh_t

Ilkeler:
  * ZAMAN SERISINDE `train_test_split` YOK. TimeSeriesSplit + gap kullanilir;
    gap, prosesin olu zamani kadardir (geleceğin gecmise sizmasini engeller).
  * Son %20 zaman dilimi HOLDOUT'tur, CV'de hic gorulmez.
  * Her hedef icin naive/dogrusal/agac modelleri BIRLIKTE egitilir. "RF iyi"
    demek zayiftir; "naive 4.1, ridge 2.8, RF 1.6" demek modelin gercekten bir
    sey ogrendigini kanitlar.
  * `persistence` (son degeri tekrarla) taban cizgisi bilerek raporlanir:
    genelde cok iyi skor verir ama OPTIMIZASYONDA KULLANILAMAZ -- "buhari
    dusursem ne olur?" sorusuna cevap veremez. Bu ayrimi sunumda yapmak gerekir.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from twin.config import Profile, get_profile, get_settings
from twin.features.build import build_feature_table, make_supervised
from twin.models import evaluate as ev
from twin.models.registry import (
    ModelBundle,
    compute_feature_stats,
    current_environment,
    new_version,
    save_bundle,
)

log = logging.getLogger("twin.train")


# --------------------------------------------------------------------------- #
def build_candidates(kind: str, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    rf_cfg = dict(cfg.get("random_forest", {}))
    gb_cfg = dict(cfg.get("gradient_boosting", {}))

    if kind == "classification":
        return {
            "naive": DummyClassifier(strategy="prior"),
            "logistic": make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 400),
                max_depth=rf_cfg.get("max_depth"),
                min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
                max_features=rf_cfg.get("max_features", 0.5),
                class_weight="balanced_subsample",
                n_jobs=rf_cfg.get("n_jobs", -1),
                random_state=seed,
            ),
            "gradient_boosting": HistGradientBoostingClassifier(
                max_iter=gb_cfg.get("max_iter", 400),
                learning_rate=gb_cfg.get("learning_rate", 0.06),
                max_leaf_nodes=gb_cfg.get("max_leaf_nodes", 31),
                early_stopping=gb_cfg.get("early_stopping", True),
                random_state=seed,
            ),
        }
    return {
        "naive": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=rf_cfg.get("n_estimators", 400),
            max_depth=rf_cfg.get("max_depth"),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 5),
            max_features=rf_cfg.get("max_features", 0.5),
            n_jobs=rf_cfg.get("n_jobs", -1),
            random_state=seed,
        ),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=gb_cfg.get("max_iter", 400),
            learning_rate=gb_cfg.get("learning_rate", 0.06),
            max_leaf_nodes=gb_cfg.get("max_leaf_nodes", 31),
            early_stopping=gb_cfg.get("early_stopping", True),
            random_state=seed,
        ),
    }


TREE_MODELS = ("random_forest", "gradient_boosting")


def _apply_optimizer_safe_tiebreak(
    candidates: List[Dict[str, Any]], metric: str, higher_better: bool, tolerance: float
) -> Tuple[Dict[str, Any], Optional[str]]:
    """En iyi modeli sec; beraberlik halinde AGAC modelini tercih et.

    Bu model sadece bir tahminci degil, OPTIMIZASYONUN UZERINDE ARAMA YAPTIGI
    vekil yuzeydir. Fark burada kritik:

      * Agac modeli egitim destegi disinda DOYUMA gider -> cozucu uydurulmus
        bir kazanc bulamaz.
      * Dogrusal model sinirsiz ekstrapole eder. Her degisken tek tek emniyet
        kutusunun icinde olsa bile BIRLESIMLERI hic gorulmemis olabilir; ridge
        butun katsayilari toplar ve kendinden emin bir sacmalik uretir
        ("%10 enerji tasarrufu") -- hicbir hata mesaji vermeden.

    Odenen bedel binde birkac dogruluk; kazanilan, guvenle optimize edilebilen
    bir yuzey. Fark esigi asiyorsa dogruluk feda EDILMEZ.
    """
    def key(row: Dict[str, Any]) -> float:
        return row.get(metric, -np.inf if higher_better else np.inf)

    pick = max if higher_better else min
    best = pick(candidates, key=key)
    if best["model"] in TREE_MODELS:
        return best, None

    trees = [r for r in candidates if r["model"] in TREE_MODELS]
    if not trees:
        return best, None

    best_tree = pick(trees, key=key)
    gap = abs(key(best_tree) - key(best)) / max(abs(key(best)), 1e-9)
    if gap > tolerance:
        return best, None

    note = (
        f"Optimizasyon-guvenli secim: {best['model']} ({metric}={key(best):.4f}) yerine "
        f"{best_tree['model']} ({metric}={key(best_tree):.4f}) secildi -- fark %{gap * 100:.1f}, "
        f"esik %{tolerance * 100:.0f}. Agac modeli egitim destegi disinda doyuma gider; "
        f"dogrusal model sinirsiz ekstrapole edip sahte kazanc uretir."
    )
    return best_tree, note


def selected_model_names(kind: str, cfg: Dict[str, Any]) -> List[str]:
    """Konfigurasyondaki `models` listesini bu problem tipine cevirir.

    Dogrusal referansin adi problem tipine gore degisir: regresyonda `ridge`,
    siniflandirmada `logistic`. Konfigurasyonda ikisinden biri yazilmissa
    digerini de kabul ediyoruz -- aksi halde siniflandirma modelleri SESSIZCE
    dogrusal taban cizgisi olmadan egitilir ve "modelimiz dogrusaldan iyi"
    iddiasi kanitsiz kalir.
    """
    wanted = set(cfg.get("models", ["naive", "ridge", "random_forest", "gradient_boosting"]))
    if wanted & {"ridge", "logistic", "linear"}:
        wanted |= {"ridge", "logistic"}
    return [name for name in build_candidates(kind, cfg, 0) if name in wanted]


def _predict(model: Any, X: pd.DataFrame, kind: str) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if kind == "classification" else model.predict(X)


def cross_validate(
    model: Any, X: pd.DataFrame, y: pd.Series, kind: str, n_splits: int, gap: int
) -> Dict[str, float]:
    """Ileri dogru zincirli CV. `gap`, olu zaman kadar bosluk birakir."""
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores: List[Dict[str, float]] = []
    for train_idx, val_idx in splitter.split(X):
        y_train = y.iloc[train_idx]
        if kind == "classification" and y_train.nunique() < 2:
            continue
        fitted = model.fit(X.iloc[train_idx], y_train)
        scores.append(ev.score(kind, y.iloc[val_idx], _predict(fitted, X.iloc[val_idx], kind)))
    if not scores:
        return {}
    return {k: float(np.nanmean([s.get(k, np.nan) for s in scores])) for k in scores[0]}


def persistence_baseline(y_holdout: pd.Series, y_last_train: float, kind: str) -> Dict[str, float]:
    """Bir onceki olculen degeri tekrarla.

    Neden raporluyoruz: cok iyi skor verir ve modelin 'iyi' gorunmesini
    engeller -- ama karsi-olgusal soru soramaz, yani optimize edilemez.
    """
    pred = y_holdout.shift(1)
    pred.iloc[0] = y_last_train
    return ev.score(kind, y_holdout.values, pred.values)


# --------------------------------------------------------------------------- #
def train_target(
    frame: pd.DataFrame,
    profile: Profile,
    target: str,
    settings: Optional[Any] = None,
) -> Tuple[Optional[ModelBundle], pd.DataFrame, List[str]]:
    settings = settings or get_settings()
    cfg = settings.get_path("training", {}) or {}
    spec = profile.targets[target]
    kind = spec.kind
    seed = int(cfg.get("random_state", 42))
    notes: List[str] = []

    X, y, features = make_supervised(frame, profile, target)
    if len(X) < 500:
        log.warning("[%s] Yetersiz veri (%d satir) -- atlaniyor", target, len(X))
        return None, pd.DataFrame(), ["Yetersiz veri"]
    if kind == "classification" and y.nunique() < 2:
        log.warning("[%s] Tek sinif var -- atlaniyor", target)
        return None, pd.DataFrame(), ["Tek sinif"]

    # --- holdout: son %20 zaman dilimi, CV'de hic gorulmez ------------------ #
    holdout_frac = float(cfg.get("holdout_fraction", 0.2))
    cut = int(len(X) * (1 - holdout_frac))
    X_train, X_hold = X.iloc[:cut], X.iloc[cut:]
    y_train, y_hold = y.iloc[:cut], y.iloc[cut:]

    step_min = int(pd.Timedelta(settings.get_path("features.resample", "5min")).total_seconds() // 60)
    dead_time = int(settings.get_path("features.process_dead_time_min", 15))
    gap = max(1, dead_time // step_min)

    log.info("[%s] egitim=%d holdout=%d ozellik=%d gap=%d adim",
             target, len(X_train), len(X_hold), len(features), gap)

    rows: List[Dict[str, Any]] = []
    fitted_models: Dict[str, Any] = {}
    wanted = set(selected_model_names(kind, cfg))

    for name, model in build_candidates(kind, cfg, seed).items():
        if name not in wanted:
            continue
        cv = cross_validate(model, X_train, y_train, kind, int(cfg.get("n_splits", 4)), gap)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        holdout = ev.score(kind, y_hold.values, _predict(model, X_hold, kind))
        rows.append({"model": name, **holdout, **{f"cv_{k}": v for k, v in cv.items()}})
        log.info("  %-18s %s", name, {k: round(v, 4) for k, v in holdout.items() if k != "n"})

    rows.append({"model": "persistence*", **persistence_baseline(y_hold, float(y_train.iloc[-1]), kind)})

    comparison = ev.format_comparison(rows, kind)
    metric, higher_better = ev.primary_metric(kind)

    # persistence taban cizgisi kazanan olarak SECILEMEZ: iyi skor verir ama
    # karsi-olgusal soru soramaz, yani optimize edilemez.
    selectable = [r for r in rows if r["model"] in fitted_models]
    best, tiebreak_note = _apply_optimizer_safe_tiebreak(
        selectable, metric, higher_better, float(cfg.get("optimizer_safe_tolerance", 0.03))
    )
    if tiebreak_note:
        notes.append(tiebreak_note)
        log.info("[%s] %s", target, tiebreak_note)

    best_name = best["model"]
    best_model = fitted_models[best_name]
    log.info("[%s] secilen model: %s (%s=%.4f)", target, best_name, metric, best[metric])

    warning = ev.leakage_check(kind, best)
    if warning:
        log.warning("[%s] %s", target, warning)
        notes.append(warning)

    naive_row = next((r for r in rows if r["model"] == "naive"), None)
    if naive_row and metric in naive_row and np.isfinite(naive_row.get(metric, np.nan)):
        lift = (
            (best[metric] - naive_row[metric]) / max(abs(naive_row[metric]), 1e-9)
            if higher_better
            else (naive_row[metric] - best[metric]) / max(abs(naive_row[metric]), 1e-9)
        )
        notes.append(f"Naive taban cizgisine gore iyilesme: {lift * 100:.1f}%")

    importance = ev.top_features(best_model, X_hold, y_hold, kind, n=20, random_state=seed)

    bundle = ModelBundle(
        target=target,
        kind=kind,
        model=best_model,
        features=features,
        version=new_version(),
        profile=profile.name,
        line_id=profile.line_id,
        algorithm=best_name,
        metrics={
            "holdout": {k: v for k, v in best.items() if k != "model"},
            "comparison": comparison.to_dict(orient="records"),
            "top_features": importance.to_dict(orient="records"),
            "notes": notes,
        },
        feature_stats=compute_feature_stats(X),
        train_start=str(X.index.min()),
        train_end=str(X.index.max()),
        holdout_start=str(X_hold.index.min()) if len(X_hold) else None,
        environment=current_environment(),
        n_rows=len(X),
    )
    bundle.attach_support(X_train)
    return bundle, comparison, notes


# --------------------------------------------------------------------------- #
FAST_OVERRIDES = {
    "n_splits": 2,
    "random_forest": {"n_estimators": 120, "min_samples_leaf": 8, "max_features": 0.4, "n_jobs": -1},
    "gradient_boosting": {"max_iter": 150, "learning_rate": 0.1, "max_leaf_nodes": 31,
                          "early_stopping": True},
}


def train_all(
    targets: Optional[List[str]] = None,
    days: Optional[int] = None,
    profile_name: Optional[str] = None,
    fast: bool = False,
) -> Dict[str, Dict[str, Any]]:
    settings = get_settings()
    if fast:
        # Gelistirme dongusu icin: daha az agac, daha az fold. Metrikler biraz
        # kotuleşir -- MUSTERIYE GOSTERILECEK SAYILAR BU MODLA URETILMEMELI.
        cfg = dict(settings.get_path("training", {}) or {})
        cfg.update(FAST_OVERRIDES)
        settings["training"] = cfg
        log.warning("HIZLI MOD: metrikler tam egitimden daha kotudur, rapora konmamalidir.")
    profile = get_profile(profile_name)
    start = datetime.now(timezone.utc) - timedelta(days=days) if days else None

    log.info("Feature tablosu olusturuluyor...")
    frame = build_feature_table(start=start, profile=profile)
    if frame.empty:
        log.error("Veri yok. Once: python -m twin.simulator.run history --days 90")
        return {}

    results: Dict[str, Dict[str, Any]] = {}
    for target in (targets or list(profile.targets)):
        if target not in profile.targets:
            log.warning("Profilde tanimsiz hedef: %s", target)
            continue
        log.info("=" * 72)
        log.info("HEDEF: %s -- %s", target, profile.targets[target].desc)
        bundle, comparison, notes = train_target(frame, profile, target, settings)
        if bundle is None:
            continue
        save_bundle(bundle)
        results[target] = {
            "version": bundle.version,
            "algorithm": bundle.algorithm,
            "holdout": bundle.metrics["holdout"],
            "comparison": comparison,
            "notes": notes,
        }
    return results


def print_report(results: Dict[str, Dict[str, Any]]) -> None:
    for target, res in results.items():
        print(f"\n{'=' * 78}\n  {target}   ->  secilen: {res['algorithm']}  (v{res['version']})\n{'=' * 78}")
        print(res["comparison"].round(4).to_string(index=False))
        for note in res["notes"]:
            print(f"  ! {note}")
    print("\n(*) persistence = 'son olculen degeri tekrarla'.")
    print("    Regresyonda: iyi skor verir ama karsi-olgusal soru soramaz")
    print("    ('buhari dusursem ne olur?') -> OPTIMIZASYONDA KULLANILAMAZ.")
    print("    Siniflandirmada: etiket otokorelasyonunu olcer. Devam eden bir")
    print("    alarmi surdurur ama ILK alarmi hicbir zaman veremez -- yani")
    print("    erken uyari degeri sifirdir. Yuksek PR-AUC'si yaniltici olmasin.")


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Dijital ikiz model egitimi")
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--days", type=int, default=None, help="Son N gunun verisiyle egit")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Hizli gelistirme modu (az agac/fold) -- rapor icin kullanmayin")
    args = parser.parse_args(argv)

    results = train_all(args.targets, args.days, args.profile, args.fast)
    if not results:
        return 1
    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
