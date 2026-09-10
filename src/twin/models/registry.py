"""Model kayit defteri.

Her egitim, disk uzerinde surumlenmis bir "paket" (bundle) uretir:

    data/models/<profil>/<hedef>/<surum>/
        model.joblib          egitilmis tahminci
        meta.json             ozellik listesi, metrikler, egitim araligi, veri zarfi

`meta.json` icindeki `feature_stats` optimizasyon icin kritiktir: RandomForest
egitim araliginin disina ekstrapole EDEMEZ, sadece sabit deger basar. Arama
uzayini bu zarfla kesmezsek optimizasyon "buhari kapat, enerji sifir" gibi
fiziksel olmayan cozumler onerir.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from twin.config import get_settings, resolve_path

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Egitilmis model + onu guvenle kullanmak icin gereken her sey."""

    target: str
    kind: str                       # regression | classification
    model: Any
    features: List[str]
    version: str
    profile: str
    line_id: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    # Holdout'un basladigi an. Dashboard "gercek vs tahmin" grafigini SADECE bu
    # andan sonrasi icin cizer: modelin egitildigi donemde iyi uyum gostermesi
    # zaten beklenir ve hicbir sey kanitlamaz.
    holdout_start: Optional[str] = None
    n_rows: int = 0
    algorithm: str = ""

    # -- tahmin -------------------------------------------------------------- #
    def _median(self, feature: str) -> float:
        return float(self.feature_stats.get(feature, {}).get("median", 0.0))

    def align(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Bir tahmin cercevesini bu modelin bekledigi hale getirir.

        Uc is yapar:
          1. Eksik kolonlari egitim medyaniyla ekler.
          2. Kolon sirasini egitimdekine sabitler -- sklearn farkli siralamayi
             sessizce kabul eder ve YANLIS tahmin uretir.
          3. Kalan NaN'lari egitim medyaniyla doldurur.

        (3) opsiyonel degil: egitimde NaN'li satirlar atiliyor ama CANLI veride
        atma lüksü yok (lag'i henuz dolmamis bir kolon, gelmemis bir lab sonucu,
        anlik sensor bosluğu). Agac tabanli modeller NaN'i sindirir, dogrusal
        modeller PATLAR -- yani bu hata ancak dogrusal model secildiginde ortaya
        cikar ve tam da demo sirasinda cikar. Doldurma burada, tek yerde yapilir.
        """
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            frame = frame.assign(**{f: self._median(f) for f in missing})
        X = frame[self.features]
        if X.isna().to_numpy().any():
            X = X.fillna(pd.Series({f: self._median(f) for f in self.features}))
        return X

    def imputed_features(self, frame: pd.DataFrame) -> List[str]:
        """Hangi ozellikler doldurularak tahmin edildi?

        Sessiz doldurma, bozuk bir sensoru gizleyebilir. Cagiran taraf bunu
        raporlayabilsin diye ayri bir sorgu olarak duruyor.
        """
        present = [f for f in self.features if f in frame.columns]
        absent = [f for f in self.features if f not in frame.columns]
        nan_cols = [f for f in present if frame[f].isna().any()]
        return sorted(absent + nan_cols)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        X = self.align(frame)
        if self.kind == "classification":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    # -- guven -------------------------------------------------------------- #
    def in_domain(
        self,
        frame: pd.DataFrame,
        tolerance: float = 0.0,
        columns: Optional[List[str]] = None,
    ) -> pd.Series:
        """Satirlar egitim verisinin gordugu araligin icinde mi?

        `columns` verilirse sadece o kolonlar kontrol edilir. Tum ozellik
        listesini birden istemek pratikte her zaman False dondurur
        (0.98^n, n=190 -> ~%2); anlamli kontrol, degistirdigimiz
        degiskenler uzerinde yapilir.
        """
        X = self.align(frame)
        checked = [c for c in (columns or X.columns) if c in X.columns]
        ok = pd.Series(True, index=X.index)
        for col in checked:
            stats = self.feature_stats.get(col)
            if not stats:
                continue
            span = max(stats["p99"] - stats["p01"], 1e-9)
            lo = stats["p01"] - tolerance * span
            hi = stats["p99"] + tolerance * span
            ok &= X[col].between(lo, hi)
        return ok

    # -- yogunluk (joint support) ------------------------------------------ #
    #
    # Kutu kontrolu (in_domain) yetmez: her degisken tek tek araliginda olup
    # BIRLESIMLERI hic gorulmemis olabilir. Gercek soru sudur:
    #   "Onerdigim calisma noktasina BENZER bir noktayi egitimde kac kez gordum?"
    #
    # Onceki surumde bu, agacin yaprak sayimlariyla yapiliyordu. Sorun: ne
    # HistGradientBoosting ne de Ridge `.apply()` sunar -- kontrol SESSIZCE
    # devre disi kalir ve sistem "bol gozlenmis" diye rapor eder. Olculmeyen
    # bir seyi "yuksek" diye raporlamak, hic raporlamamaktan kotudur.
    #
    # Bunun yerine model-BAGIMSIZ bir komsuluk sayimi: egitim tasariminin bir
    # alt ornegi saklanir, aday nokta ile normalize edilmis maks-norm mesafesi
    # olculur. Her tahminci turuyle calisir.

    def attach_support(self, X_train: pd.DataFrame, max_rows: int = 2000) -> None:
        """Egitim tasarimindan temsili bir alt ornek sakla (yogunluk kontrolu icin).

        Sadece ham degiskenler saklanir (lag/rolling kopyalari degil) -- hem
        yer kazandirir hem de "benzer rejim" sorusunun fiziksel karsiligi budur.
        Alt ornek esit araliklarla secilir ki tum zaman araligini temsil etsin.
        """
        columns = [c for c in X_train.columns if "__" not in c]
        if not columns:
            return
        frame = X_train[columns]
        if len(frame) > max_rows:
            frame = frame.iloc[np.linspace(0, len(frame) - 1, max_rows).astype(int)]
        spans = {}
        for col in columns:
            stats = self.feature_stats.get(col, {})
            span = float(stats.get("p99", 0.0)) - float(stats.get("p01", 0.0))
            spans[col] = span if span > 1e-9 else 1.0
        self.support_columns = columns
        self.support_sample = frame.to_numpy(dtype=float)
        self.support_scale = np.array([spans[c] for c in columns], dtype=float)

    def support(self, frame: pd.DataFrame, radius: float = 0.15) -> np.ndarray:
        """Her satir icin: egitimde kac benzer nokta var?

        Benzerlik, degisken araliklarina gore normalize edilmis maks-norm
        mesafesidir; `radius=0.15` "her degiskende araligin %15'i kadar
        yakinlik" demektir. Olcum yapilamiyorsa NaN doner -- 0 degil, cunku
        "hic benzeri yok" ile "olcemedim" ayni sey degildir.
        """
        sample = getattr(self, "support_sample", None)
        columns = getattr(self, "support_columns", None)
        if sample is None or not columns:
            return np.full(len(frame), np.nan)
        usable = [c for c in columns if c in frame.columns]
        if not usable:
            return np.full(len(frame), np.nan)
        index = [columns.index(c) for c in usable]
        scale = self.support_scale[index]
        train = sample[:, index]
        query = frame[usable].to_numpy(dtype=float)

        counts = np.empty(len(query), dtype=float)
        for i, point in enumerate(query):
            distance = np.max(np.abs(train - point) / scale, axis=1)
            counts[i] = float(np.sum(distance <= radius))
        return counts


# --------------------------------------------------------------------------- #
def model_root() -> Path:
    return resolve_path(get_settings().get_path("serving.model_dir", "data/models"))


def new_version() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def compute_feature_stats(X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    described = X.describe(percentiles=[0.01, 0.5, 0.99]).T
    return {
        str(col): {
            "min": float(row["min"]),
            "p01": float(row["1%"]),
            "median": float(row["50%"]),
            "p99": float(row["99%"]),
            "max": float(row["max"]),
        }
        for col, row in described.iterrows()
    }


def save_bundle(bundle: ModelBundle) -> Path:
    path = model_root() / bundle.profile / bundle.target / bundle.version
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": bundle.model,
            "support_sample": getattr(bundle, "support_sample", None),
            "support_columns": getattr(bundle, "support_columns", None),
            "support_scale": getattr(bundle, "support_scale", None),
        },
        path / "model.joblib",
        compress=3,
    )
    meta = {
        "target": bundle.target,
        "kind": bundle.kind,
        "features": bundle.features,
        "version": bundle.version,
        "profile": bundle.profile,
        "line_id": bundle.line_id,
        "algorithm": bundle.algorithm,
        "metrics": bundle.metrics,
        "feature_stats": bundle.feature_stats,
        "train_start": bundle.train_start,
        "train_end": bundle.train_end,
        "holdout_start": bundle.holdout_start,
        "n_rows": bundle.n_rows,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (path.parent / "latest.txt").write_text(bundle.version, encoding="utf-8")
    log.info("Model kaydedildi: %s", path)
    return path


def load_bundle(target: str, profile: Optional[str] = None, version: Optional[str] = None) -> ModelBundle:
    profile = profile or get_settings().get_path("profile", "paper")
    base = model_root() / profile / target
    if version is None:
        pointer = base / "latest.txt"
        if not pointer.exists():
            raise FileNotFoundError(f"Egitilmis model yok: {base} (once `make train`)")
        version = pointer.read_text(encoding="utf-8").strip()
    path = base / version
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    payload = joblib.load(path / "model.joblib")

    bundle = ModelBundle(
        target=meta["target"],
        kind=meta["kind"],
        model=payload["model"],
        features=meta["features"],
        version=meta["version"],
        profile=meta["profile"],
        line_id=meta["line_id"],
        metrics=meta.get("metrics", {}),
        feature_stats=meta.get("feature_stats", {}),
        train_start=meta.get("train_start"),
        train_end=meta.get("train_end"),
        holdout_start=meta.get("holdout_start"),
        n_rows=meta.get("n_rows", 0),
        algorithm=meta.get("algorithm", ""),
    )
    if payload.get("support_sample") is not None:
        bundle.support_sample = payload["support_sample"]
        bundle.support_columns = payload["support_columns"]
        bundle.support_scale = payload["support_scale"]
    return bundle


def load_all(profile: Optional[str] = None) -> Dict[str, ModelBundle]:
    """Egitilmis tum hedefleri yukler; eksik olanlari sessizce atlar."""
    profile = profile or get_settings().get_path("profile", "paper")
    root = model_root() / profile
    bundles: Dict[str, ModelBundle] = {}
    if not root.exists():
        return bundles
    for target_dir in sorted(root.iterdir()):
        if not (target_dir / "latest.txt").exists():
            continue
        try:
            bundles[target_dir.name] = load_bundle(target_dir.name, profile)
        except Exception as exc:  # pragma: no cover
            log.warning("Model yuklenemedi (%s): %s", target_dir.name, exc)
    return bundles


def list_versions(target: str, profile: Optional[str] = None) -> List[str]:
    profile = profile or get_settings().get_path("profile", "paper")
    base = model_root() / profile / target
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())
