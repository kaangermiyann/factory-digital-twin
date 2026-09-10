"""Canli skorlama.

Iki isi var:
  1. Guncel ozellik vektorunu uretmek (API ve dashboard bunu paylasir).
  2. Periyodik olarak tahmin uretip `twin-predictions` index'ine yazmak.

Ikinci is, dashboard'daki "canli vs tahmin" kiyaslamasinin ve model drift
takibinin kaynagidir. `y_true` bastan bostur; olculen deger geldiginde geriye
donuk doldurulur -- boylece "modelin gecen hafta ne kadar isabetliydi?"
sorusuna veri uzerinden cevap verilir.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from twin.config import Grade, Profile, get_profile, get_settings
from twin.features.build import DOWNTIME_FLAG, build_feature_table
from twin.models.registry import ModelBundle, load_all
from twin.schema import Dataset
from twin.storage import Repository, get_repository

log = logging.getLogger("twin.scorer")

# Lag/rolling ozellikleri icin gereken minimum gecmis
WARMUP_HOURS = 4


class LiveContext:
    """Depodan guncel ozellik tablosunu ceker ve kisa sure onbellekte tutar.

    Onbellek olmadan dashboard her etkilesimde 2.7M satirlik telemetriyi
    yeniden okur ve kullanilamaz hale gelir.
    """

    def __init__(self, profile: Optional[Profile] = None, repo: Optional[Repository] = None,
                 cache_seconds: int = 20) -> None:
        self.profile = profile or get_profile()
        self.repo = repo or get_repository()
        self.cache_seconds = cache_seconds
        self._cache: Optional[Tuple[float, pd.DataFrame]] = None

    def frame(self, hours: int = 6, force: bool = False) -> pd.DataFrame:
        now = time.time()
        if not force and self._cache and now - self._cache[0] < self.cache_seconds:
            cached = self._cache[1]
            if not cached.empty:
                span = (cached["ts"].max() - cached["ts"].min()).total_seconds() / 3600.0
                if span >= hours * 0.9:
                    return cached

        end = self.repo.max_ts(Dataset.TELEMETRY, filters={"line_id": self.profile.line_id})
        end = end or datetime.now(timezone.utc)
        start = end - timedelta(hours=hours + WARMUP_HOURS)
        frame = build_feature_table(start=start, end=end, profile=self.profile, repo=self.repo)
        self._cache = (now, frame)
        return frame

    def latest_row(self, hours: int = 6) -> Optional[pd.Series]:
        """Skorlanabilir en son satir (durus disinda, lag'leri dolmus)."""
        frame = self.frame(hours=hours)
        if frame.empty:
            return None
        running = frame[frame.get(DOWNTIME_FLAG, 0) == 0]
        if running.empty:
            return None
        row = running.iloc[-1]
        return row.rename(row["ts"])

    def current_grade(self, hours: int = 6) -> Tuple[Optional[str], Optional[Grade]]:
        frame = self.frame(hours=hours)
        if frame.empty or "product_code" not in frame.columns:
            return None, None
        code = str(frame["product_code"].iloc[-1])
        return code, self.profile.grades.get(code)


# --------------------------------------------------------------------------- #
def features_hash(frame: pd.DataFrame, features: List[str]) -> List[str]:
    """Her satirin ozellik vektoru icin kisa bir parmak izi.

    Ne ise yarar: bir tavsiye tartismaya acildiginda "bu tahmin tam olarak hangi
    girdilerle uretildi?" sorusunu cevaplar. Vektorize -- satir satir hash almak
    600 satirda bile gereksiz yere yavastir.
    """
    columns = [f for f in features if f in frame.columns]
    if not columns:
        return [""] * len(frame)
    digest = pd.util.hash_pandas_object(frame[columns].round(4), index=False)
    return [f"{int(v) & 0xFFFFFFFFFFFF:012x}" for v in digest]


def score_frame(
    frame: pd.DataFrame, bundles: Dict[str, ModelBundle], profile: Profile
) -> List[Dict[str, Any]]:
    """Bir ozellik tablosunun tamamini skorlar -> Prediction kayitlari."""
    if frame.empty:
        return []
    usable = frame[frame.get(DOWNTIME_FLAG, 0) == 0]
    if usable.empty:
        return []

    rows: List[Dict[str, Any]] = []
    for target, bundle in bundles.items():
        missing = [f for f in bundle.features if f not in usable.columns]
        if missing:
            log.debug("[%s] %d ozellik eksik, medyanla dolduruluyor", target, len(missing))
        predictions = bundle.predict(usable)
        truth = usable[target] if target in usable.columns else pd.Series(np.nan, index=usable.index)
        digests = features_hash(usable, bundle.features)
        for ts, y_pred, y_true, digest in zip(usable["ts"], predictions, truth, digests):
            rows.append(
                {
                    "ts": pd.Timestamp(ts).isoformat(),
                    "line_id": profile.line_id,
                    "target": target,
                    "model_name": bundle.algorithm,
                    "model_version": bundle.version,
                    "y_pred": float(y_pred),
                    "y_true": None if pd.isna(y_true) else float(y_true),
                    "horizon_min": int(profile.targets[target].horizon_min or 0),
                    "features_hash": digest,
                }
            )
    return rows


def backfill(hours: int = 24, profile_name: Optional[str] = None) -> int:
    """Gecmis veriyi topluca skorlar.

    Demo icin sart: dashboard acildiginda 'canli vs tahmin' grafiginin dolu
    olmasi gerekir, canli dongunun veri biriktirmesini beklemek yerine.
    """
    profile = get_profile(profile_name)
    repo = get_repository()
    bundles = load_all(profile.name)
    if not bundles:
        log.error("Egitilmis model yok. Once: python -m twin.models.train")
        return 0

    end = repo.max_ts(Dataset.TELEMETRY, filters={"line_id": profile.line_id}) or datetime.now(timezone.utc)
    start = end - timedelta(hours=hours + WARMUP_HOURS)
    frame = build_feature_table(start=start, end=end, profile=profile, repo=repo)
    rows = score_frame(frame, bundles, profile)
    written = repo.write(Dataset.PREDICTIONS, rows)
    log.info("%d tahmin yazildi (%s -> %s)", written, start, end)
    return written


def run_loop(interval_s: Optional[int] = None, profile_name: Optional[str] = None) -> None:
    settings = get_settings()
    interval = interval_s or int(settings.get_path("serving.scorer_interval_s", 60))
    profile = get_profile(profile_name)
    repo = get_repository()
    bundles = load_all(profile.name)
    if not bundles:
        log.error("Egitilmis model yok. Once: python -m twin.models.train")
        return

    context = LiveContext(profile, repo, cache_seconds=0)
    log.info("Skorlama dongusu basladi (%d sn). Hedefler: %s", interval, list(bundles))
    last_ts: Optional[pd.Timestamp] = None
    try:
        while True:
            frame = context.frame(hours=2, force=True)
            if not frame.empty:
                if last_ts is not None:
                    frame = frame[frame["ts"] > last_ts]
                rows = score_frame(frame, bundles, profile)
                if rows:
                    repo.write(Dataset.PREDICTIONS, rows)
                    last_ts = pd.Timestamp(max(r["ts"] for r in rows))
                    log.info("%d tahmin yazildi (son: %s)", len(rows), last_ts)
            time.sleep(interval)
    except KeyboardInterrupt:
        log.info("Durduruldu.")


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Canli skorlama")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--backfill", type=int, default=None, help="Son N saati topluca skorla ve cik")
    parser.add_argument("--interval", type=int, default=None)
    args = parser.parse_args(argv)

    if args.backfill:
        backfill(args.backfill, args.profile)
    else:
        run_loop(args.interval, args.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
