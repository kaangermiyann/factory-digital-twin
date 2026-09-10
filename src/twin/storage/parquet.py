"""Parquet backend -- docker'siz lokal gelistirme ve demo icin.

Uretimde kullanilmaz; amaci "Elasticsearch ayakta degil" diye demo'nun
durmamasidir. Ayni Repository arayuzunu uygular.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from twin.config import get_settings, resolve_path
from twin.schema import DATASET_TIME_FIELD, Dataset
from twin.storage.base import Repository


class ParquetRepository(Repository):
    name = "parquet"

    def __init__(self, root: Optional[str] = None) -> None:
        settings = get_settings()
        self.root = resolve_path(root or settings.get_path("storage.parquet.root", "data/processed"))
        self._cache: Dict[str, tuple] = {}

    # -- yasam dongusu ------------------------------------------------------ #
    def _dir(self, dataset: Dataset) -> Path:
        return self.root / dataset.value

    def ensure_schema(self) -> None:
        for dataset in Dataset:
            self._dir(dataset).mkdir(parents=True, exist_ok=True)

    def health(self) -> Dict[str, Any]:
        """Satir sayilari parquet METADATA'sindan okunur.

        Dosyalari acip saymak 2.7M satirda saniyeler surer; metadata okumak
        milisaniye. Diger backend'lerle ayni sozlesme (`counts`) donulur ki
        dashboard hangi depoyla calistigini bilmek zorunda kalmasin.
        """
        import pyarrow.parquet as pq

        counts: Dict[str, int] = {}
        stats: Dict[str, Any] = {}
        for dataset in Dataset:
            files = sorted(self._dir(dataset).glob("*.parquet"))
            rows = 0
            for path in files:
                try:
                    rows += pq.ParquetFile(path).metadata.num_rows
                except Exception:  # bozuk/yarim yazilmis dosya sayimi durdurmasin
                    continue
            counts[dataset.value] = rows
            stats[dataset.value] = {
                "files": len(files),
                "rows": rows,
                "bytes": sum(f.stat().st_size for f in files),
            }
        return {"backend": "parquet", "root": str(self.root), "ok": True,
                "counts": counts, "datasets": stats}

    # -- yazma -------------------------------------------------------------- #
    def write(self, dataset: Dataset, rows: Sequence[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        frame = pd.DataFrame(list(rows))
        for col in _time_columns(frame):
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
        # Dict/list kolonlari parquet'e sigmaz -> JSON string'e cevir
        for col in frame.columns:
            if frame[col].map(lambda v: isinstance(v, (dict, list))).any():
                frame[col] = frame[col].map(_json)

        target = self._dir(dataset)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"part-{datetime.utcnow():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:8]}.parquet"
        frame.to_parquet(path, index=False)
        self._cache.pop(dataset.value, None)
        return len(frame)

    # -- okuma -------------------------------------------------------------- #
    def _load(self, dataset: Dataset) -> pd.DataFrame:
        files = sorted(self._dir(dataset).glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        signature = tuple((f.name, f.stat().st_mtime_ns) for f in files)
        cached = self._cache.get(dataset.value)
        if cached and cached[0] == signature:
            return cached[1]
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        field = DATASET_TIME_FIELD[dataset]
        if field in frame.columns:
            frame[field] = pd.to_datetime(frame[field], utc=True, errors="coerce")
            frame = frame.sort_values(field).reset_index(drop=True)
        self._cache[dataset.value] = (signature, frame)
        return frame

    def read(
        self,
        dataset: Dataset,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        frame = self._load(dataset)
        if frame.empty:
            return frame
        field = DATASET_TIME_FIELD[dataset]
        if field in frame.columns:
            if start is not None:
                frame = frame[frame[field] >= _utc(start)]
            if end is not None:
                frame = frame[frame[field] <= _utc(end)]
        frame = self._apply_filters(frame, filters)
        if columns:
            keep = [c for c in columns if c in frame.columns]
            frame = frame[keep]
        if limit:
            frame = frame.head(limit)
        return frame.reset_index(drop=True)

    def latest(
        self,
        dataset: Dataset,
        n: int = 1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        frame = self._load(dataset)
        if frame.empty:
            return frame
        frame = self._apply_filters(frame, filters)
        field = DATASET_TIME_FIELD[dataset]
        if field in frame.columns:
            frame = frame.sort_values(field, ascending=False)
        return frame.head(n).reset_index(drop=True)


def _utc(value: datetime) -> pd.Timestamp:
    """Naive ise UTC olarak yorumla, tz'li ise UTC'ye cevir.

    Cagiranlar bazen tz'li (depodan gelen) bazen naive (elle verilen) zaman
    gonderiyor; ikisini de kabul etmezsek karsilastirma calisma aninda patlar.
    """
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _time_columns(frame: pd.DataFrame) -> List[str]:
    """`ts` ve `*_ts` kolonlari -- hepsi UTC datetime olarak saklanir."""
    return [c for c in frame.columns if c == "ts" or c.endswith("_ts")]


def _json(value: Any) -> Any:
    import json

    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str)
    return value
