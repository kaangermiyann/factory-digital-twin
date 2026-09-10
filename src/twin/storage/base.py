"""Depolama soyutlamasi.

Butun sistem bu arayuze konusur. Elasticsearch, MySQL ve Parquet
implementasyonlari birbirinin yerine gecebilir; ust katmanlarda tek satir
degismez.
"""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from twin.schema import Dataset, Record


class Repository(abc.ABC):
    """Zaman serisi + olay deposu icin ortak arayuz."""

    name: str = "base"

    # -- yasam dongusu ------------------------------------------------------ #
    @abc.abstractmethod
    def ensure_schema(self) -> None:
        """Index/tablo/klasor yapisini olustur (idempotent)."""

    @abc.abstractmethod
    def health(self) -> Dict[str, Any]:
        """Baglanti durumu ve kaba istatistikler."""

    def close(self) -> None:  # pragma: no cover - cogu backend icin gereksiz
        return None

    # -- yazma -------------------------------------------------------------- #
    @abc.abstractmethod
    def write(self, dataset: Dataset, rows: Sequence[Dict[str, Any]]) -> int:
        """Sozluk listesi yazar, yazilan satir sayisini doner."""

    def write_records(self, dataset: Dataset, records: Iterable[Record]) -> int:
        rows = [r.to_dict() for r in records]
        return self.write(dataset, rows) if rows else 0

    def write_frame(self, dataset: Dataset, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        return self.write(dataset, frame.to_dict(orient="records"))

    # -- okuma -------------------------------------------------------------- #
    @abc.abstractmethod
    def read(
        self,
        dataset: Dataset,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Zaman araligi + esitlik filtreleriyle veri okur.

        `filters` degerleri skaler ya da liste olabilir (liste => IN sorgusu).
        """

    @abc.abstractmethod
    def latest(
        self,
        dataset: Dataset,
        n: int = 1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """En yeni n kaydi doner (ts azalan)."""

    def max_ts(self, dataset: Dataset, filters: Optional[Dict[str, Any]] = None) -> Optional[datetime]:
        frame = self.latest(dataset, n=1, filters=filters)
        if frame.empty:
            return None
        from twin.schema import DATASET_TIME_FIELD

        field = DATASET_TIME_FIELD[dataset]
        return pd.to_datetime(frame.iloc[0][field]).to_pydatetime()

    # -- yardimci ----------------------------------------------------------- #
    @staticmethod
    def _apply_filters(frame: pd.DataFrame, filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        if not filters:
            return frame
        mask = pd.Series(True, index=frame.index)
        for key, value in filters.items():
            if key not in frame.columns:
                continue
            if isinstance(value, (list, tuple, set)):
                mask &= frame[key].isin(list(value))
            else:
                mask &= frame[key] == value
        return frame[mask]
