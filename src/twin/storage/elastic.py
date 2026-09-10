"""Elasticsearch backend -- zaman serisi icin birincil depo.

Tasarim notlari:
  * Hacimli veri kumeleri aylik index'e bolunur (twin-telemetry-2026.01),
    okuma wildcard ile yapilir -> ILM ile eski aylar warm/frozen'a tasinir.
  * `tag`, `asset_id`, `variable` -> keyword (agregasyon icin sart)
  * `refresh_interval` yuksek tutulur; canli yazimda gereksiz refresh maliyetlidir.
  * Buyuk okumalar `search_after` + PIT ile sayfalanir (from/size 10k limiti yok).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from twin.config import get_settings
from twin.schema import DATASET_TIME_FIELD, Dataset
from twin.storage.base import Repository

log = logging.getLogger(__name__)

MAPPINGS_DIR = Path(__file__).parent / "mappings"


class ElasticRepository(Repository):
    name = "elastic"

    def __init__(self, **overrides: Any) -> None:
        from elasticsearch import Elasticsearch  # gecikmeli import: opsiyonel bagimlilik

        cfg = dict(get_settings().get_path("storage.elastic", {}) or {})
        cfg.update(overrides)
        self.cfg = cfg
        self.prefix = cfg.get("index_prefix", "twin")
        self.chunk_size = int(cfg.get("bulk_chunk_size", 5000))

        auth = None
        if cfg.get("username"):
            auth = (cfg["username"], cfg.get("password"))
        self.client = Elasticsearch(
            cfg.get("hosts", ["http://localhost:9200"]),
            basic_auth=auth,
            verify_certs=bool(cfg.get("verify_certs", False)),
            request_timeout=60,
        )

    # -- index adlandirma --------------------------------------------------- #
    def index_for(self, dataset: Dataset, ts: Optional[datetime] = None) -> str:
        base = f"{self.prefix}-{dataset.value}"
        if not dataset.is_time_partitioned:
            return base
        stamp = pd.Timestamp(ts or datetime.utcnow())
        return f"{base}-{stamp:%Y.%m}"

    def pattern_for(self, dataset: Dataset) -> str:
        base = f"{self.prefix}-{dataset.value}"
        return f"{base}-*" if dataset.is_time_partitioned else base

    # -- yasam dongusu ------------------------------------------------------ #
    def ensure_schema(self) -> None:
        """Her veri kumesi icin index template olusturur (idempotent)."""
        for dataset in Dataset:
            path = MAPPINGS_DIR / f"{dataset.value}.json"
            if not path.exists():
                log.warning("Mapping bulunamadi, dynamic mapping kullanilacak: %s", dataset.value)
                continue
            body = json.loads(path.read_text(encoding="utf-8"))
            body.setdefault("template", {}).setdefault("settings", {}).update(
                {
                    "number_of_shards": self.cfg.get("number_of_shards", 1),
                    "number_of_replicas": self.cfg.get("number_of_replicas", 0),
                    "refresh_interval": self.cfg.get("refresh_interval", "30s"),
                    "codec": "best_compression",
                }
            )
            body["index_patterns"] = [self.pattern_for(dataset)]
            self.client.indices.put_index_template(
                name=f"{self.prefix}-{dataset.value}-tpl", **body
            )
            log.info("Index template hazir: %s", dataset.value)

    def health(self) -> Dict[str, Any]:
        try:
            cluster = self.client.cluster.health()
            counts = {}
            for dataset in Dataset:
                try:
                    counts[dataset.value] = int(
                        self.client.count(index=self.pattern_for(dataset), ignore_unavailable=True)["count"]
                    )
                except Exception:  # index henuz yok
                    counts[dataset.value] = 0
            return {"backend": "elastic", "ok": True, "status": cluster["status"], "counts": counts}
        except Exception as exc:  # pragma: no cover
            return {"backend": "elastic", "ok": False, "error": str(exc)}

    def close(self) -> None:
        self.client.close()

    # -- yazma -------------------------------------------------------------- #
    def write(self, dataset: Dataset, rows: Sequence[Dict[str, Any]]) -> int:
        from elasticsearch.helpers import bulk

        if not rows:
            return 0
        field = DATASET_TIME_FIELD[dataset]

        def actions():
            for row in rows:
                ts = row.get(field)
                yield {"_index": self.index_for(dataset, pd.Timestamp(ts) if ts else None), "_source": row}

        written, errors = bulk(
            self.client, actions(), chunk_size=self.chunk_size, raise_on_error=False, stats_only=False
        )
        if errors:
            log.error("Bulk yazimda %d hata (ilk: %s)", len(errors), errors[0])
        return int(written)

    # -- okuma -------------------------------------------------------------- #
    @staticmethod
    def _build_query(
        time_field: str,
        start: Optional[datetime],
        end: Optional[datetime],
        filters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        must: List[Dict[str, Any]] = []
        if start or end:
            rng: Dict[str, Any] = {}
            if start:
                rng["gte"] = pd.Timestamp(start).isoformat()
            if end:
                rng["lte"] = pd.Timestamp(end).isoformat()
            must.append({"range": {time_field: rng}})
        for key, value in (filters or {}).items():
            if isinstance(value, (list, tuple, set)):
                must.append({"terms": {key: list(value)}})
            else:
                must.append({"term": {key: value}})
        return {"bool": {"must": must}} if must else {"match_all": {}}

    def read(
        self,
        dataset: Dataset,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        field = DATASET_TIME_FIELD[dataset]
        query = self._build_query(field, start, end, filters)
        page_size = min(limit or 10000, 10000)

        hits: List[Dict[str, Any]] = []
        search_after = None
        while True:
            body: Dict[str, Any] = {
                "query": query,
                "size": page_size,
                "sort": [{field: "asc"}, {"_doc": "asc"}],
            }
            if columns:
                body["_source"] = columns
            if search_after:
                body["search_after"] = search_after
            resp = self.client.search(
                index=self.pattern_for(dataset), ignore_unavailable=True, **body
            )
            page = resp["hits"]["hits"]
            if not page:
                break
            hits.extend(h["_source"] for h in page)
            if limit and len(hits) >= limit:
                hits = hits[:limit]
                break
            if len(page) < page_size:
                break
            search_after = page[-1]["sort"]

        frame = pd.DataFrame(hits)
        if not frame.empty and field in frame.columns:
            frame[field] = pd.to_datetime(frame[field], utc=True, errors="coerce")
        return frame

    def latest(
        self,
        dataset: Dataset,
        n: int = 1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        field = DATASET_TIME_FIELD[dataset]
        resp = self.client.search(
            index=self.pattern_for(dataset),
            ignore_unavailable=True,
            query=self._build_query(field, None, None, filters),
            size=n,
            sort=[{field: "desc"}],
        )
        frame = pd.DataFrame(h["_source"] for h in resp["hits"]["hits"])
        if not frame.empty and field in frame.columns:
            frame[field] = pd.to_datetime(frame[field], utc=True, errors="coerce")
        return frame

    # -- ES'e ozgu hizlandirma ---------------------------------------------- #
    def downsample(
        self,
        dataset: Dataset,
        interval: str,
        start: datetime,
        end: datetime,
        group_field: str = "variable",
        value_field: str = "value",
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Sunucu tarafinda date_histogram ile yeniden ornekleme.

        150M satirlik telemetriyi Python'a cekmek yerine ES'te ozetlemek
        buyuklukce daha hizlidir -- feature pipeline bunu kullanir.
        """
        field = DATASET_TIME_FIELD[dataset]
        resp = self.client.search(
            index=self.pattern_for(dataset),
            ignore_unavailable=True,
            size=0,
            query=self._build_query(field, start, end, filters),
            aggs={
                "by_var": {
                    "terms": {"field": group_field, "size": 500},
                    "aggs": {
                        "over_time": {
                            "date_histogram": {"field": field, "fixed_interval": interval},
                            "aggs": {
                                "avg": {"avg": {"field": value_field}},
                                "min": {"min": {"field": value_field}},
                                "max": {"max": {"field": value_field}},
                            },
                        }
                    },
                }
            },
        )
        rows = []
        for var_bucket in resp["aggregations"]["by_var"]["buckets"]:
            for time_bucket in var_bucket["over_time"]["buckets"]:
                rows.append(
                    {
                        "ts": pd.to_datetime(time_bucket["key_as_string"], utc=True),
                        group_field: var_bucket["key"],
                        "value": time_bucket["avg"]["value"],
                        "value_min": time_bucket["min"]["value"],
                        "value_max": time_bucket["max"]["value"],
                    }
                )
        return pd.DataFrame(rows)
