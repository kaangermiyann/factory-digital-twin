"""Depolama katmani fabrikasi.

    from twin.storage import get_repository
    repo = get_repository()          # settings.yaml -> storage.backend
    repo = get_repository("elastic") # acikca

Backend degistirmek, ust katmanlarda hicbir degisiklik gerektirmez.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from twin.config import get_settings
from twin.storage.base import Repository

__all__ = ["Repository", "get_repository", "build_repository"]


def build_repository(backend: Optional[str] = None, **kwargs) -> Repository:
    backend = (backend or get_settings().get_path("storage.backend", "parquet")).lower()

    if backend == "parquet":
        from twin.storage.parquet import ParquetRepository

        return ParquetRepository(**kwargs)
    if backend in ("elastic", "elasticsearch", "es"):
        from twin.storage.elastic import ElasticRepository

        return ElasticRepository(**kwargs)
    if backend == "mysql":
        from twin.storage.mysql import MySQLRepository

        return MySQLRepository(**kwargs)
    raise ValueError(f"Bilinmeyen backend: {backend!r} (parquet | elastic | mysql)")


@lru_cache(maxsize=4)
def get_repository(backend: Optional[str] = None) -> Repository:
    """Surec basina tekil repository (baglanti havuzu paylasilir)."""
    return build_repository(backend)
