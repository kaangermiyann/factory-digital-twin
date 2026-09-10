"""MySQL backend.

Iki rolu var:
  1. Master data (asset, sensor_registry, product, shift, tariff) -- dogal yeri.
  2. Musteri "sadece MySQL" derse zaman serisi deposu olarak da calisir.

Zaman serisi icin `ts` uzerinde index + aylik partition sart; buyuk hacimde
Elasticsearch'e gore yavastir, bu dosyanin ust kismindaki uyari kasitlidir.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from twin.config import get_settings
from twin.schema import DATASET_TIME_FIELD, Dataset
from twin.storage.base import Repository

log = logging.getLogger(__name__)

# Zaman serisi tablolari icin sema. JSON kolonu, sema evrimini (yeni tag)
# ALTER TABLE gerektirmeden karsilar -- ES'in dynamic mapping esdegeri.
DDL = {
    Dataset.TELEMETRY: """
        CREATE TABLE IF NOT EXISTS telemetry (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            ts DATETIME(3) NOT NULL,
            line_id VARCHAR(32) NOT NULL,
            asset_id VARCHAR(64) NOT NULL,
            tag VARCHAR(64) NOT NULL,
            variable VARCHAR(64) NOT NULL,
            value DOUBLE NULL,
            unit VARCHAR(16) NULL,
            role VARCHAR(16) NULL,
            quality SMALLINT DEFAULT 100,
            batch_id VARCHAR(64) NULL,
            INDEX ix_tel_ts (ts),
            INDEX ix_tel_line_var_ts (line_id, variable, ts),
            INDEX ix_tel_batch (batch_id)
        ) ENGINE=InnoDB ROW_FORMAT=COMPRESSED
    """,
    Dataset.ENERGY: """
        CREATE TABLE IF NOT EXISTS energy (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            ts DATETIME(3) NOT NULL,
            meter_id VARCHAR(64) NOT NULL,
            asset_id VARCHAR(64) NOT NULL,
            line_id VARCHAR(32) NOT NULL,
            medium VARCHAR(24) NOT NULL,
            value DOUBLE NULL,
            unit VARCHAR(16) NULL,
            batch_id VARCHAR(64) NULL,
            INDEX ix_en_ts (ts), INDEX ix_en_meter_ts (meter_id, ts)
        ) ENGINE=InnoDB
    """,
    Dataset.BATCHES: """
        CREATE TABLE IF NOT EXISTS batches (
            batch_id VARCHAR(64) PRIMARY KEY,
            line_id VARCHAR(32) NOT NULL,
            product_code VARCHAR(32) NOT NULL,
            start_ts DATETIME(3) NOT NULL,
            end_ts DATETIME(3) NULL,
            produced_qty DOUBLE NULL,
            scrap_qty DOUBLE NULL,
            uom VARCHAR(16) NULL,
            order_id VARCHAR(64) NULL,
            shift_id SMALLINT NULL,
            crew VARCHAR(16) NULL,
            INDEX ix_b_start (start_ts), INDEX ix_b_line (line_id)
        ) ENGINE=InnoDB
    """,
    Dataset.QUALITY: """
        CREATE TABLE IF NOT EXISTS quality (
            sample_id VARCHAR(64) PRIMARY KEY,
            batch_id VARCHAR(64) NOT NULL,
            ts DATETIME(3) NOT NULL,
            line_id VARCHAR(32) NOT NULL,
            property VARCHAR(64) NOT NULL,
            value DOUBLE NULL,
            unit VARCHAR(16) NULL,
            spec_min DOUBLE NULL,
            spec_max DOUBLE NULL,
            passed TINYINT(1) NULL,
            INDEX ix_q_batch (batch_id), INDEX ix_q_ts (ts)
        ) ENGINE=InnoDB
    """,
    Dataset.EVENTS: """
        CREATE TABLE IF NOT EXISTS events (
            event_id VARCHAR(64) PRIMARY KEY,
            ts_start DATETIME(3) NOT NULL,
            ts_end DATETIME(3) NULL,
            line_id VARCHAR(32) NOT NULL,
            asset_id VARCHAR(64) NOT NULL,
            type VARCHAR(24) NOT NULL,
            category VARCHAR(16) NULL,
            reason_code VARCHAR(32) NULL,
            reason_text VARCHAR(255) NULL,
            duration_s DOUBLE NULL,
            INDEX ix_e_start (ts_start), INDEX ix_e_line (line_id)
        ) ENGINE=InnoDB
    """,
    Dataset.PREDICTIONS: """
        CREATE TABLE IF NOT EXISTS predictions (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            ts DATETIME(3) NOT NULL,
            line_id VARCHAR(32) NOT NULL,
            target VARCHAR(64) NOT NULL,
            model_name VARCHAR(64) NOT NULL,
            model_version VARCHAR(32) NOT NULL,
            y_pred DOUBLE NULL,
            y_true DOUBLE NULL,
            horizon_min INT DEFAULT 0,
            features_hash VARCHAR(64) NULL,
            INDEX ix_p_ts (ts), INDEX ix_p_target_ts (target, ts)
        ) ENGINE=InnoDB
    """,
    Dataset.RECOMMENDATIONS: """
        CREATE TABLE IF NOT EXISTS recommendations (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            ts DATETIME(3) NOT NULL,
            line_id VARCHAR(32) NOT NULL,
            product_code VARCHAR(32) NULL,
            current_setpoints JSON NULL,
            recommended_setpoints JSON NULL,
            predicted_current JSON NULL,
            predicted_optimized JSON NULL,
            saving_try_per_ton DOUBLE NULL,
            confidence VARCHAR(16) NULL,
            accepted TINYINT(1) NULL,
            notes JSON NULL,
            INDEX ix_r_ts (ts)
        ) ENGINE=InnoDB
    """,
    Dataset.QUARANTINE: """
        CREATE TABLE IF NOT EXISTS quarantine (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            ts DATETIME(3) NULL,
            dataset VARCHAR(32) NULL,
            reason VARCHAR(64) NULL,
            payload JSON NULL,
            INDEX ix_qt_ts (ts)
        ) ENGINE=InnoDB
    """,
}

# Master data -- ES'te degil, burada durur.
MASTER_DDL = """
CREATE TABLE IF NOT EXISTS asset (
    asset_id VARCHAR(64) PRIMARY KEY,
    parent_id VARCHAR(64) NULL,
    line_id VARCHAR(32) NULL,
    type VARCHAR(32) NULL,
    description VARCHAR(255) NULL,
    nominal_capacity DOUBLE NULL,
    rated_power_kw DOUBLE NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS sensor_registry (
    tag VARCHAR(64) PRIMARY KEY,
    variable VARCHAR(64) NOT NULL,
    asset_id VARCHAR(64) NULL,
    description VARCHAR(255) NULL,
    unit VARCHAR(16) NULL,
    role VARCHAR(16) NOT NULL,
    low_limit DOUBLE NULL,
    high_limit DOUBLE NULL,
    op_low DOUBLE NULL,
    op_high DOUBLE NULL,
    max_rate_of_change DOUBLE NULL,
    sampling_period_s INT NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS product (
    product_code VARCHAR(32) PRIMARY KEY,
    description VARCHAR(255) NULL,
    attributes JSON NULL,
    margin_try_per_ton DOUBLE NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS shift_calendar (
    shift_id SMALLINT PRIMARY KEY,
    start_hour TINYINT NOT NULL,
    end_hour TINYINT NOT NULL,
    crew VARCHAR(16) NULL
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS energy_tariff (
    id INT AUTO_INCREMENT PRIMARY KEY,
    medium VARCHAR(24) NOT NULL,
    hour_of_day TINYINT NOT NULL,
    price_try_per_unit DOUBLE NOT NULL,
    valid_from DATE NULL
) ENGINE=InnoDB;
"""

JSON_COLUMNS = {
    Dataset.RECOMMENDATIONS: {
        "current_setpoints",
        "recommended_setpoints",
        "predicted_current",
        "predicted_optimized",
        "notes",
    },
    Dataset.QUARANTINE: {"payload"},
}


class MySQLRepository(Repository):
    name = "mysql"

    def __init__(self, **overrides: Any) -> None:
        from sqlalchemy import create_engine

        cfg = dict(get_settings().get_path("storage.mysql", {}) or {})
        cfg.update(overrides)
        url = (
            f"mysql+pymysql://{cfg['username']}:{cfg['password']}"
            f"@{cfg['host']}:{cfg['port']}/{cfg['database']}?charset=utf8mb4"
        )
        self.engine = create_engine(url, pool_size=int(cfg.get("pool_size", 5)), pool_pre_ping=True)

    # -- yasam dongusu ------------------------------------------------------ #
    def ensure_schema(self) -> None:
        from sqlalchemy import text

        with self.engine.begin() as conn:
            for ddl in DDL.values():
                conn.execute(text(ddl))
            for stmt in filter(None, (s.strip() for s in MASTER_DDL.split(";"))):
                conn.execute(text(stmt))
        log.info("MySQL semasi hazir")

    def health(self) -> Dict[str, Any]:
        from sqlalchemy import text

        try:
            counts = {}
            with self.engine.connect() as conn:
                for dataset in Dataset:
                    try:
                        counts[dataset.value] = conn.execute(
                            text(f"SELECT COUNT(*) FROM {dataset.value}")
                        ).scalar_one()
                    except Exception:
                        counts[dataset.value] = 0
            return {"backend": "mysql", "ok": True, "counts": counts}
        except Exception as exc:  # pragma: no cover
            return {"backend": "mysql", "ok": False, "error": str(exc)}

    def close(self) -> None:
        self.engine.dispose()

    # -- yazma -------------------------------------------------------------- #
    def write(self, dataset: Dataset, rows: Sequence[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        frame = pd.DataFrame(list(rows))
        for col in JSON_COLUMNS.get(dataset, set()):
            if col in frame.columns:
                frame[col] = frame[col].map(lambda v: json.dumps(v, default=str) if v is not None else None)
        field = DATASET_TIME_FIELD[dataset]
        if field in frame.columns:
            frame[field] = pd.to_datetime(frame[field], utc=True, errors="coerce").dt.tz_localize(None)
        frame.to_sql(dataset.value, self.engine, if_exists="append", index=False, chunksize=2000)
        return len(frame)

    # -- okuma -------------------------------------------------------------- #
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
        select = ", ".join(columns) if columns else "*"
        where: List[str] = []
        params: Dict[str, Any] = {}
        if start is not None:
            where.append(f"{field} >= :start")
            params["start"] = pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tzinfo else start
        if end is not None:
            where.append(f"{field} <= :end")
            params["end"] = pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo else end
        for i, (key, value) in enumerate((filters or {}).items()):
            if isinstance(value, (list, tuple, set)):
                names = []
                for j, item in enumerate(value):
                    names.append(f":f{i}_{j}")
                    params[f"f{i}_{j}"] = item
                where.append(f"{key} IN ({', '.join(names)})")
            else:
                where.append(f"{key} = :f{i}")
                params[f"f{i}"] = value

        sql = f"SELECT {select} FROM {dataset.value}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {field} ASC"
        if limit:
            sql += f" LIMIT {int(limit)}"

        from sqlalchemy import text

        frame = pd.read_sql(text(sql), self.engine, params=params)
        if not frame.empty and field in frame.columns:
            frame[field] = pd.to_datetime(frame[field], utc=True, errors="coerce")
        return self._decode_json(dataset, frame)

    def latest(
        self,
        dataset: Dataset,
        n: int = 1,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        field = DATASET_TIME_FIELD[dataset]
        where: List[str] = []
        params: Dict[str, Any] = {}
        for i, (key, value) in enumerate((filters or {}).items()):
            where.append(f"{key} = :f{i}")
            params[f"f{i}"] = value
        sql = f"SELECT * FROM {dataset.value}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {field} DESC LIMIT {int(n)}"

        from sqlalchemy import text

        frame = pd.read_sql(text(sql), self.engine, params=params)
        if not frame.empty and field in frame.columns:
            frame[field] = pd.to_datetime(frame[field], utc=True, errors="coerce")
        return self._decode_json(dataset, frame)

    @staticmethod
    def _decode_json(dataset: Dataset, frame: pd.DataFrame) -> pd.DataFrame:
        for col in JSON_COLUMNS.get(dataset, set()):
            if col in frame.columns:
                frame[col] = frame[col].map(lambda v: json.loads(v) if isinstance(v, str) else v)
        return frame
