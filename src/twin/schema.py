"""Canonical veri semasi.

Depolama backend'i (Elasticsearch / MySQL / Parquet) ne olursa olsun, sistemin
tamami bu semaya konusur. Musteri verisi geldiginde SADECE ingest adaptoru
degisir; asagidaki yapilar sabit kalir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class Dataset(str, Enum):
    """Mantiksal veri kumeleri -> ES index / MySQL tablo adlari."""

    TELEMETRY = "telemetry"
    ENERGY = "energy"
    BATCHES = "batches"
    QUALITY = "quality"
    EVENTS = "events"
    PREDICTIONS = "predictions"
    RECOMMENDATIONS = "recommendations"
    QUARANTINE = "quarantine"

    @property
    def is_time_partitioned(self) -> bool:
        """Aylik index'e bolunenler (hacimli olanlar)."""
        return self in (
            Dataset.TELEMETRY,
            Dataset.ENERGY,
            Dataset.PREDICTIONS,
            Dataset.QUARANTINE,
        )


class Role(str, Enum):
    SETPOINT = "setpoint"
    MEASUREMENT = "measurement"
    CONTEXT = "context"


class Medium(str, Enum):
    ELECTRICITY = "electricity"
    STEAM = "steam"
    NATURAL_GAS = "natural_gas"
    COMPRESSED_AIR = "compressed_air"
    WATER = "water"


class EventType(str, Enum):
    DOWNTIME = "downtime"
    GRADE_CHANGE = "grade_change"
    ALARM = "alarm"
    MAINTENANCE = "maintenance"


@dataclass
class Record:
    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, datetime):
                out[key] = value.isoformat()
            elif isinstance(value, Enum):
                out[key] = value.value
        return out


# --------------------------------------------------------------------------- #
@dataclass
class TelemetryPoint(Record):
    """Tek bir sensor okumasi (long format)."""

    ts: datetime
    asset_id: str
    line_id: str
    tag: str
    variable: str          # canonical ad (profile.yaml tag eslemesiyle cozulmus)
    value: float
    unit: str = ""
    role: str = Role.MEASUREMENT.value
    quality: int = 100     # 0-100, historian veri kalite bayragi
    batch_id: Optional[str] = None


@dataclass
class EnergyReading(Record):
    ts: datetime
    meter_id: str
    asset_id: str
    line_id: str
    medium: str
    value: float
    unit: str = "kWh"
    batch_id: Optional[str] = None


@dataclass
class ProductionBatch(Record):
    """Kagit: reel. Celik: heat. Surekli proseslerde: zaman dilimi."""

    batch_id: str
    line_id: str
    product_code: str
    start_ts: datetime
    end_ts: Optional[datetime]
    produced_qty: float
    scrap_qty: float = 0.0
    uom: str = "ton"
    order_id: Optional[str] = None
    shift_id: Optional[int] = None
    crew: Optional[str] = None


@dataclass
class QualitySample(Record):
    sample_id: str
    batch_id: str
    ts: datetime
    line_id: str
    property: str
    value: float
    unit: str = ""
    spec_min: Optional[float] = None
    spec_max: Optional[float] = None
    passed: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.passed is None:
            lo = self.spec_min if self.spec_min is not None else float("-inf")
            hi = self.spec_max if self.spec_max is not None else float("inf")
            self.passed = bool(lo <= self.value <= hi)


@dataclass
class ProcessEvent(Record):
    event_id: str
    ts_start: datetime
    ts_end: Optional[datetime]
    line_id: str
    asset_id: str
    type: str
    category: str = "unplanned"     # planned | unplanned
    reason_code: Optional[str] = None
    reason_text: Optional[str] = None
    duration_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.duration_s is None and self.ts_end is not None:
            self.duration_s = (self.ts_end - self.ts_start).total_seconds()


@dataclass
class Prediction(Record):
    """Canli skorlama ciktisi. y_true sonradan geriye donuk doldurulur.

    Dashboard'daki 'canli vs tahmin' kiyaslamasi ve drift takibi bu kayittan beslenir.
    """

    ts: datetime
    line_id: str
    target: str
    model_name: str
    model_version: str
    y_pred: float
    y_true: Optional[float] = None
    horizon_min: int = 0
    features_hash: Optional[str] = None


@dataclass
class Recommendation(Record):
    """Optimizasyon ciktisi -- operatore gosterilen tavsiye."""

    ts: datetime
    line_id: str
    product_code: str
    current_setpoints: Dict[str, float]
    recommended_setpoints: Dict[str, float]
    predicted_current: Dict[str, float]
    predicted_optimized: Dict[str, float]
    saving_try_per_ton: float
    confidence: str = "medium"       # high | medium | low
    accepted: Optional[bool] = None
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Wide (model) tablosu icin ayrilmis kolon adlari
# --------------------------------------------------------------------------- #
TS = "ts"
LINE_ID = "line_id"
BATCH_ID = "batch_id"
PRODUCT_CODE = "product_code"
SHIFT_ID = "shift_id"
CREW = "crew"
DOWNTIME_FLAG = "downtime_flag"

RESERVED_COLUMNS = {TS, LINE_ID, BATCH_ID, PRODUCT_CODE, SHIFT_ID, CREW, DOWNTIME_FLAG}


DATASET_TIME_FIELD = {
    Dataset.TELEMETRY: "ts",
    Dataset.ENERGY: "ts",
    Dataset.BATCHES: "start_ts",
    Dataset.QUALITY: "ts",
    Dataset.EVENTS: "ts_start",
    Dataset.PREDICTIONS: "ts",
    Dataset.RECOMMENDATIONS: "ts",
    Dataset.QUARANTINE: "ts",
}
