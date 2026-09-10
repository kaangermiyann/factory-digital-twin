"""FastAPI servis katmani.

    uvicorn twin.serving.api:app --reload

Uc noktalar:
    GET  /health              sistem + model durumu
    GET  /models              egitilmis modeller ve holdout metrikleri
    GET  /kpis                canli ozet (SEC, nem, uretim, risk, maliyet)
    GET  /compare             gercek vs tahmin serisi (dashboard ana grafigi)
    GET  /drift               haftalik MAE takibi -- yeniden egitim tetigi
    POST /predict             verilen ozellik vektoru icin tum hedefler
    POST /whatif              "hizi 900 yaparsam ne olur?"
    POST /optimize            kisitli optimizasyon -> tavsiye edilen setpointler

NOT: Servis SALT OKUNURDUR. PLC/DCS'e hicbir sekilde yazmaz. Cikti operatore
tavsiye olarak gosterilir (advisory / acik dongu). Kapali dongu kontrol ayri
bir proje ve ayri bir emniyet onayidir.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from twin.config import get_profile, get_settings
from twin.models.registry import load_all
from twin.optimize.objective import Economics
from twin.optimize.search import apply_setpoints, optimize_row, what_if
from twin.schema import Dataset
from twin.serving.scorer import LiveContext, score_frame
from twin.storage import get_repository

log = logging.getLogger("twin.api")

app = FastAPI(
    title="Factory Digital Twin API",
    description="Fabrika optimizasyonu icin dijital ikiz -- tahmin, senaryo ve optimizasyon servisi",
    version="0.1.0",
)


class _State:
    """Sureç yasam dongusu boyunca paylasilan agir nesneler."""

    def __init__(self) -> None:
        self.profile = get_profile()
        self.settings = get_settings()
        self.repo = get_repository()
        self.bundles = load_all(self.profile.name)
        self.context = LiveContext(self.profile, self.repo)
        if not self.bundles:
            log.warning("Egitilmis model bulunamadi -- once `make train`")

    def require_models(self) -> Dict[str, Any]:
        if not self.bundles:
            raise HTTPException(503, "Egitilmis model yok. Once: python -m twin.models.train")
        return self.bundles

    def require_row(self, hours: int = 6) -> pd.Series:
        row = self.context.latest_row(hours=hours)
        if row is None:
            raise HTTPException(503, "Canli veri yok. Once: python -m twin.simulator.run history --days 90")
        return row


state = _State()


# --------------------------------------------------------------------------- #
# Semalar
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(..., description="Ozellik adi -> deger")
    targets: Optional[List[str]] = None


class WhatIfRequest(BaseModel):
    setpoints: Dict[str, float] = Field(..., description="Degistirilecek setpointler")
    hour: Optional[int] = Field(None, description="Tarife saati (varsayilan: simdi)")


class OptimizeRequest(BaseModel):
    variables: Optional[List[str]] = Field(None, description="Sadece bu setpointleri ara")
    respect_rate_limits: bool = Field(True, description="Degisim hizi limitlerini uygula")
    hour: Optional[int] = None


# --------------------------------------------------------------------------- #
# Uc noktalar
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> Dict[str, Any]:
    storage = state.repo.health()
    latest = state.repo.max_ts(Dataset.TELEMETRY, filters={"line_id": state.profile.line_id})
    return {
        "ok": bool(storage.get("ok")) and bool(state.bundles),
        "profile": state.profile.name,
        "line_id": state.profile.line_id,
        "storage": storage,
        "models": sorted(state.bundles),
        "latest_data_ts": latest.isoformat() if latest else None,
    }


@app.get("/models")
def models() -> Dict[str, Any]:
    return {
        target: {
            "algorithm": bundle.algorithm,
            "version": bundle.version,
            "kind": bundle.kind,
            "n_features": len(bundle.features),
            "n_rows": bundle.n_rows,
            "train_start": bundle.train_start,
            "train_end": bundle.train_end,
            "holdout": bundle.metrics.get("holdout", {}),
            "comparison": bundle.metrics.get("comparison", []),
            "top_features": bundle.metrics.get("top_features", [])[:10],
            "notes": bundle.metrics.get("notes", []),
        }
        for target, bundle in state.require_models().items()
    }


@app.get("/kpis")
def kpis(hours: int = Query(6, ge=1, le=168)) -> Dict[str, Any]:
    """Canli ozet: olculen degerler + model tahminleri + anlik maliyet."""
    bundles = state.require_models()
    row = state.require_row(hours)
    code, grade = state.context.current_grade(hours)

    frame = pd.DataFrame([row.values], columns=row.index)
    predicted = {t: float(b.predict(frame)[0]) for t, b in bundles.items()}
    # Doldurulan ozellikler seffaf olsun: sessiz imputasyon bozuk bir sensoru
    # gizleyebilir. Sayi buyurse veri kalitesine bakilmali.
    imputed = sorted({f for b in bundles.values() for f in b.imputed_features(frame)})
    measured = {t: (None if pd.isna(row.get(t)) else float(row.get(t)))
                for t in bundles if t in row.index}

    economics = Economics.from_settings(state.settings)
    hour = int(pd.Timestamp(row.name).hour)
    elec = float(row.get("elec_kwh_t", np.nan))
    thermal = float(row.get("thermal_kwh_t", np.nan))
    cost = (
        economics.energy_cost_try_per_ton(elec, thermal, hour)
        if np.isfinite(elec) and np.isfinite(thermal)
        else None
    )

    spec = grade.spec("reel_moisture_pct") if grade else None
    return {
        "ts": pd.Timestamp(row.name).isoformat(),
        "product_code": code,
        "grade_desc": grade.desc if grade else None,
        "shift_id": int(row.get("shift_id", 0)),
        "crew": row.get("crew"),
        "measured": measured,
        "predicted": predicted,
        "energy_cost_try_per_ton": round(cost, 2) if cost else None,
        "electricity_price_try_per_kwh": economics.electricity_price(hour),
        "quality_spec": {"target": "reel_moisture_pct", "low": spec[0], "high": spec[1]} if spec else None,
        "imputed_feature_count": len(imputed),
        "imputed_features": imputed[:10],
    }


@app.get("/compare")
def compare(
    target: str = Query(..., description="Kiyaslanacak hedef"),
    hours: int = Query(12, ge=1, le=720),
) -> Dict[str, Any]:
    """Gercek vs tahmin serisi -- dashboard'un ana grafigi.

    Once `twin-predictions` deposundan okur (canli skorlama ciktisi); orada
    kayit yoksa mevcut modelle aninda hesaplar (demo'nun bos ekran gostermemesi
    icin).
    """
    end = state.repo.max_ts(Dataset.TELEMETRY, filters={"line_id": state.profile.line_id})
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)

    stored = state.repo.read(
        Dataset.PREDICTIONS, start=start, end=end,
        filters={"line_id": state.profile.line_id, "target": target},
    )
    source = "predictions_store"
    if stored.empty:
        bundles = state.require_models()
        if target not in bundles:
            raise HTTPException(404, f"Bu hedef icin model yok: {target}")
        frame = state.context.frame(hours=hours)
        rows = score_frame(frame, {target: bundles[target]}, state.profile)
        stored = pd.DataFrame(rows)
        source = "computed_on_the_fly"
    if stored.empty:
        return {"target": target, "source": source, "points": [], "metrics": {}}

    stored["ts"] = pd.to_datetime(stored["ts"], utc=True)
    stored = stored.sort_values("ts")
    both = stored.dropna(subset=["y_true"])
    metrics: Dict[str, float] = {}
    if len(both) > 5:
        error = both["y_true"] - both["y_pred"]
        metrics = {
            "mae": float(error.abs().mean()),
            "bias": float(error.mean()),
            "n": int(len(both)),
        }
    return {
        "target": target,
        "source": source,
        "model_version": str(stored["model_version"].iloc[-1]) if "model_version" in stored else None,
        "metrics": metrics,
        "points": [
            {
                "ts": pd.Timestamp(r.ts).isoformat(),
                "y_true": None if pd.isna(r.y_true) else float(r.y_true),
                "y_pred": float(r.y_pred),
            }
            for r in stored.itertuples()
        ],
    }


@app.get("/drift")
def drift(target: str, weeks: int = Query(6, ge=1, le=52)) -> Dict[str, Any]:
    """Haftalik MAE takibi.

    Egitim MAE'sinin 1.5 katini asmak, yeniden egitim tetigidir. Bakim, proses
    degisikligi veya hammadde degisimi sonrasi bu esik gecilir.
    """
    bundles = state.require_models()
    if target not in bundles:
        raise HTTPException(404, f"Bu hedef icin model yok: {target}")

    end = datetime.now(timezone.utc)
    frame = state.repo.read(
        Dataset.PREDICTIONS, start=end - timedelta(weeks=weeks),
        filters={"line_id": state.profile.line_id, "target": target},
    )
    baseline = float(bundles[target].metrics.get("holdout", {}).get("mae", float("nan")))
    if frame.empty:
        return {"target": target, "baseline_mae": baseline, "weekly": [], "retrain_recommended": False}

    frame = frame.dropna(subset=["y_true"])
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    weekly = (
        frame.assign(error=(frame["y_true"] - frame["y_pred"]).abs())
        .set_index("ts")
        .resample("W")["error"]
        .agg(["mean", "count"])
        .reset_index()
    )
    rows = [
        {"week": pd.Timestamp(r.ts).date().isoformat(), "mae": float(r.mean), "n": int(r.count)}
        for r in weekly.itertuples()
    ]
    recent = rows[-1]["mae"] if rows else float("nan")
    return {
        "target": target,
        "baseline_mae": baseline,
        "threshold_mae": baseline * 1.5,
        "recent_mae": recent,
        "weekly": rows,
        "retrain_recommended": bool(np.isfinite(baseline) and np.isfinite(recent) and recent > baseline * 1.5),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> Dict[str, Any]:
    bundles = state.require_models()
    targets = request.targets or list(bundles)
    frame = pd.DataFrame([request.features])
    out: Dict[str, Any] = {}
    for target in targets:
        if target not in bundles:
            raise HTTPException(404, f"Bu hedef icin model yok: {target}")
        bundle = bundles[target]
        out[target] = {
            "value": float(bundle.predict(frame)[0]),
            "in_domain": bool(bundle.in_domain(frame).iloc[0]),
        }
    return {"predictions": out}


@app.post("/whatif")
def whatif(request: WhatIfRequest) -> Dict[str, Any]:
    bundles = state.require_models()
    row = state.require_row()
    _, grade = state.context.current_grade()

    unknown = [k for k in request.setpoints if k not in state.profile.variables]
    if unknown:
        raise HTTPException(400, f"Tanimsiz degisken: {unknown}")

    result = what_if(row, bundles, state.profile, grade, request.setpoints, request.hour)
    frame = pd.DataFrame(
        [apply_setpoints(row, request.setpoints, state.profile).values], columns=row.index
    )
    result["in_domain"] = bool(all(b.in_domain(frame).iloc[0] for b in bundles.values()))
    result["ts"] = pd.Timestamp(row.name).isoformat()
    return result


@app.post("/optimize")
def optimize(request: OptimizeRequest) -> Dict[str, Any]:
    bundles = state.require_models()
    row = state.require_row()
    code, grade = state.context.current_grade()

    result = optimize_row(
        row, bundles, state.profile, grade,
        hour=request.hour,
        variables=request.variables,
        respect_rate_limits=request.respect_rate_limits,
        settings=state.settings,
    )
    payload = result.to_dict()
    payload.update({
        "ts": pd.Timestamp(row.name).isoformat(),
        "product_code": code,
        "units": {name: state.profile.var(name).unit for name in result.setpoints},
    })

    # Tavsiyeyi kaydet: kabul/ret takibi ve sonradan kazanc dogrulamasi icin
    state.repo.write(Dataset.RECOMMENDATIONS, [{
        "ts": pd.Timestamp(row.name).isoformat(),
        "line_id": state.profile.line_id,
        "product_code": code,
        "current_setpoints": result.current_setpoints,
        "recommended_setpoints": result.setpoints,
        "predicted_current": result.predicted_current,
        "predicted_optimized": result.predicted,
        "saving_try_per_ton": result.saving_try_per_ton,
        "confidence": result.confidence,
        "accepted": None,
        "notes": result.notes,
    }])
    return payload


@app.get("/setpoints")
def setpoints() -> Dict[str, Any]:
    """Optimize edilebilir degiskenler ve sinirlari -- dashboard slider'lari icin."""
    row = state.context.latest_row()
    return {
        "variables": [
            {
                "name": v.name,
                "desc": v.desc,
                "unit": v.unit,
                "op_low": v.op_low,
                "op_high": v.op_high,
                "max_delta": v.max_delta,
                "current": None if row is None or v.name not in row.index else float(row[v.name]),
            }
            for v in state.profile.controllables
        ]
    }
