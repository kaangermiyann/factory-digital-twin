"""Dashboard veri erisimi -- onbellekli, sayfalar arasi paylasilan.

Tek kural: "gercek vs tahmin" grafigi VARSAYILAN OLARAK holdout doneminden
cizilir. Modelin egitildigi donemde iyi uyum gostermesi zaten beklenir ve
hicbir sey kanitlamaz; gosterilmesi gereken, modelin HIC GORMEDIGI veridir.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from twin.config import Profile, get_profile, get_settings
from twin.models.registry import ModelBundle, load_all
from twin.schema import Dataset
from twin.serving.scorer import LiveContext, score_frame
from twin.storage import Repository, get_repository


@st.cache_resource
def resources() -> Tuple[Profile, Repository, Dict[str, ModelBundle], Any]:
    profile = get_profile()
    return profile, get_repository(), load_all(profile.name), get_settings()


def holdout_start(bundles: Dict[str, ModelBundle]) -> Optional[pd.Timestamp]:
    """Modellerin hicbirinin gormedigi donemin baslangici.

    Birden fazla model varsa EN GEC holdout basi alinir; boylece gosterilen
    aralik her model icin gercekten gorulmemis olur.
    """
    stamps = [
        pd.to_datetime(b.holdout_start, utc=True)
        for b in bundles.values()
        if getattr(b, "holdout_start", None)
    ]
    return max(stamps) if stamps else None


def deduplicate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Ayni (zaman, hedef) icin tek satir birak -- en YENI model surumu kazanir.

    Tahmin deposu EKLEMELIDIR (append-only): `make score` iki kez kosarsa ayni
    zaman damgasi icin birden fazla satir olusur; grafik ust uste cizer, tablo
    tekrarlar, "kac tahmin var" sayisi sisirilir. Metrikler bozulmaz (kopyalar
    ozdestir) ama gorsel ve sayimlar yaniltici olur.

    En yeni surumu secmek ayni zamanda model guncellemesini dogal olarak
    devreye sokar: yeniden egitimden sonra eski tahminler gorunmez olur.
    """
    if frame.empty or not {"ts", "target"}.issubset(frame.columns):
        return frame
    order = ["ts", "target"] + (["model_version"] if "model_version" in frame.columns else [])
    return (
        frame.sort_values(order)
        .drop_duplicates(subset=["ts", "target"], keep="last")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=60, show_spinner="Tahminler okunuyor...")
def load_comparison(_stamp: float) -> pd.DataFrame:
    """`twin-predictions` deposundan gercek + tahmin serisi.

    Depoda kayit yoksa mevcut modellerle aninda hesaplanir -- demo bos ekran
    gostermemeli.
    """
    profile, repo, bundles, _ = resources()
    frame = repo.read(Dataset.PREDICTIONS, filters={"line_id": profile.line_id})

    if frame.empty and bundles:
        context = LiveContext(profile, repo, cache_seconds=0)
        rows = score_frame(context.frame(hours=72, force=True), bundles, profile)
        frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")

    frame = deduplicate_predictions(frame)
    frame["error"] = frame["y_true"] - frame["y_pred"]
    return frame.sort_values("ts").reset_index(drop=True)


@st.cache_data(ttl=30, show_spinner=False)
def load_context(hours: int, _stamp: float) -> pd.DataFrame:
    """Uretim baglami (urun, ekip, durus) -- grafiklerin uzerine serilir."""
    profile, repo, _, _ = resources()
    return LiveContext(profile, repo, cache_seconds=0).frame(hours=hours, force=True)


def metrics_for(part: pd.DataFrame) -> Dict[str, float]:
    """Sadece gercek degeri BILINEN satirlar uzerinden hata metrikleri."""
    both = part.dropna(subset=["y_true", "y_pred"])
    if both.empty:
        return {"n": 0}
    error = both["y_true"] - both["y_pred"]
    denominator = both["y_true"].abs().replace(0, np.nan)
    return {
        "n": int(len(both)),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "mape": float((error.abs() / denominator).mean() * 100.0),
        "bias": float(error.mean()),
        "within_1pct": float((error.abs() / denominator <= 0.01).mean() * 100.0),
    }


def spec_violations(part: pd.DataFrame, spec: Optional[Tuple[float, float]]) -> Dict[str, float]:
    """Spec disina cikma oranlari -- olculen ve tahmin edilen ayri ayri.

    Ikisinin YAKIN olmasi, modelin sadece ortalamayi degil UC DURUMLARI da
    yakaladigi anlamina gelir; asil deger orada.
    """
    if not spec:
        return {}
    both = part.dropna(subset=["y_true", "y_pred"])
    if both.empty:
        return {}
    low, high = spec
    actual = ~both["y_true"].between(low, high)
    predicted = ~both["y_pred"].between(low, high)
    caught = int((actual & predicted).sum())
    return {
        "actual_pct": float(actual.mean() * 100.0),
        "predicted_pct": float(predicted.mean() * 100.0),
        "caught_pct": float(caught / actual.sum() * 100.0) if actual.sum() else float("nan"),
        "actual_n": int(actual.sum()),
    }


def target_unit(profile: Profile, target: str) -> str:
    return profile.var(target).unit if target in profile.variables else ""


def target_label(profile: Profile, target: str) -> str:
    spec = profile.targets.get(target)
    return spec.desc if spec else target


def regression_targets(frame: pd.DataFrame, profile: Profile) -> List[str]:
    available = set(frame["target"].unique()) if not frame.empty else set()
    return [
        name for name, spec in profile.targets.items()
        if spec.kind == "regression" and name in available
    ]
