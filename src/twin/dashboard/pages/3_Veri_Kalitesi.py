"""Veri kalitesi -- musteri verisi geldiginde ilk bakilacak ekran.

Ana ekran (Gercek vs Tahmin) bilerek sade tutuldu; bu sayfa isteyen icin.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from twin.dashboard import theme  # noqa: E402
from twin.dashboard.data import load_context, resources  # noqa: E402
from twin.features.build import DOWNTIME_FLAG  # noqa: E402
from twin.schema import Dataset  # noqa: E402

st.set_page_config(page_title="Veri Kalitesi", page_icon="🔍", layout="wide")
theme.use_dark(str(st.get_option("theme.base") or "light").lower() == "dark")

profile, repo, bundles, settings = resources()
if not bundles:
    st.error("Egitilmis model yok -- once `make train`")
    st.stop()

frame = load_context(24, pd.Timestamp.utcnow().floor("30s").value)
if frame.empty:
    st.error("Veri bulunamadi -- once `make sim`")
    st.stop()

running = frame[frame.get(DOWNTIME_FLAG, 0) == 0]
latest = running.iloc[-1] if not running.empty else frame.iloc[-1]
latest_row = latest.rename(latest["ts"])
grade_code = str(latest.get("product_code", "?"))
grade = profile.grades.get(grade_code)
hour_now = int(pd.Timestamp(latest["ts"]).hour)

st.markdown("#### Veri kalitesi")
st.caption(
    "Müşteri verisi geldiğinde **ilk bakılacak ekran budur**. Model kurmadan önce "
    "verinin taşıyabileceğini bilmek gerekir; kötü veriyle devam etmek en pahalı hatadır."
)

health = repo.health()
counts = health.get("counts", {})
cols = st.columns(4)
cols[0].metric("Depo", repo.name)
cols[1].metric("Telemetri satırı", f"{counts.get('telemetry', 0):,}")
cols[2].metric("Duruş oranı", f"%{frame.get(DOWNTIME_FLAG, pd.Series([0])).mean() * 100:.1f}")
cols[3].metric("Model satırı (pencere)", f"{len(frame):,}")

st.markdown("**Sensör bazlı doluluk ve aralık kontrolü**")
audit: List[Dict[str, Any]] = []
for var in profile.variables.values():
    if var.name not in frame.columns:
        audit.append({"Değişken": var.name, "Rol": var.role, "Doluluk %": 0.0,
                      "Durum": "VERİ YOK"})
        continue
    series = frame[var.name]
    completeness = float(series.notna().mean() * 100)
    out_of_range = 0.0
    if var.op_low is not None and var.op_high is not None:
        out_of_range = float((~series.between(var.op_low, var.op_high)).mean() * 100)
    stuck = float(series.diff().abs().lt(1e-9).mean() * 100)
    status = "OK"
    if completeness < 90:
        status = "EKSİK VERİ"
    elif stuck > 80:
        status = "DONUK SENSÖR?"
    elif out_of_range > 5:
        status = "ARALIK DIŞI"
    audit.append({
        "Değişken": var.name, "Rol": var.role, "Birim": var.unit,
        "Doluluk %": round(completeness, 1),
        "Aralık dışı %": round(out_of_range, 1),
        "Sabit kalma %": round(stuck, 1),
        "Durum": status,
    })
audit_frame = pd.DataFrame(audit)
problems = audit_frame[audit_frame["Durum"] != "OK"]
if not problems.empty:
    st.warning(f"{len(problems)} değişkende dikkat gerektiren durum var.")
st.dataframe(audit_frame, width="stretch", hide_index=True)

st.markdown("**Kalite (lab) spec uyumu**")
quality = repo.read(Dataset.QUALITY, filters={"line_id": profile.line_id})
if quality.empty:
    st.info("Laboratuvar verisi yok.")
else:
    summary = (
        quality.groupby("property")
        .agg(ornek=("value", "size"), ortalama=("value", "mean"), spec_ici_orani=("passed", "mean"))
        .reset_index()
    )
    summary["spec_ici_orani"] = (summary["spec_ici_orani"] * 100).round(1)
    st.dataframe(summary.round(3), width="stretch", hide_index=True)
    st.caption(
        "Spec içi oranı %100'e çok yakınsa **aşırı kalite (over-quality)** ihtimali vardır: "
        "spec'in ortasında değil, alt sınırına yakın üretmek para kazandırır."
    )
