"""Optimizasyon -- what-if ve kisitli arama.

Ana ekran (Gercek vs Tahmin) bilerek sade tutuldu; bu sayfa isteyen icin.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from twin.dashboard import theme  # noqa: E402
from twin.dashboard.data import load_context, resources  # noqa: E402
from twin.features.build import BREAK_LABEL, DOWNTIME_FLAG  # noqa: E402
from twin.optimize.objective import Economics  # noqa: E402
from twin.optimize.search import optimize_row, what_if  # noqa: E402

st.set_page_config(page_title="Optimizasyon", page_icon="🎯", layout="wide")
theme.use_dark(str(st.get_option("theme.base") or "light").lower() == "dark")

profile, repo, bundles, settings = resources()
if not bundles:
    st.error("Egitilmis model yok -- once `make train`")
    st.stop()

frame = load_context(12, pd.Timestamp.utcnow().floor("30s").value)
if frame.empty:
    st.error("Veri bulunamadi -- once `make sim`")
    st.stop()

running = frame[frame.get(DOWNTIME_FLAG, 0) == 0]
latest = running.iloc[-1] if not running.empty else frame.iloc[-1]
latest_row = latest.rename(latest["ts"])
grade_code = str(latest.get("product_code", "?"))
grade = profile.grades.get(grade_code)
hour_now = int(pd.Timestamp(latest["ts"]).hour)
economics = Economics.from_settings(settings)

st.markdown("#### What-if — senaryo analizi")
st.caption(
    "Setpoint'i değiştir, ikiz anında sonucu tahmin etsin. "
    "Bu ekran aynı zamanda bir **operatör eğitim aracıdır**."
)

controllables = [v for v in profile.controllables if v.name in latest_row.index]
picks = st.multiselect(
    "Değiştirilecek setpoint'ler", [v.name for v in controllables],
    default=[v.name for v in controllables[:3]],
    format_func=lambda n: f"{profile.var(n).desc} ({profile.var(n).unit})",
)

scenario: Dict[str, float] = {}
slider_cols = st.columns(min(3, max(len(picks), 1)))
for i, name in enumerate(picks):
    var = profile.var(name)
    current = float(latest_row[name])
    step = (var.op_high - var.op_low) / 200.0
    scenario[name] = slider_cols[i % len(slider_cols)].slider(
        f"{var.desc} [{var.unit}]",
        float(var.op_low), float(var.op_high), current, step=float(step),
        key=f"wi_{name}",
    )

if scenario:
    outcome = what_if(latest_row, bundles, profile, grade, scenario, hour_now)
    cols = st.columns(len(outcome["scenario"]) + 1)
    for i, (name, value) in enumerate(outcome["scenario"].items()):
        unit = profile.var(name).unit if name in profile.variables else ""
        delta = outcome["delta"][name]
        spec = grade.spec(name) if grade else None
        cols[i].metric(
            profile.targets[name].desc.split(" (")[0],
            (f"%{value * 100:.1f}" if name == BREAK_LABEL else f"{value:,.2f} {unit}"),
            (f"%{delta * 100:+.1f}" if name == BREAK_LABEL else f"{delta:+,.2f}"),
            delta_color="off",
        )
        if spec:
            status = theme.status_of(value, *spec)
            cols[i].markdown(
                f"<span style='color:{theme.STATUS[status]};font-size:0.78rem'>● "
                f"{theme.STATUS_LABEL[status]}</span>", unsafe_allow_html=True)
    cols[-1].metric("Enerji maliyeti", f"{outcome['cost_delta_try_per_ton']:+,.1f} ₺/t",
                    delta_color="off")
    for violation in outcome["violations"]:
        st.error(f"Kısıt ihlali — {violation}")

st.divider()
st.markdown("#### Kısıtlı optimizasyon")
st.caption(
    "Bağlam (ürün, hammadde, hava) sabit tutulur; sadece operatörün gerçekten "
    "çevirebildiği kollar, **emniyet aralığı ∩ değişim hızı limiti ∩ eğitim verisi zarfı** "
    "içinde aranır."
)

opt_cols = st.columns([1, 1, 2])
respect_limits = opt_cols[0].toggle("Değişim hızı limiti", value=True,
                                    help="|Δsetpoint| ≤ max_delta (ani sıçramayı engeller)")
annual_hours = opt_cols[1].number_input("Yıllık çalışma saati", 4000, 8760, 7800, step=100)

if st.button("⚙️ Optimize et", type="primary"):
    with st.spinner("Setpoint uzayı taranıyor..."):
        result = optimize_row(latest_row, bundles, profile, grade,
                              hour=hour_now, respect_rate_limits=respect_limits,
                              settings=settings)
    st.session_state["optimization"] = result

result = st.session_state.get("optimization")
if result is not None:
    rows = []
    for name, value in result.setpoints.items():
        var = profile.var(name)
        current = result.current_setpoints[name]
        rows.append({
            "Değişken": var.desc,
            "Birim": var.unit,
            "Mevcut": round(current, 2),
            "Tavsiye": round(value, 2),
            "Δ": round(value - current, 2),
            "Emniyet aralığı": f"{var.op_low:g} … {var.op_high:g}",
        })
    table = pd.DataFrame(rows)
    changed = table[table["Δ"].abs() > 1e-6]

    st.dataframe(changed if not changed.empty else table,
                 width="stretch", hide_index=True)

    # --- setpointlerin aralik icindeki konumu (aralik grafigi) ---------- #
    fig = go.Figure()
    labels = [profile.var(n).desc for n in result.setpoints]
    for i, name in enumerate(result.setpoints):
        var = profile.var(name)
        span = max(var.op_high - var.op_low, 1e-9)
        norm = lambda v: (v - var.op_low) / span * 100.0  # noqa: E731
        fig.add_trace(go.Scatter(
            x=[0, 100], y=[labels[i]] * 2, mode="lines",
            line=dict(color=theme.GRID, width=6), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[norm(result.current_setpoints[name])], y=[labels[i]], mode="markers",
            marker=dict(color=theme.SERIES[0], size=11,
                        line=dict(color="rgba(128,128,120,0.5)", width=2)),
            name="mevcut", showlegend=i == 0,
            hovertemplate=f"{labels[i]}<br>mevcut %{{customdata:.2f}} {var.unit}<extra></extra>",
            customdata=[result.current_setpoints[name]]))
        fig.add_trace(go.Scatter(
            x=[norm(result.setpoints[name])], y=[labels[i]], mode="markers",
            marker=dict(color=theme.SERIES[1], size=11, symbol="diamond",
                        line=dict(color="rgba(128,128,120,0.5)", width=2)),
            name="tavsiye", showlegend=i == 0,
            hovertemplate=f"{labels[i]}<br>tavsiye %{{customdata:.2f}} {var.unit}<extra></extra>",
            customdata=[result.setpoints[name]]))
    fig.update_layout(**theme.layout(
        height=60 + 34 * len(labels), hovermode="closest",
        xaxis=dict(title="emniyet aralığı içindeki konum (%)", range=[-4, 104],
                   showgrid=True, gridcolor=theme.GRID, zeroline=False),
        yaxis=dict(autorange="reversed", showgrid=False, linecolor=theme.GRID)))
    st.plotly_chart(fig, key="opt_range")

    st.markdown("**Tahmini sonuç**")
    outcome_cols = st.columns(len(result.predicted) + 1)
    for i, (name, value) in enumerate(result.predicted.items()):
        before = result.predicted_current[name]
        unit = profile.var(name).unit if name in profile.variables else ""
        outcome_cols[i].metric(
            profile.targets[name].desc.split(" (")[0],
            (f"%{value * 100:.1f}" if name == BREAK_LABEL else f"{value:,.1f} {unit}"),
            (f"%{(value - before) * 100:+.1f}" if name == BREAK_LABEL else f"{value - before:+,.1f}"),
            delta_color="off")

    saving = result.saving_try_per_ton
    annual = saving * float(latest.get("production_rate_tph", 0) or 0) * annual_hours
    badge = {"high": "🟢 Yüksek", "medium": "🟡 Orta", "low": "🔴 Ekstrapolasyon"}[result.confidence]
    outcome_cols[-1].metric("Kazanç", f"{saving:,.1f} ₺/t")
    st.markdown(
        f"### {annual / 1e6:,.1f} milyon ₺ / yıl "
        f"<span style='font-size:0.9rem;color:{theme.INK_MUTED}'>"
        f"({latest.get('production_rate_tph', 0):.1f} t/h × {annual_hours:,} saat varsayımıyla)</span>",
        unsafe_allow_html=True)
    st.markdown(f"**Güven:** {badge}")

    for note in result.notes:
        st.caption(f"· {note}")
    for violation in result.violations:
        st.error(f"Kısıt ihlali — {violation}")
    st.caption(
        f"{result.n_evaluations:,} aday setpoint kombinasyonu değerlendirildi. "
        "Tavsiye `twin-recommendations` deposuna yazıldı (kabul/ret takibi için)."
    )
