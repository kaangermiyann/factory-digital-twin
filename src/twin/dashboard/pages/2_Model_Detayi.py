"""Model detayi -- karsilastirma, ozellik onemi, PDP.

Ana ekran (Gercek vs Tahmin) bilerek sade tutuldu; bu sayfa isteyen icin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from twin.dashboard import theme  # noqa: E402
from twin.dashboard.data import load_context, resources  # noqa: E402
from twin.features.build import DOWNTIME_FLAG  # noqa: E402
from twin.models.evaluate import partial_dependence_curve as pdp_curve  # noqa: E402

st.set_page_config(page_title="Model Detayi", page_icon="🧠", layout="wide")
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

st.markdown("#### Model karşılaştırması")
st.caption(
    "“RandomForest iyi” demek zayıftır. Naive taban çizgisiyle birlikte göstermek, "
    "modelin gerçekten bir şey öğrendiğini kanıtlar. "
    "`persistence*` (son değeri tekrarla) genelde iyi skor verir ama karşı-olgusal "
    "soru soramaz — **optimizasyonda kullanılamaz**."
)

target = st.selectbox("Hedef", list(bundles),
                      format_func=lambda t: f"{profile.targets[t].desc}  ({t})")
bundle = bundles[target]

m1, m2, m3, m4 = st.columns(4)
holdout = bundle.metrics.get("holdout", {})
m1.metric("Seçilen model", bundle.algorithm)
if bundle.kind == "regression":
    m2.metric("MAE (holdout)", f"{holdout.get('mae', float('nan')):,.2f}")
    m3.metric("MAPE", f"%{holdout.get('mape', float('nan')):.2f}")
    m4.metric("R²", f"{holdout.get('r2', float('nan')):.3f}")
else:
    m2.metric("PR-AUC", f"{holdout.get('pr_auc', float('nan')):.3f}")
    m3.metric("Recall @ %50 kesinlik", f"%{holdout.get('recall_at_p50', 0) * 100:.0f}")
    m4.metric("Pozitif oran", f"%{holdout.get('positive_rate', 0) * 100:.1f}")

for note in bundle.metrics.get("notes", []):
    (st.warning if "SIZINTI" in note else st.info)(note)

comparison = pd.DataFrame(bundle.metrics.get("comparison", []))
left, right = st.columns([1, 1])
with left:
    st.markdown("**Holdout metrikleri**")
    st.dataframe(comparison.round(4), width="stretch", hide_index=True)
    metric_col = "mae" if bundle.kind == "regression" else "pr_auc"
    if metric_col in comparison.columns:
        plot = comparison.dropna(subset=[metric_col]).sort_values(metric_col,
                                                                  ascending=bundle.kind != "regression")
        st.plotly_chart(
            theme.horizontal_bars(plot["model"].tolist(), plot[metric_col].tolist(), height=220),
            key="model_bar",
        )
        st.caption(f"↑ {metric_col.upper()} — tek seri, büyüklük tek hue ile kodlandı.")

with right:
    st.markdown("**En etkili değişkenler** (permutation importance)")
    st.caption(
        "Proses mühendisinin onaylayacağı çıktı budur. Listenin başındaki değişken "
        "beklentiyle uyuşmuyorsa ya veri ya da etiketleme hatalıdır."
    )
    importance = pd.DataFrame(bundle.metrics.get("top_features", []))
    if not importance.empty:
        top = importance.head(14)
        st.plotly_chart(
            theme.horizontal_bars(top["feature"].tolist(), top["importance"].tolist(), height=420),
            key="feat_bar",
        )

st.divider()
st.markdown("**Kısmi bağımlılık (PDP)** — aksiyona dönüşebilen tek açıklanabilirlik çıktısı")
st.caption(
    "Diğer her şey sabitken bu değişkeni gezdirdiğimizde model ne diyor? "
    "Eğrideki kırılma noktası doğrudan bir işletme kuralına dönüşür: "
    "*“bu değerin ötesinde enerji tüketimi tırmanıyor.”*"
)
pdp_options = [v.name for v in profile.controllables if v.name in bundle.features]
if pdp_options:
    pdp_var = st.selectbox("Değişken", pdp_options,
                           format_func=lambda n: f"{profile.var(n).desc} ({profile.var(n).unit})",
                           key="pdp_var")
    pdp_source = bundle.align(frame).dropna()
    if len(pdp_source) >= 30:
        curve = pdp_curve(bundle.model, pdp_source, pdp_var, bundle.kind)
        fig = go.Figure(theme.line(profile.targets[target].desc, curve[pdp_var],
                                   curve["prediction"], slot=0))
        current_value = float(latest_row[pdp_var])
        fig.add_vline(x=current_value, line=dict(color=theme.INK_MUTED, width=1, dash="dot"),
                      annotation_text="şu an", annotation_position="top",
                      annotation_font=dict(size=10, color=theme.INK_MUTED))
        unit = profile.var(target).unit if target in profile.variables else ""
        fig.update_layout(**theme.layout(
            height=280, hovermode="x",
            xaxis=dict(title=f"{profile.var(pdp_var).desc} [{profile.var(pdp_var).unit}]",
                       showgrid=False, zeroline=False, linecolor=theme.GRID),
            yaxis=dict(title=unit, showgrid=True, gridcolor=theme.GRID,
                       zeroline=False, linecolor=theme.GRID)))
        st.plotly_chart(fig, key="pdp")
        st.caption(
            "Eğri yalnızca modelin **gördüğü** aralıkta anlamlıdır; uçlarda "
            "RandomForest ekstrapole edemediği için düzleşir."
        )
    else:
        st.info("PDP için pencerede yeterli veri yok — zaman penceresini genişletin.")

st.divider()
st.markdown("**Eğitim künyesi**")
st.json({
    "surum": bundle.version,
    "algoritma": bundle.algorithm,
    "egitim_araligi": [bundle.train_start, bundle.train_end],
    "satir": bundle.n_rows,
    "ozellik": len(bundle.features),
}, expanded=False)


# --------------------------------------------------------------------------- #
