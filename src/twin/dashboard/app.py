"""Gercek vs Tahmin -- dijital ikizin ana ekrani.

    streamlit run src/twin/dashboard/app.py

Bu sayfa TEK BIR SORUYA cevap verir:

    "Model, sahadan olculen degeri tutturabiliyor mu?"

Bilerek disarida birakilanlar: model karsilastirma tablolari, ozellik onemleri,
egitim metrikleri, optimizasyon. Bunlar `pages/` altinda duruyor -- isteyen
bakar, ana ekrani mesgul etmezler.

EN KRITIK TASARIM KARARI -- grafik varsayilan olarak HOLDOUT doneminden cizilir:
modelin egitildigi donemde iyi uyum gostermesi zaten beklenir, oradaki guzel
grafik hicbir sey kanitlamaz ve gostermek yaniltici olur. Gosterilen, modelin
hic gormedigi veridir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

if __package__ in (None, ""):  # `streamlit run <dosya>` ile calistirildiginda
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from twin.dashboard import theme  # noqa: E402
from twin.dashboard.data import (  # noqa: E402
    holdout_start,
    load_failures,
    load_comparison,
    load_context,
    metrics_for,
    regression_targets,
    resources,
    spec_violations,
    target_label,
    target_unit,
)
from twin.features.build import BREAK_LABEL, DOWNTIME_FLAG  # noqa: E402

st.set_page_config(page_title="Gerçek vs Tahmin", page_icon="📈", layout="wide")
theme.use_dark(str(st.get_option("theme.base") or "light").lower() == "dark")

profile, repo, bundles, settings = resources()

failures = load_failures()
if failures:
    st.error(
        "**{} model yüklenemedi** — bu hedefler aşağıda görünmeyecek:\n\n{}\n\n"
        "Model dosyaları eğitildikleri ortama bağlıdır (pickle). Çözüm: "
        "`./start.sh --fresh` veya `make train`.".format(
            len(failures),
            "\n".join(f"- `{name}`: {reason}" for name, reason in failures.items()),
        ),
        icon="🚨",
    )

if not bundles:
    st.error("Eğitilmiş model yok.\n\n```\nmake sim\nmake train\nmake score\n```")
    st.stop()

comparison = load_comparison(pd.Timestamp.utcnow().floor("60s").value)
if comparison.empty:
    st.error("Tahmin kaydı yok.\n\n```\nmake score\n```")
    st.stop()

cutoff = holdout_start(bundles)
unseen = comparison[comparison["ts"] >= cutoff] if cutoff is not None else comparison


# --------------------------------------------------------------------------- #
# Kenar cubugu
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(f"### 📈 {profile.plant} / {profile.line_id}")
    st.caption(profile.description)

    only_unseen = st.toggle(
        "Sadece görülmemiş veri", value=True,
        help="Modelin eğitildiği dönemi grafikten çıkarır. Kapatırsanız eğitim "
             "dönemi de görünür — orada uyum iyi olur ama bir şey kanıtlamaz.",
    )
    source = unseen if (only_unseen and not unseen.empty) else comparison

    span_days = max((source["ts"].max() - source["ts"].min()).days, 1)
    window = st.select_slider(
        "Gösterilen aralık", options=sorted({1, 3, 7, 14, span_days}),
        value=min(7, span_days), format_func=lambda d: f"son {d} gün",
    )
    if st.button("↻ Yenile", width="stretch"):
        st.cache_data.clear()

    st.divider()
    st.caption(f"**Depo:** `{repo.name}` · **Profil:** `{profile.name}`")
    if cutoff is not None:
        st.caption(f"**Holdout başı:** {cutoff:%d.%m.%Y %H:%M}")
    for target, bundle in sorted(bundles.items()):
        st.caption(f"`{target}` → {bundle.algorithm}")

end = source["ts"].max()
window_frame = source[source["ts"] >= end - pd.Timedelta(days=window)]

# --------------------------------------------------------------------------- #
st.markdown("## Gerçek vs Tahmin")
if only_unseen and cutoff is not None:
    st.success(
        f"**Model bu veriyi hiç görmedi** — {cutoff:%d.%m.%Y} sonrası holdout dönemi. "
        "Eğitim döneminde iyi uyum göstermek beklenen bir şeydir ve hiçbir şey kanıtlamaz.",
        icon="✅",
    )
else:
    st.warning(
        "Eğitim dönemi de gösteriliyor. Buradaki uyum modelin başarısını **ölçmez** — "
        "model bu veriyi ezberlemiş olabilir.",
        icon="⚠️",
    )

targets = regression_targets(window_frame, profile)
if not targets:
    st.info("Karşılaştırılacak regresyon hedefi yok.")
    st.stop()


# --------------------------------------------------------------------------- #
# 1) Ozet -- hedef basina hata
# --------------------------------------------------------------------------- #
st.markdown("#### Görülmemiş veri üzerinde hata")
for column, target in zip(st.columns(len(targets)), targets):
    stats = metrics_for(window_frame[window_frame["target"] == target])
    unit = target_unit(profile, target)
    label = target_label(profile, target).split(" (")[0]

    if not stats.get("n"):
        column.metric(label, "—")
        continue
    column.metric(label, f"{stats['mae']:,.2f} {unit}",
                  f"MAPE %{stats['mape']:.2f}", delta_color="off")
    column.caption(
        f"ortalama sapma **{abs(stats['bias']):,.2f} {unit}** "
        f"({'yüksek' if stats['bias'] < 0 else 'düşük'} tahmin) · {stats['n']:,} nokta"
    )

st.caption(
    "MAE = ortalama mutlak hata. **Sapma (bias)** sıfıra yakınsa model sistematik olarak "
    "yüksek/düşük tahmin etmiyor demektir — MAE kadar önemlidir ve genelde atlanır."
)
st.divider()


# --------------------------------------------------------------------------- #
# 2) Zaman serisi karsilastirmasi
# --------------------------------------------------------------------------- #
def shade_downtime(fig: go.Figure, frame: pd.DataFrame) -> None:
    """Duruş dönemlerini gri şeritle işaretle -- oradaki kopukluk model hatası değil."""
    if frame.empty or DOWNTIME_FLAG not in frame.columns:
        return
    flag = frame[DOWNTIME_FLAG].fillna(0).astype(int).to_numpy()
    stamps = frame["ts"].to_numpy()
    start = None
    for i, value in enumerate(flag):
        if value and start is None:
            start = stamps[i]
        elif not value and start is not None:
            fig.add_vrect(x0=start, x1=stamps[i], fillcolor=theme.BAND,
                          line_width=0, layer="below")
            start = None
    if start is not None:
        fig.add_vrect(x0=start, x1=stamps[-1], fillcolor=theme.BAND,
                      line_width=0, layer="below")


context = load_context(int(window * 24) + 4, pd.Timestamp.utcnow().floor("60s").value)
grade = profile.grades.get(str(context["product_code"].iloc[-1])) if not context.empty else None

for target in targets:
    part = window_frame[window_frame["target"] == target].dropna(subset=["y_true"])
    if part.empty:
        continue
    unit = target_unit(profile, target)
    spec = grade.spec(target) if grade else None

    st.markdown(f"##### {target_label(profile, target)}")

    fig = go.Figure()
    if spec:
        theme.spec_band(fig, spec[0], spec[1], f"spec {spec[0]:g}–{spec[1]:g}")
    shade_downtime(fig, context)
    fig.add_trace(theme.line("ölçülen", part["ts"], part["y_true"], slot=0))
    fig.add_trace(theme.line("tahmin", part["ts"], part["y_pred"], slot=1, dash="dash"))
    fig.update_layout(**theme.layout(
        height=300,
        yaxis=dict(title=unit, showgrid=True, gridcolor=theme.GRID,
                   zeroline=False, linecolor=theme.GRID)))
    st.plotly_chart(fig, key=f"ts_{target}")

    left, right = st.columns([3, 2])

    # --- hata zaman icinde: drift buradan gorulur -------------------------- #
    with left:
        rolling = (part.set_index("ts")["error"].abs()
                   .rolling("6h", min_periods=6).mean().reset_index())
        training_mae = float(bundles[target].metrics.get("holdout", {}).get("mae", np.nan))
        error_fig = go.Figure(theme.line("hata", rolling["ts"], rolling["error"], slot=0))
        if np.isfinite(training_mae):
            error_fig.add_hline(
                y=training_mae, line=dict(color=theme.INK_MUTED, width=1, dash="dot"),
                annotation_text=f"eğitimdeki MAE {training_mae:,.2f}",
                annotation_position="top left",
                annotation_font=dict(size=10, color=theme.INK_MUTED))
            error_fig.add_hline(
                y=training_mae * 1.5,
                line=dict(color=theme.STATUS["warning"], width=1, dash="dot"),
                annotation_text="yeniden eğitim eşiği (1.5×)",
                annotation_position="bottom left",
                annotation_font=dict(size=10, color=theme.STATUS["warning"]))
        error_fig.update_layout(**theme.layout(
            height=240, showlegend=False,
            yaxis=dict(title=f"|hata| [{unit}]", showgrid=True, gridcolor=theme.GRID,
                       zeroline=False, linecolor=theme.GRID)))
        st.plotly_chart(error_fig, key=f"err_{target}")
        st.caption(
            "6 saatlik ortalama hata. Eğitimdeki seviyenin **1,5 katını** aşarsa model "
            "kaymış demektir (bakım, proses değişikliği, hammadde) → yeniden eğitim."
        )

    # --- sacilim: y=x cizgisine yakinlik ----------------------------------- #
    with right:
        sample = part.sample(min(len(part), 2500), random_state=0)
        low = float(min(sample["y_true"].min(), sample["y_pred"].min()))
        high = float(max(sample["y_true"].max(), sample["y_pred"].max()))
        scatter = go.Figure()
        scatter.add_trace(go.Scatter(
            x=[low, high], y=[low, high], mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color=theme.INK_MUTED, width=1, dash="dot")))
        scatter.add_trace(go.Scatter(
            x=sample["y_true"], y=sample["y_pred"], mode="markers", showlegend=False,
            marker=dict(color=theme.SERIES[0], size=4, opacity=0.45, line=dict(width=0)),
            hovertemplate="ölçülen %{x:.2f}<br>tahmin %{y:.2f}<extra></extra>"))
        scatter.update_layout(**theme.layout(
            height=240, hovermode="closest",
            xaxis=dict(title=f"ölçülen [{unit}]", showgrid=True, gridcolor=theme.GRID,
                       zeroline=False, linecolor=theme.GRID),
            yaxis=dict(title=f"tahmin [{unit}]", showgrid=True, gridcolor=theme.GRID,
                       zeroline=False, linecolor=theme.GRID, scaleanchor="x")))
        st.plotly_chart(scatter, key=f"sc_{target}")
        st.caption("Noktalar kesikli çizgiye ne kadar yakınsa o kadar iyi.")

    # --- uc durumlari yakalayabiliyor mu? ---------------------------------- #
    violations = spec_violations(part, spec)
    if violations.get("actual_n"):
        caught = violations["caught_pct"]
        st.markdown(
            f"**Spec dışı dönemler** — ölçülen %{violations['actual_pct']:.1f} · "
            f"model %{violations['predicted_pct']:.1f}"
            + (f" · **yakalanan %{caught:.0f}**" if np.isfinite(caught) else "")
        )
        st.caption(
            "Ortalamayı tutturmak kolaydır; **uç durumları** yakalamak zordur. "
            "Bir kalite modelinin asıl sınavı budur."
        )
    st.divider()


# --------------------------------------------------------------------------- #
# 3) Durus riski -- siniflandirma, farkli okunur
# --------------------------------------------------------------------------- #
risk = window_frame[window_frame["target"] == BREAK_LABEL].dropna(subset=["y_true"])
if not risk.empty:
    st.markdown(f"##### {target_label(profile, BREAK_LABEL)}")
    st.caption(
        "Bu bir **olasılık** tahminidir, ölçülen bir büyüklük değil — düz çizgiyle "
        "kıyaslanamaz. Doğru okuma: *gerçek duruşlardan önce risk yükseliyor mu?*"
    )
    fig = go.Figure()
    fig.add_trace(theme.line("tahmin edilen risk", risk["ts"], risk["y_pred"] * 100, slot=1))
    events = risk[risk["y_true"] > 0.5]
    if not events.empty:
        fig.add_trace(go.Scatter(
            name="gerçekleşen duruş", x=events["ts"], y=events["y_pred"] * 100, mode="markers",
            marker=dict(color=theme.STATUS["critical"], size=9, symbol="x", line=dict(width=0)),
            hovertemplate="duruş öncesi · risk %{y:.1f}%<extra></extra>"))
    ceiling = float(profile.constraints.get("break_risk_ceiling", 0.06)) * 100
    fig.add_hline(y=ceiling, line=dict(color=theme.STATUS["warning"], width=1, dash="dot"),
                  annotation_text=f"tavan %{ceiling:.0f}", annotation_position="top left",
                  annotation_font=dict(size=10, color=theme.STATUS["warning"]))
    fig.update_layout(**theme.layout(
        height=260,
        yaxis=dict(title="risk (%)", showgrid=True, gridcolor=theme.GRID,
                   zeroline=False, linecolor=theme.GRID)))
    st.plotly_chart(fig, key="risk")

    holdout = bundles[BREAK_LABEL].metrics.get("holdout", {})
    st.caption(
        f"Holdout: her 2 alarmdan 1'i gerçek olduğunda duruşların "
        f"**%{holdout.get('recall_at_p50', 0) * 100:.0f}**'ini yakalıyor "
        f"(PR-AUC {holdout.get('pr_auc', float('nan')):.3f}; rastgele {risk['y_true'].mean():.3f})."
    )
    st.divider()


# --------------------------------------------------------------------------- #
# 4) Tablo gorunumu -- erisilebilirlik + disari aktarma
# --------------------------------------------------------------------------- #
with st.expander("Tablo görünümü ve dışa aktarma"):
    table = (window_frame.pivot_table(index="ts", columns="target", values=["y_true", "y_pred"])
             .tail(200).round(3).sort_index(ascending=False))
    st.dataframe(table, width="stretch")
    st.download_button(
        "CSV indir", window_frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"gercek_vs_tahmin_{profile.line_id}.csv", mime="text/csv",
    )

st.caption(
    "Model karşılaştırması, özellik önemleri ve optimizasyon → sol menüdeki diğer sayfalar."
)
