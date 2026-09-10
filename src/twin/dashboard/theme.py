"""Dashboard gorsel dili.

Tek bir yerde tanimli: renk yuvalari, mark ozellikleri, plotly duzeni.
Kural: kategorik renkler SABIT SIRAYLA atanir, dondurulmez. Durum renkleri
(iyi/uyari/kritik) seri rengi olarak asla kullanilmaz ve tek basina anlam
tasimaz -- yanlarinda mutlaka etiket bulunur.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

# Kategorik yuvalar -- sirayla atanir, asla dondurulmez (dogrulanmis palet).
# Koyu tema ayni sekiz hue'nun koyu zemine gore adimlanmis hali; ayri palet degil.
SERIES_LIGHT: List[str] = [
    "#2a78d6",  # 1 mavi   -- olculen / gercek
    "#008300",  # 2 yesil  -- tahmin
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 sari
    "#1baf7a",  # 5 turkuaz
    "#eb6834",  # 6 turuncu
    "#4a3aa7",  # 7 mor
    "#e34948",  # 8 kirmizi
]
SERIES_DARK: List[str] = [
    "#3987e5", "#008300", "#d55181", "#c98500",
    "#199e70", "#d95926", "#9085e9", "#e66767",
]

# Aktif palet. `use_dark()` ile tema basinda bir kez ayarlanir.
SERIES: List[str] = list(SERIES_LIGHT)


def use_dark(dark: bool) -> None:
    """Streamlit temasina gore aktif paleti sec.

    Koyu tema icin renkler otomatik cevrilmez; koyu zemine gore ayrica
    dogrulanmis adimlar kullanilir.
    """
    SERIES[:] = SERIES_DARK if dark else SERIES_LIGHT

# Tek hueli sirali ramp (buyukluk kodlamasi: onem, MAE, vb.)
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

# Durum renkleri -- rezerve. Seri rengi olarak kullanilmaz.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Hem acik hem koyu temada okunan notr tonlar
INK_MUTED = "#8a8a84"
GRID = "rgba(128,128,120,0.18)"
BAND = "rgba(128,128,120,0.13)"

FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=12, color=INK_MUTED)


def layout(height: int = 320, **overrides: Any) -> Dict[str, Any]:
    """Ortak plotly duzeni: seffaf zemin (Streamlit temasina uyar), recessive
    izgara, ust kosede yatay lejant."""
    base: Dict[str, Any] = dict(
        height=height,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=FONT,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        xaxis=dict(showgrid=False, zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False, linecolor=GRID),
    )
    base.update(overrides)
    return base


def line(name: str, x, y, slot: int = 0, dash: Optional[str] = None, width: float = 2.0) -> go.Scatter:
    """2 px cizgi. Tahmin serisi kesikli cizilir -> kimlik renge tek basina
    yuklenmez (renk korlugu / siyah-beyaz cikti icin)."""
    return go.Scatter(
        name=name, x=x, y=y, mode="lines",
        line=dict(color=SERIES[slot % len(SERIES)], width=width, dash=dash),
        connectgaps=False,
    )


def spec_band(fig: go.Figure, low: float, high: float, label: str = "spec") -> go.Figure:
    """Kalite spec araligi: notr gri bant. Seri rengi kullanilmaz --
    bant bir veri serisi degil, referans alanidir."""
    fig.add_hrect(y0=low, y1=high, fillcolor=BAND, line_width=0, layer="below",
                  annotation_text=label, annotation_position="top left",
                  annotation_font=dict(size=10, color=INK_MUTED))
    return fig


def horizontal_bars(labels: List[str], values: List[float], hover: Optional[List[str]] = None,
                    height: int = 380) -> go.Figure:
    """Tek serili yatay bar (onem, hata vb.).

    Tek seri -> lejant yok (baslik zaten seriyi adlandirir). Buyukluk sirali
    tek hue ile kodlanir; kategorik palet buraya girmez.
    """
    if not values:
        return go.Figure(layout=layout(height=height))
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    steps = len(SEQUENTIAL) - 1
    colors = [SEQUENTIAL[min(steps, int((v - low) / span * steps))] for v in values]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertext=hover, hovertemplate="%{y}<br>%{x:.4g}<extra></extra>",
        )
    )
    fig.update_layout(**layout(height=height, hovermode="closest",
                               yaxis=dict(autorange="reversed", showgrid=False, linecolor=GRID),
                               xaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False)))
    fig.update_traces(marker_cornerradius=4)
    return fig


def status_of(value: Optional[float], low: Optional[float], high: Optional[float]) -> str:
    """Spec durumunu ada cevirir. Renk TEK BASINA anlam tasimaz; cagiran taraf
    bunu metin etiketiyle birlikte gosterir."""
    if value is None:
        return "warning"
    if low is not None and value < low:
        return "critical"
    if high is not None and value > high:
        return "critical"
    if low is not None and high is not None:
        margin = (high - low) * 0.12
        if value < low + margin or value > high - margin:
            return "serious"
    return "good"


STATUS_LABEL = {
    "good": "spec ici",
    "serious": "spec sinirina yakin",
    "critical": "SPEC DISI",
    "warning": "veri yok",
}
