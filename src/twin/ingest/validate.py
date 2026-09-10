"""Veri kalite kapisi.

Faz 1'in (Veri Kesfi) somut ciktisi bu modulun urettigi rapordur. Musteriye
"veriniz kotu" demek yerine "su tag'in %31'i fiziksel aralik disinda, su tag 4
gundur sabit deger basiyor" demek gerekir -- ancak boyle duzeltilebilir.

Kontroller:
  1. Fiziksel aralik disi     -> sensor arizasi / birim hatasi
  2. Donuk (stuck) sensor     -> uzun sure hic degismeyen deger
  3. Zaman boslugu            -> historian kesintisi
  4. Cift kayit               -> ayni tag + ayni zaman damgasi
  5. Kalite bayragi           -> historian'in kendi "Bad" isareti
  6. Varyans yoklugu          -> modelin ogrenebilecegi bilgi yok
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from twin.config import Profile

log = logging.getLogger(__name__)

STUCK_MINUTES = 120          # bu sure boyunca hic degismeyen sensor supheli
GAP_MINUTES = 30             # bundan uzun bosluk raporlanir
MIN_QUALITY = 50             # historian kalite bayragi esigi


@dataclass
class QualityReport:
    dataset: str
    total: int = 0
    accepted: int = 0
    rejections: Dict[str, int] = field(default_factory=dict)
    rejected_by_variable: List[Dict[str, Any]] = field(default_factory=list)
    per_variable: List[Dict[str, Any]] = field(default_factory=list)
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def rejected(self) -> int:
        return self.total - self.accepted

    def render(self) -> str:
        lines = [
            "",
            "=" * 74,
            f"  VERI KALITE RAPORU -- {self.dataset}",
            "=" * 74,
            f"  Toplam kayit      : {self.total:,}",
            f"  Kabul edilen      : {self.accepted:,}  (%{self.accepted / max(self.total, 1) * 100:.2f})",
            f"  Karantinaya alinan: {self.rejected:,}",
        ]
        if self.rejections:
            lines.append("\n  Ret sebepleri:")
            for reason, count in sorted(self.rejections.items(), key=lambda kv: -kv[1]):
                lines.append(f"    - {reason:<28} {count:>10,}")
        if self.rejected_by_variable:
            lines.append("\n  En cok reddedilen degiskenler:")
            for item in self.rejected_by_variable[:8]:
                lines.append(
                    f"    - {item['degisken']:<28} {item['reddedilen']:>8,} / {item['toplam']:<8,}"
                    f" (%{item['oran']:.1f})  {item['sebep']}"
                )
                if item["ipucu"]:
                    lines.append(f"        -> {item['ipucu']}")
        if self.per_variable:
            frame = pd.DataFrame(self.per_variable)
            problems = frame[frame["durum"] != "OK"]
            lines.append(f"\n  Degisken sayisi: {len(frame)} | sorunlu: {len(problems)}")
            if not problems.empty:
                lines.append("\n" + problems.to_string(index=False))
        if self.gaps:
            lines.append(f"\n  Zaman bosluklari ({len(self.gaps)} adet, en uzun 5):")
            for gap in sorted(self.gaps, key=lambda g: -g["minutes"])[:5]:
                lines.append(f"    - {gap['start']} -> {gap['end']}  ({gap['minutes']:.0f} dk)")
        for warning in self.warnings:
            lines.append(f"\n  ! {warning}")
        lines.append("=" * 74)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _stuck_ratio(series: pd.Series, periods: int) -> float:
    """Ard arda `periods` adet ayni deger gelen noktalarin orani."""
    if len(series) < periods or periods < 2:
        return 0.0
    unchanged = series.diff().abs() < 1e-12
    run = unchanged.rolling(periods, min_periods=periods).sum()
    return float((run >= periods - 1).mean())


def validate_telemetry(
    frame: pd.DataFrame, profile: Profile
) -> Tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    """(kabul, red, rapor) doner. Reddedilen satirlarda `_reject_reason` bulunur."""
    report = QualityReport(dataset="telemetry", total=len(frame))
    if frame.empty:
        report.warnings.append("Veri bos.")
        return frame, frame, report

    work = frame.copy()
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    work["_reject_reason"] = None

    def mark(mask: pd.Series, reason: str) -> None:
        fresh = mask & work["_reject_reason"].isna()
        count = int(fresh.sum())
        if count:
            work.loc[fresh, "_reject_reason"] = reason
            report.rejections[reason] = report.rejections.get(reason, 0) + count

    mark(work["ts"].isna(), "gecersiz zaman damgasi")
    mark(work["value"].isna(), "bos deger")
    mark(~np.isfinite(work["value"].astype(float, errors="ignore")), "sonsuz/NaN deger")
    if "quality" in work.columns:
        mark(work["quality"].fillna(100) < MIN_QUALITY, "historian kalite bayragi dusuk")

    duplicated = work.duplicated(subset=["ts", "variable"], keep="first")
    mark(duplicated, "cift kayit (ayni tag + zaman)")

    # --- fiziksel aralik: emniyet araliginin %25 disi tolere edilir --------- #
    for name, part in work.groupby("variable"):
        if name not in profile.variables:
            continue
        var = profile.var(str(name))
        if var.op_low is None or var.op_high is None:
            continue
        span = var.op_high - var.op_low
        low, high = var.op_low - 0.25 * span, var.op_high + 0.25 * span
        outside = part.index[~part["value"].between(low, high)]
        mark(work.index.isin(outside), "fiziksel aralik disi")

    accepted = work[work["_reject_reason"].isna()].drop(columns=["_reject_reason"])
    rejected = work[work["_reject_reason"].notna()]
    report.accepted = len(accepted)

    # Hangi TAG reddediliyor? Toplam sayi tek basina aksiyona donusmez;
    # "su tag'in %100'u aralik disi" ise sebep neredeyse her zaman BIRIM
    # hatasidir (kPa yerine bar gonderilmis gibi) ve tek satirlik bir
    # duzeltmeyle cozulur. Rapor bunu soylemezse haftalar kaybedilir.
    if not rejected.empty and "variable" in work.columns:
        totals = work.groupby("variable").size()
        for name, part in rejected.groupby("variable"):
            total = int(totals.get(name, len(part)))
            ratio = 100.0 * len(part) / max(total, 1)
            reason = part["_reject_reason"].mode().iloc[0]
            hint = ""
            if ratio > 95 and reason == "fiziksel aralik disi":
                hint = ("Tag'in TAMAMI aralik disi -- neredeyse kesin BIRIM/OLCEK hatasi. "
                        "Beklenen birimi profile.yaml ile karsilastirin.")
            elif ratio > 40:
                hint = "Reddedilme orani cok yuksek; sensor arizasi veya yanlis tag eslemesi olabilir."
            report.rejected_by_variable.append({
                "degisken": str(name), "reddedilen": len(part), "toplam": total,
                "oran": ratio, "sebep": reason, "ipucu": hint,
            })
        report.rejected_by_variable.sort(key=lambda r: -r["reddedilen"])

    # --- degisken bazli tani ------------------------------------------------ #
    step = _median_step_minutes(accepted)
    stuck_periods = max(2, int(STUCK_MINUTES / max(step, 1e-6)))

    for name, part in accepted.groupby("variable"):
        part = part.sort_values("ts")
        values = part["value"].astype(float)
        stuck = _stuck_ratio(values, stuck_periods)
        variance = float(values.std())
        span = float(values.max() - values.min())

        distinct = int(values.nunique())
        role = profile.var(str(name)).role if name in profile.variables else "measurement"

        status = "OK"
        if len(part) < 10:
            status = "COK AZ VERI"
        elif stuck > 0.6 and role == "context" and 1 < distinct <= 12:
            # Recete/siparis degerleri (hedef gramaj, slab kalinligi...) kampanya
            # boyunca sabit kalir. Bu bir sensor arizasi degil, prosesin dogasidir;
            # "donuk sensor" diye isaretlemek raporu gurultuye bogar.
            status = "KADEMELI (recete degeri)"
        elif stuck > 0.6:
            status = "DONUK SENSOR"
        elif variance < 1e-9:
            status = "SABIT (varyans yok)"
        elif name in profile.variables and profile.var(str(name)).role == "setpoint" and span < 1e-6:
            # Optimizasyon icin oldurucu: hic degismemis setpoint ogrenilemez
            status = "SETPOINT HIC DEGISMEMIS"

        report.per_variable.append({
            "degisken": name,
            "kayit": len(part),
            "ortalama": round(float(values.mean()), 3),
            "std": round(variance, 4),
            "farkli_deger": distinct,
            "sabit_kalma_%": round(stuck * 100, 1),
            "durum": status,
        })

    frozen = [r["degisken"] for r in report.per_variable if r["durum"] == "SETPOINT HIC DEGISMEMIS"]
    if frozen:
        report.warnings.append(
            f"{len(frozen)} setpoint hic degismemis ({', '.join(frozen[:5])}). "
            "Model bu degiskenlerin etkisini OGRENEMEZ; optimizasyonda kullanilamazlar. "
            "Cozum: kontrollu deney tasarimi (DOE) veya daha genis tarihsel aralik."
        )

    report.gaps = _find_gaps(accepted, step)
    if report.gaps:
        lost = sum(g["minutes"] for g in report.gaps)
        report.warnings.append(f"Toplam {lost:,.0f} dakikalik veri boslugu var.")

    coverage_days = (accepted["ts"].max() - accepted["ts"].min()).total_seconds() / 86400.0 \
        if not accepted.empty else 0.0
    if coverage_days < 180:
        report.warnings.append(
            f"Veri derinligi {coverage_days:.0f} gun. Mevsimsel etkiyi (ortam nem/sicaklik) "
            "yakalamak icin en az 180 gun, ideal 365 gun gerekir."
        )
    return accepted, rejected, report


def _median_step_minutes(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 1.0
    sample = frame.sort_values("ts")["ts"].drop_duplicates().head(5000)
    if len(sample) < 3:
        return 1.0
    step = sample.diff().dt.total_seconds().median() / 60.0
    return float(step) if step and step > 0 else 1.0


def _find_gaps(frame: pd.DataFrame, step_min: float) -> List[Dict[str, Any]]:
    if frame.empty:
        return []
    stamps = frame["ts"].drop_duplicates().sort_values()
    deltas = stamps.diff().dt.total_seconds() / 60.0
    threshold = max(GAP_MINUTES, step_min * 5)
    gaps = []
    for end, delta in zip(stamps[deltas > threshold], deltas[deltas > threshold]):
        gaps.append({
            "start": (end - pd.Timedelta(minutes=float(delta))).isoformat(),
            "end": end.isoformat(),
            "minutes": float(delta),
        })
    return gaps
