"""Altin recete (golden recipe) tablosu.

    python -m twin.optimize.golden_recipe -o data/models/golden_recipe.csv

Neden bu, projenin en somut teslimati?
    Canli advisory guzeldir ama ikiz kapandiginda geriye bir sey kalmaz.
    Altin recete tablosu, "TL80 uretirken, hava soguk ve gece tarifesindeyken
    setpointler sunlar olmali" bilgisini KAGIT UZERINDE birakir. Operatorun
    ekranindan bagimsiz olarak kullanabilecegi tek ciktidir; cogu musteri icin
    en cok deger goren teslimat budur.

Yontem:
    Gecmis veri anlamli segmentlere bolunur (urun x ortam x tarife). Her segment
    icin TEMSILI bir calisma noktasi (medyan) cikarilir ve o nokta uzerinde
    kisitli optimizasyon kosulur. Cikti, segment basina bir satir.

Uyari:
    Az ornekli segmentler icin uretilen recete guvenilmezdir; tablo `n` ve
    `guven` kolonlarini bilerek tasir ve dusuk destekli satirlar isaretlenir.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from twin.config import Profile, get_profile, get_settings
from twin.features.build import DOWNTIME_FLAG, build_feature_table
from twin.models.registry import load_all
from twin.optimize.objective import Economics
from twin.optimize.search import optimize_row

log = logging.getLogger("twin.golden")

MIN_SEGMENT_ROWS = 40          # bunun altinda recete uretmeyi reddediyoruz

AMBIENT_BINS = [(-99.0, 10.0, "soguk"), (10.0, 20.0, "ilik"), (20.0, 99.0, "sicak")]


def tariff_period(hour: int, economics: Economics) -> str:
    if hour in economics.peak_hours:
        return "puant"
    if hour in economics.offpeak_hours:
        return "gece"
    return "gunduz"


def ambient_band(temp: float) -> str:
    for low, high, label in AMBIENT_BINS:
        if low <= temp < high:
            return label
    return "bilinmiyor"


def _representative_row(part: pd.DataFrame) -> pd.Series:
    """Segmentin temsili calisma noktasi: sayisal kolonlarin medyani.

    Medyan, ortalamadan iyidir -- proses verisi duruş/gecis anlarinda uzun
    kuyruklar tasir ve ortalama gercekte hic gorulmemis bir noktayi tarif eder.
    """
    numeric = part.select_dtypes(include=[np.number]).median()
    row = pd.Series(numeric, dtype=float)
    row.name = pd.Timestamp(part["ts"].median())
    return row


def build_table(
    profile: Optional[Profile] = None,
    days: Optional[int] = None,
    min_rows: int = MIN_SEGMENT_ROWS,
) -> pd.DataFrame:
    settings = get_settings()
    profile = profile or get_profile()
    bundles = load_all(profile.name)
    if not bundles:
        raise RuntimeError("Egitilmis model yok. Once: python -m twin.models.train")

    start = None
    if days:
        start = pd.Timestamp.utcnow() - pd.Timedelta(days=days)
    frame = build_feature_table(start=start, profile=profile)
    if frame.empty:
        raise RuntimeError("Veri yok. Once: python -m twin.simulator.run history --days 90")

    frame = frame[frame.get(DOWNTIME_FLAG, 0) == 0].copy()
    economics = Economics.from_settings(settings)
    frame["_ambient"] = frame.get("ambient_temp_c", pd.Series(15.0, index=frame.index)).map(ambient_band)
    frame["_tariff"] = frame["ts"].dt.hour.map(lambda h: tariff_period(int(h), economics))

    rows: List[Dict[str, Any]] = []
    groups = frame.groupby(["product_code", "_ambient", "_tariff"], dropna=False)
    log.info("%d segment degerlendirilecek", len(groups))

    for (product, ambient, tariff), part in groups:
        if len(part) < min_rows:
            log.debug("Atlandi (yetersiz ornek): %s / %s / %s -> %d satir",
                      product, ambient, tariff, len(part))
            continue

        grade = profile.grades.get(str(product))
        row = _representative_row(part)
        hour = int(part["ts"].dt.hour.median())

        try:
            result = optimize_row(row, bundles, profile, grade, hour=hour,
                                  respect_rate_limits=False, settings=settings)
        except Exception as exc:  # pragma: no cover - tek segment tum tabloyu dusurmemeli
            log.warning("Segment optimize edilemedi (%s/%s/%s): %s", product, ambient, tariff, exc)
            continue

        record: Dict[str, Any] = {
            "urun": product,
            "ortam": ambient,
            "tarife": tariff,
            "n": len(part),
            "guven": result.confidence,
            "kisit_ihlali": "; ".join(result.violations) or "",
            "kazanc_TL_ton": round(result.saving_try_per_ton, 2),
        }
        for name, value in result.setpoints.items():
            record[f"{name}"] = round(value, 3)
            record[f"{name}__mevcut"] = round(result.current_setpoints[name], 3)
        for name, value in result.predicted.items():
            record[f"beklenen__{name}"] = round(value, 3)
        rows.append(record)

    if not rows:
        raise RuntimeError(
            f"Hicbir segment {min_rows} satir esigini gecemedi. "
            "Daha uzun gecmis uretin veya --min-rows dusurun."
        )

    table = pd.DataFrame(rows).sort_values(["urun", "ortam", "tarife"]).reset_index(drop=True)
    log.info("%d segment icin recete uretildi", len(table))
    return table


def render_markdown(table: pd.DataFrame, profile: Profile) -> str:
    """Operatore basilabilir ozet: sadece degisen setpointler ve kazanc."""
    lines = ["# Altin Recete Tablosu", ""]
    lines.append(f"Hat: **{profile.plant} / {profile.line_id}** — {profile.description}")
    lines.append("")
    lines.append("> Bu tablo, dijital ikiz kapali olsa bile kullanilabilir. Her satir, o")
    lines.append("> kosullarda kisitlari saglayan en dusuk maliyetli calisma noktasidir.")
    lines.append("")
    lines.append("## ⚠️ Kazanc rakamlari nasil okunmali")
    lines.append("")
    lines.append("Buradaki kazanclar **modelin tahminidir, olculmus tasarruf degildir.**")
    lines.append("Optimize edilen calisma noktasi tanimi geregi gecmiste sik gozlenmemis")
    lines.append("bir noktadir; model orada bir miktar iyimserdir. Uc kural:")
    lines.append("")
    lines.append("1. Bu sayilari **ust sinir** olarak okuyun, taahhut olarak degil.")
    lines.append("2. Guven sutunu 🟢 olmayan satirlari once **kucuk adimlarla deneyin**")
    lines.append("   (tavsiyenin yarisi kadar), sonucu olcun, sonra devam edin.")
    lines.append("3. Gercek kazanc, `docs/04-optimizasyon.md` icindeki **normalize edilmis")
    lines.append("   fark** yontemiyle dogrulanir: ayni baglamda model-tabani ile")
    lines.append("   gerceklesen tuketim karsilastirilir.")
    lines.append("")
    lines.append("Sozlesmeye yazilacak hedef icin `docs/05` bakin: dogrulanmis enerji")
    lines.append("iyilesmesi beklentisi **%2**'dir -- bilerek mutevazi tutulmustur.")
    lines.append("")

    for _, record in table.iterrows():
        badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(record["guven"], "")
        lines.append(f"## {record['urun']} · ortam: {record['ortam']} · tarife: {record['tarife']}")
        lines.append(f"{badge} guven **{record['guven']}** · {int(record['n'])} gozlem · "
                     f"beklenen kazanc **{record['kazanc_TL_ton']:.1f} ₺/ton**")
        if record["kisit_ihlali"]:
            lines.append(f"> ⚠️ {record['kisit_ihlali']}")
        lines.append("")
        lines.append("| Setpoint | Mevcut | Tavsiye | Δ | Birim |")
        lines.append("|---|---:|---:|---:|---|")
        for var in profile.controllables:
            current, target = f"{var.name}__mevcut", var.name
            if current not in record or target not in record:
                continue
            delta = record[target] - record[current]
            if abs(delta) < 1e-6:
                continue
            lines.append(f"| {var.desc} | {record[current]:.2f} | {record[target]:.2f} "
                         f"| {delta:+.2f} | {var.unit} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Altin recete tablosu uret")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--days", type=int, default=None, help="Son N gunun verisini kullan")
    parser.add_argument("--min-rows", type=int, default=MIN_SEGMENT_ROWS)
    parser.add_argument("-o", "--output", default="data/models/golden_recipe.csv")
    args = parser.parse_args(argv)

    profile = get_profile(args.profile)
    table = build_table(profile, args.days, args.min_rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    markdown = output.with_suffix(".md")
    markdown.write_text(render_markdown(table, profile), encoding="utf-8")

    print(f"\nYazildi: {output}  ve  {markdown}")
    summary = table[["urun", "ortam", "tarife", "n", "guven", "kazanc_TL_ton"]]
    print(summary.to_string(index=False))
    weighted = float((table["kazanc_TL_ton"] * table["n"]).sum() / table["n"].sum())
    print(f"\nGozlem-agirlikli ortalama kazanc: {weighted:.1f} TL/ton")
    low_confidence = int((table["guven"] != "high").sum())
    if low_confidence:
        print(
            f"\nUYARI: {low_confidence}/{len(table)} segment 'high' guven seviyesinde DEGIL.\n"
            "  Bu sayilar modelin tahminidir, olculmus tasarruf degildir; ust sinir olarak\n"
            "  okuyun. Once kucuk adimlarla deneyin ve sonucu olcun -- dogrulama yontemi\n"
            "  docs/04-optimizasyon.md icinde."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
