"""Musteri verisi -> canonical sema.

**Projenin tamaminda, gercek veri geldiginde degisecek olan tek yer burasidir.**
Feature/model/optimizasyon/dashboard katmanlari canonical semaya bakar ve
verinin nereden geldigini bilmez.

Desteklenen giris bicimleri:
  * GENIS (wide) CSV/Parquet   -- historian ihracatinin standart bicimi
        timestamp, 10SIC0102.SP, 10PIC0613.SP, ...
  * UZUN (long) CSV/Parquet    -- OPC-UA / Kafka akislarinin bicimi
        timestamp, tag, value, quality

Kullanim:
    python -m twin.ingest.loader telemetry data/raw/telemetry_2026*.csv --wide
    python -m twin.ingest.loader batches   data/raw/production.csv
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from twin.config import Profile, get_profile
from twin.ingest.validate import QualityReport, validate_telemetry
from twin.schema import Dataset
from twin.storage import Repository, get_repository

log = logging.getLogger("twin.ingest")

# Musteri kolon adi -> canonical alan adi. Yeni musteride burasi genisletilir.
COLUMN_ALIASES = {
    "timestamp": "ts", "time": "ts", "datetime": "ts", "date_time": "ts", "tarih": "ts",
    "tagname": "tag", "tag_name": "tag", "point": "tag",
    "val": "value", "pv": "value", "deger": "value",
    "batch": "batch_id", "reel_id": "batch_id", "heat_id": "batch_id", "lot": "batch_id",
    "product": "product_code", "grade": "product_code", "urun": "product_code",
    "line": "line_id", "machine": "line_id", "hat": "line_id",
    "start": "start_ts", "start_time": "start_ts", "baslangic": "start_ts",
    "end": "end_ts", "end_time": "end_ts", "bitis": "end_ts",
    "qty": "produced_qty", "produced": "produced_qty", "uretim": "produced_qty",
    "scrap": "scrap_qty", "waste": "scrap_qty", "broke": "scrap_qty", "iskarta": "scrap_qty",
    "reason": "reason_code", "reason_cd": "reason_code",
    "downtime_start": "ts_start", "downtime_end": "ts_end",
}

# Birim donusumleri: (kaynak, hedef) -> carpan/fonksiyon
UNIT_CONVERSIONS = {
    ("kpa", "bar"): lambda v: v / 100.0,
    ("bar", "kpa"): lambda v: v * 100.0,
    ("psi", "bar"): lambda v: v * 0.0689476,
    ("f", "c"): lambda v: (v - 32.0) * 5.0 / 9.0,
    ("k", "c"): lambda v: v - 273.15,
    ("kg/h", "t/h"): lambda v: v / 1000.0,
    ("mwh", "kwh"): lambda v: v * 1000.0,
    ("mpm", "m/min"): lambda v: v,
}


def read_any(pattern: str) -> pd.DataFrame:
    """CSV / Parquet / Excel -- glob desenini kabul eder."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Dosya bulunamadi: {pattern}")
    frames = []
    for path in paths:
        suffix = Path(path).suffix.lower()
        if suffix in (".parquet", ".pq"):
            frames.append(pd.read_parquet(path))
        elif suffix in (".xlsx", ".xls"):
            frames.append(pd.read_excel(path))
        else:
            frames.append(pd.read_csv(path, sep=None, engine="python"))
        log.info("Okundu: %s (%d satir)", path, len(frames[-1]))
    return pd.concat(frames, ignore_index=True)


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace(" ", "_")
        renamed[col] = COLUMN_ALIASES.get(key, key)
    return frame.rename(columns=renamed)


def to_utc(series: pd.Series, source_tz: Optional[str] = None) -> pd.Series:
    """Zaman damgalarini UTC'ye cevirir.

    Musteri verisi neredeyse her zaman lokal saattedir ve yaz saati gecislerinde
    ya tekrar eden ya da hic olmayan saatler icerir. `ambiguous`/`nonexistent`
    politikalarini acikca belirtmezsek pandas patlar ve is durur.
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=source_tz is None)
    if source_tz:
        parsed = (
            parsed.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
            .dt.tz_convert("UTC")
        )
    return parsed


def convert_unit(values: pd.Series, source_unit: str, target_unit: str) -> pd.Series:
    if not source_unit or not target_unit:
        return values
    key = (source_unit.strip().lower(), target_unit.strip().lower())
    if key[0] == key[1]:
        return values
    converter = UNIT_CONVERSIONS.get(key)
    if converter is None:
        log.warning("Birim donusumu tanimsiz: %s -> %s (deger degistirilmedi)", *key)
        return values
    return converter(values)


# --------------------------------------------------------------------------- #
def wide_to_canonical(
    frame: pd.DataFrame,
    profile: Profile,
    source_tz: Optional[str] = None,
    unit_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Genis historian ihracati -> canonical telemetri (long).

    Tag -> canonical degisken eslemesi `profile.tag_to_name` uzerinden yapilir;
    yani proses sozlugu tek kaynaktan gelir.
    """
    frame = normalize_columns(frame)
    if "ts" not in frame.columns:
        raise ValueError("Zaman kolonu bulunamadi (timestamp/time/datetime/tarih)")

    frame["ts"] = to_utc(frame["ts"], source_tz)
    frame = frame.dropna(subset=["ts"])

    tag_to_name = {k.strip().lower(): v for k, v in profile.tag_to_name.items()}
    known = {c: tag_to_name[str(c).lower()] for c in frame.columns
             if str(c).lower() in tag_to_name}
    unknown = [c for c in frame.columns if c not in known and c != "ts"]
    if unknown:
        log.warning("Eslesmeyen %d kolon atlandi (profile.yaml'a tag ekleyin): %s",
                    len(unknown), unknown[:8])
    if not known:
        raise ValueError("Hicbir kolon profildeki tag'lerle eslesmedi -- profile.yaml tag alanlarini kontrol edin")

    melted = frame[["ts"] + list(known)].melt(id_vars="ts", var_name="tag", value_name="value")
    melted["variable"] = melted["tag"].map(known)
    melted = melted.dropna(subset=["value"])

    unit_map = unit_map or {}
    rows: List[pd.DataFrame] = []
    for name, part in melted.groupby("variable"):
        var = profile.var(str(name))
        values = convert_unit(part["value"].astype(float), unit_map.get(str(name), var.unit), var.unit)
        rows.append(part.assign(value=values, unit=var.unit, role=var.role, asset_id=var.asset))
    canonical = pd.concat(rows, ignore_index=True)
    canonical["line_id"] = profile.line_id
    canonical["quality"] = 100
    canonical["batch_id"] = None
    return canonical[["ts", "line_id", "asset_id", "tag", "variable",
                      "value", "unit", "role", "quality", "batch_id"]]


def long_to_canonical(
    frame: pd.DataFrame, profile: Profile, source_tz: Optional[str] = None
) -> pd.DataFrame:
    frame = normalize_columns(frame)
    for required in ("ts", "tag", "value"):
        if required not in frame.columns:
            raise ValueError(f"Zorunlu kolon eksik: {required}")

    frame["ts"] = to_utc(frame["ts"], source_tz)
    tag_to_name = {k.strip().lower(): v for k, v in profile.tag_to_name.items()}
    frame["variable"] = frame["tag"].astype(str).str.lower().map(tag_to_name)

    unmatched = frame["variable"].isna().sum()
    if unmatched:
        log.warning("%d satirin tag'i eslesmedi ve atlandi", unmatched)
    frame = frame.dropna(subset=["ts", "variable", "value"])

    meta = frame["variable"].map(lambda n: profile.var(n))
    frame["unit"] = [v.unit for v in meta]
    frame["role"] = [v.role for v in meta]
    frame["asset_id"] = [v.asset for v in meta]
    frame["line_id"] = profile.line_id
    if "quality" not in frame.columns:
        frame["quality"] = 100
    if "batch_id" not in frame.columns:
        frame["batch_id"] = None
    return frame[["ts", "line_id", "asset_id", "tag", "variable",
                  "value", "unit", "role", "quality", "batch_id"]]


# --------------------------------------------------------------------------- #
def ingest(
    dataset: Dataset,
    pattern: str,
    wide: bool = False,
    source_tz: Optional[str] = None,
    profile: Optional[Profile] = None,
    repo: Optional[Repository] = None,
    dry_run: bool = False,
) -> QualityReport:
    """Dosyadan oku -> canonical'a cevir -> kalite kapisindan gecir -> yaz.

    Kapidan gecemeyen kayitlar SESSIZCE DUSURULMEZ; `quarantine` veri kumesine
    yazilir ve raporda sayilir. "Veri temizlendi" demek yerine "su kadar kayit
    su sebeple ayrildi" demek zorundayiz.
    """
    profile = profile or get_profile()
    repo = repo or get_repository()
    raw = read_any(pattern)

    if dataset is Dataset.TELEMETRY:
        canonical = wide_to_canonical(raw, profile, source_tz) if wide \
            else long_to_canonical(raw, profile, source_tz)
        accepted, rejected, report = validate_telemetry(canonical, profile)
    else:
        canonical = normalize_columns(raw)
        for col in [c for c in canonical.columns if c == "ts" or c.endswith("_ts")]:
            canonical[col] = to_utc(canonical[col], source_tz)
        if "line_id" not in canonical.columns:
            canonical["line_id"] = profile.line_id
        accepted, rejected = canonical, pd.DataFrame()
        report = QualityReport(dataset=dataset.value, total=len(canonical), accepted=len(canonical))

    if dry_run:
        log.info("DRY RUN -- yazilmadi")
        return report

    repo.ensure_schema()
    written = repo.write_frame(dataset, accepted)
    if not rejected.empty:
        repo.write(Dataset.QUARANTINE, [
            {"ts": r.get("ts"), "dataset": dataset.value, "reason": r.get("_reject_reason"),
             "payload": {k: v for k, v in r.items() if not k.startswith("_")}}
            for r in rejected.to_dict(orient="records")
        ])
    log.info("%d kayit yazildi, %d karantinaya alindi", written, len(rejected))
    return report


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Musteri verisi yukleyici")
    parser.add_argument("dataset", choices=[d.value for d in Dataset])
    parser.add_argument("pattern", help="Dosya yolu veya glob deseni")
    parser.add_argument("--wide", action="store_true", help="Genis (pivot) bicim")
    parser.add_argument("--tz", default=None, help="Kaynak saat dilimi, orn. Europe/Istanbul")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = ingest(
        Dataset(args.dataset), args.pattern, wide=args.wide, source_tz=args.tz,
        profile=get_profile(args.profile), dry_run=args.dry_run,
    )
    print(report.render())
    return 0


if __name__ == "__main__":
    sys.exit(main())
