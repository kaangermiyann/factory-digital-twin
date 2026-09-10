#!/usr/bin/env python3
"""Musteriye gonderilecek `sensor_registry` sablonunu uretir.

    python scripts/export_sensor_registry.py --profile paper -o data/raw/sensor_registry.csv

Neden bu script var:
    docs/01-veri-sozlesmesi.md musteriden `sensor_registry` istiyor. "Bize sensor
    listesi gonderin" demek yerine, DOLDURULMAYA HAZIR bir sablon gondermek
    cevap oranini ciddi sekilde artirir. Profilimizdeki ornek tag'ler ornek satir
    olarak gider; musteri kendi tag'leriyle degistirir.

    `role` kolonu bilerek EN ONE alinmistir: bu kolon doldurulmazsa optimizasyon
    yapilamaz, sadece tahmin teslim edilebilir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twin.config import get_profile  # noqa: E402

COLUMN_HELP = {
    "tag": "Historian'daki tam tag adi, orn. 10FIC0234.PV",
    "role": "setpoint = operator degistirebilir | measurement = prosesin ciktisi | context = disaridan gelir",
    "description": "Turkce aciklama -- proses muhendisinin anlayacagi dilde",
    "asset_id": "Hangi ekipman/bolum (plant.area.line.machine)",
    "unit": "bar / kPa / C / m-min / kWh-t / % ...",
    "low_limit": "Fiziksel gecerli alt sinir (bunun disi sensor arizasi sayilir)",
    "high_limit": "Fiziksel gecerli ust sinir",
    "op_low": "OPERASYONEL emniyet alt siniri -- optimizasyon bu sinirin altina inmez",
    "op_high": "OPERASYONEL emniyet ust siniri",
    "max_rate_of_change": "5 dakikada izin verilen maksimum degisim",
    "sampling_period_s": "Ornekleme periyodu (saniye)",
    "notes": "Ek not (deadband, kompresyon, bilinen ariza gecmisi...)",
}


def build(profile_name: str) -> pd.DataFrame:
    profile = get_profile(profile_name)
    rows = []
    for var in profile.variables.values():
        rows.append({
            "tag": var.tag,
            "role": var.role,
            "description": var.desc,
            "asset_id": var.asset,
            "unit": var.unit,
            "low_limit": "",
            "high_limit": "",
            "op_low": var.op_low if var.op_low is not None else "",
            "op_high": var.op_high if var.op_high is not None else "",
            "max_rate_of_change": var.max_delta if var.max_delta is not None else "",
            "sampling_period_s": 60,
            "notes": "",
        })
    return pd.DataFrame(rows, columns=list(COLUMN_HELP))


def main() -> int:
    parser = argparse.ArgumentParser(description="sensor_registry sablonu uret")
    parser.add_argument("--profile", default="paper", help="paper | steel")
    parser.add_argument("-o", "--output", default="data/raw/sensor_registry_template.csv")
    args = parser.parse_args()

    frame = build(args.profile)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() in (".xlsx", ".xls"):
        with pd.ExcelWriter(output) as writer:
            frame.to_excel(writer, sheet_name="sensor_registry", index=False)
            pd.DataFrame(
                [{"kolon": k, "aciklama": v} for k, v in COLUMN_HELP.items()]
            ).to_excel(writer, sheet_name="kolon_aciklamalari", index=False)
    else:
        # CSV'de aciklamalari yorum satiri olarak basa koy
        with output.open("w", encoding="utf-8-sig") as handle:
            for column, help_text in COLUMN_HELP.items():
                handle.write(f"# {column}: {help_text}\n")
            frame.to_csv(handle, index=False)

    print(f"Yazildi: {output}  ({len(frame)} satir)")
    print("\nMusteriye giderken eklenecek not:")
    print("  'role' kolonu doldurulmadan optimizasyon yapilamaz. Bu kolon,")
    print("  operatorun gercekten cevirebildigi kollari isaretler.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
