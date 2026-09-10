"""Simulator calistirici -- canonical kayitlari depoya yazar.

    # 60 gunluk gecmis uret
    python -m twin.simulator.run history --days 60

    # canli akis (dashboard'un hareket etmesi icin)
    python -m twin.simulator.run live --speed 60
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from twin.config import get_profile
from twin.schema import Dataset, Medium
from twin.simulator.base import ProcessSimulator, SimStep
from twin.storage import get_repository

log = logging.getLogger("twin.simulator")

FLUSH_EVERY = 40_000


def build_simulator(profile_name: Optional[str] = None, seed: int = 42, start=None) -> ProcessSimulator:
    profile = get_profile(profile_name)
    if profile.name == "paper":
        from twin.simulator.paper import PaperMachineSimulator

        return PaperMachineSimulator(profile, seed=seed, start=start)
    if profile.name == "steel":
        from twin.simulator.steel import SteelPlantSimulator

        return SteelPlantSimulator(profile, seed=seed, start=start)
    raise ValueError(f"Bu profil icin simulator yok: {profile.name}")


class SimulationWriter:
    """SimStep -> canonical kayitlar -> depo."""

    def __init__(self, sim: ProcessSimulator, resolution_min: int = 1) -> None:
        self.sim = sim
        self.profile = sim.profile
        self.repo = get_repository()
        self.resolution = max(1, resolution_min)
        self.buffers: Dict[Dataset, List[Dict[str, Any]]] = {d: [] for d in Dataset}
        self.counts: Dict[str, int] = {d.value: 0 for d in Dataset}
        self._energy_acc = {"electricity": 0.0, "steam": 0.0, "natural_gas": 0.0, "tons": 0.0}

    # ------------------------------------------------------------------ #
    def consume(self, step: SimStep) -> None:
        emit_telemetry = step.ts.minute % self.resolution == 0

        if emit_telemetry:
            for name, value in step.values.items():
                if name not in self.profile.variables:
                    continue
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    continue
                var = self.profile.var(name)
                self.buffers[Dataset.TELEMETRY].append(
                    {
                        "ts": step.ts.isoformat(),
                        "line_id": self.profile.line_id,
                        "asset_id": var.asset,
                        "tag": var.tag,
                        "variable": name,
                        "value": round(float(value), 4),
                        "unit": var.unit,
                        "role": var.role,
                        "quality": 100,
                        "batch_id": step.batch_id,
                    }
                )

        self._accumulate_energy(step)

        for event in step.events:
            self.buffers[Dataset.EVENTS].append(
                {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in event.items()}
            )
        for sample in step.quality:
            row = dict(sample)
            row["ts"] = row["ts"].isoformat()
            lo = row.get("spec_min")
            hi = row.get("spec_max")
            row["passed"] = bool(
                (lo is None or row["value"] >= lo) and (hi is None or row["value"] <= hi)
            )
            self.buffers[Dataset.QUALITY].append(row)
        if step.closed_batch:
            batch = dict(step.closed_batch)
            batch["start_ts"] = batch["start_ts"].isoformat()
            batch["end_ts"] = batch["end_ts"].isoformat()
            self.buffers[Dataset.BATCHES].append(batch)

        if sum(len(b) for b in self.buffers.values()) >= FLUSH_EVERY:
            self.flush()

    # ------------------------------------------------------------------ #
    def _accumulate_energy(self, step: SimStep) -> None:
        """Sayac okumalarini saatlik uretir -- gercek fabrikadaki gibi."""
        rate = step.values.get("production_rate_tph") or 0.0
        if isinstance(rate, float) and math.isnan(rate):
            rate = 0.0
        tons = rate / 60.0
        elec_kwh_t = step.values.get("elec_kwh_t") or step.values.get("eaf_elec_kwh_t") or 0.0
        thermal_kwh_t = step.values.get("thermal_kwh_t") or step.values.get("reheat_gas_kwh_t") or 0.0
        for key, value in (("electricity", elec_kwh_t), ("steam", thermal_kwh_t)):
            if isinstance(value, float) and math.isnan(value):
                value = 0.0
            self._energy_acc[key] += value * tons
        self._energy_acc["tons"] += tons

        if step.ts.minute != 0:
            return
        medium_map = {
            "electricity": (Medium.ELECTRICITY.value, f"EM-{self.profile.line_id}-ELEC"),
            "steam": (
                Medium.STEAM.value if self.profile.name == "paper" else Medium.NATURAL_GAS.value,
                f"EM-{self.profile.line_id}-THERMAL",
            ),
        }
        for key, (medium, meter) in medium_map.items():
            self.buffers[Dataset.ENERGY].append(
                {
                    "ts": step.ts.isoformat(),
                    "meter_id": meter,
                    "asset_id": self.profile.line_id,
                    "line_id": self.profile.line_id,
                    "medium": medium,
                    "value": round(self._energy_acc[key], 3),
                    "unit": "kWh",
                    "batch_id": step.batch_id,
                }
            )
            self._energy_acc[key] = 0.0
        self._energy_acc["tons"] = 0.0

    # ------------------------------------------------------------------ #
    def flush(self) -> None:
        for dataset, rows in self.buffers.items():
            if not rows:
                continue
            written = self.repo.write(dataset, rows)
            self.counts[dataset.value] += written
            rows.clear()

    def summary(self) -> Dict[str, int]:
        return {k: v for k, v in self.counts.items() if v}


# --------------------------------------------------------------------------- #
def generate_history(days: int, profile_name: Optional[str], seed: int, resolution: int) -> Dict[str, int]:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    sim = build_simulator(profile_name, seed=seed, start=start)
    writer = SimulationWriter(sim, resolution_min=resolution)
    writer.repo.ensure_schema()

    total = days * 24 * 60
    log.info("%s profili icin %d gun (%d adim) uretiliyor...", sim.profile.name, days, total)
    checkpoint = max(total // 20, 1)
    for i in range(total):
        writer.consume(sim.step())
        if (i + 1) % checkpoint == 0:
            log.info("  %3d%%  (%s)", round(100 * (i + 1) / total), sim.ts.strftime("%Y-%m-%d %H:%M"))
    writer.flush()
    return writer.summary()


def stream_live(speed: float, profile_name: Optional[str], seed: int, minutes: Optional[int]) -> None:
    """Gercek zamanli akis. speed=60 -> 1 sn'de 1 dakika proses zamani."""
    repo = get_repository()
    profile = get_profile(profile_name)
    last = repo.max_ts(Dataset.TELEMETRY, filters={"line_id": profile.line_id})
    start = last or (datetime.now(timezone.utc) - timedelta(hours=1))
    sim = build_simulator(profile_name, seed=seed, start=start)
    writer = SimulationWriter(sim, resolution_min=1)
    writer.repo.ensure_schema()

    log.info("Canli akis basladi (%s, hiz x%.0f). Ctrl-C ile durdurun.", profile.name, speed)
    emitted = 0
    try:
        while minutes is None or emitted < minutes:
            writer.consume(sim.step())
            emitted += 1
            if emitted % 5 == 0:            # 5 dakikada bir depoya yaz
                writer.flush()
            time.sleep(60.0 / speed)
    except KeyboardInterrupt:
        log.info("Durduruldu.")
    finally:
        writer.flush()
        log.info("Yazilan: %s", writer.summary())


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Fabrika simulatoru")
    parser.add_argument("--profile", default=None, help="paper | steel (varsayilan: settings.yaml)")
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="mode", required=True)

    hist = sub.add_parser("history", help="Gecmis veri uret")
    hist.add_argument("--days", type=int, default=60)
    hist.add_argument("--resolution", type=int, default=1, help="Telemetri ornekleme araligi (dk)")

    live = sub.add_parser("live", help="Canli akis")
    live.add_argument("--speed", type=float, default=60.0, help="Hizlandirma carpani")
    live.add_argument("--minutes", type=int, default=None, help="Kac proses dakikasi uretilsin")

    args = parser.parse_args(argv)
    if args.mode == "history":
        summary = generate_history(args.days, args.profile, args.seed, args.resolution)
        log.info("Tamamlandi. Yazilan kayitlar: %s", summary)
    else:
        stream_live(args.speed, args.profile, args.seed, args.minutes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
