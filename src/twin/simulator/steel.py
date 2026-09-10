"""Demir-Celik (EAF + LF + surekli dokum + tav firini) simulatoru.

Kagit makinesinden yapisal fark: proses BATCH'tir. Setpointler heat basinda
secilir, heat boyunca sabit kalir. Bu yuzden feature pipeline'da "heat basi
ozet" mantigi devreye girer -- ayni kod, farkli zaman olcegi.

Bilincli birakilan optimizasyon boslugu:
  * Operator dokum sicakligini emniyet payiyla YUKSEK tutar  -> bosa giden elektrik
    (her +10 C tap sicakligi ~ +6 kWh/t)
  * Tav firininda hava fazlalik katsayisi yuksek birakilir   -> bacadan giden gaz
  * Oksijen/karbon (ucuz kimyasal enerji) az kullanilir      -> pahali elektrik ile telafi
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional

from twin.config import Profile
from twin.simulator.base import (
    OrnsteinUhlenbeck,
    ProcessSimulator,
    SimStep,
    clamp,
    crew_of,
    shift_of,
    sigmoid,
)

LIQUIDUS_C = {"S235JR": 1516.0, "S355J2": 1512.0, "C45E": 1495.0}
FIXED_OPS_MIN = 20.0                 # sarj + dokum + hazirlik
TRANSFER_LOSS_C = 112.0              # tap -> pota -> tandis zincirindeki toplam sicaklik kaybi
LF_HEAT_RATE_C_PER_MIN = 3.2
LF_ELEC_KWH_T_PER_MIN = 0.42

DOWNTIME_REASONS = [
    ("MEC-11", "Elektrot kirilmasi"),
    ("MEC-22", "Pota firini arizasi"),
    ("CST-31", "Surekli dokum breakout"),
    ("RHF-41", "Tav firini brulor arizasi"),
]


class SteelPlantSimulator(ProcessSimulator):
    def __init__(self, profile: Profile, seed: int = 42, start=None) -> None:
        super().__init__(profile, seed=seed, start=start)
        self.heat_size = float(profile.extras.get("heat_size_t", 80.0))

        self.scrap_quality = OrnsteinUhlenbeck(self.rng, mean=0.72, sigma=0.010, theta=0.006, low=0.40, high=0.95)
        self.scrap_moisture = OrnsteinUhlenbeck(self.rng, mean=1.6, sigma=0.06, theta=0.010, low=0.2, high=6.0)

        self.downtime_left = 0
        self.heat_minute = 0.0
        self.heat_length = 55.0
        self.product_code = self.pick_grade()
        self.batch_id = self._new_heat_id()
        self.batch_start = self.ts
        self.sp: Dict[str, float] = {}
        self.derived: Dict[str, float] = {}
        self._begin_heat()

    # ------------------------------------------------------------------ #
    def _new_heat_id(self) -> str:
        return f"HEAT-{self.ts:%Y%m%d}-{uuid.uuid4().hex[:5].upper()}"

    def _begin_heat(self) -> None:
        """Heat basinda setpointler secilir -- operator/ekip biasiyla."""
        crew = crew_of(self.ts, self.crews)
        self.product_code = self.pick_grade()
        grade = self.profile.grade(self.product_code)
        tap_lo, tap_hi = grade.spec("tap_temp_c") or (1610.0, 1665.0)

        def bounded(name: str, value: float) -> float:
            var = self.profile.var(name)
            return clamp(value, var.op_low, var.op_high)

        # Operator spec ortasi degil, UST tarafi hedefler: "pota donmasin".
        # Bu emniyet payi dogrudan bosa giden elektriktir.
        safety_bias = 0.62 + 0.30 * crew.steam_bias
        tap_target = tap_lo + safety_bias * (tap_hi - tap_lo) + 4.0 * self.rng.standard_normal()

        self.sp = {
            "active_power_mw": bounded("active_power_mw", 68.0 + 6.0 * crew.speed_bias / 10.0
                                       + 2.5 * self.rng.standard_normal()),
            "oxygen_nm3_t": bounded("oxygen_nm3_t", 31.0 + 4.0 * crew.discipline
                                    + 1.6 * self.rng.standard_normal()),
            "carbon_inject_kg_t": bounded("carbon_inject_kg_t", 13.0 + 2.5 * crew.discipline
                                          + 1.0 * self.rng.standard_normal()),
            "lime_kg_t": bounded("lime_kg_t", 42.0 + 55.0 * (0.75 - self.scrap_quality.value)
                                 + 2.0 * self.rng.standard_normal()),
            "tap_temp_target_c": bounded("tap_temp_target_c", tap_target),
            "lf_heating_min": bounded("lf_heating_min", 11.0 + 5.0 * (1.0 - crew.discipline)
                                      + 1.5 * self.rng.standard_normal()),
            "caster_speed_mmin": bounded("caster_speed_mmin", 1.10 + 0.10 * crew.discipline
                                         + 0.05 * self.rng.standard_normal()),
            "reheat_zone1_c": bounded("reheat_zone1_c", 1005.0 + 30.0 * crew.steam_bias
                                      + 8.0 * self.rng.standard_normal()),
            "reheat_zone2_c": bounded("reheat_zone2_c", 1155.0 + 30.0 * crew.steam_bias
                                      + 8.0 * self.rng.standard_normal()),
            "reheat_zone3_c": bounded("reheat_zone3_c", 1210.0 + 25.0 * crew.steam_bias
                                      + 6.0 * self.rng.standard_normal()),
            # Emniyet icin yuksek tutulan hava fazlaligi -> bacadan giden enerji
            "excess_air_ratio": bounded("excess_air_ratio", 1.14 + 0.05 * crew.steam_bias
                                        + 0.015 * self.rng.standard_normal()),
        }
        self.batch_id = self._new_heat_id()
        self.batch_start = self.ts
        self.heat_minute = 0.0
        self.derived = self._solve_heat()
        self.heat_length = self.derived["tap_to_tap_min"]

    # ------------------------------------------------------------------ #
    def _solve_heat(self) -> Dict[str, float]:
        """Heat'in enerji/sure dengesini cozer (tek adim sabit nokta iterasyonu)."""
        sp = self.sp
        charge = self.heat_size * (1.0 + 0.02 * self.rng.standard_normal())
        scrap_q = self.scrap_quality.value
        scrap_m = self.scrap_moisture.value

        # Ergitme icin gereken net enerji (kWh/t)
        required = (
            300.0
            + 0.62 * (sp["tap_temp_target_c"] - 1600.0)     # her 10 C ~ 6 kWh/t
            + 58.0 * (0.75 - scrap_q)                        # kirli/dusuk yogunluklu hurda
            + 3.1 * scrap_m                                  # nemli hurda: buharlastirma
            - 4.4 * (sp["oxygen_nm3_t"] - 25.0)              # kimyasal enerji elektrigi ikame eder
            - 3.0 * (sp["carbon_inject_kg_t"] - 8.0)
            + 0.35 * (sp["lime_kg_t"] - 42.0)                # curuf isitmasi
        )
        required = max(required, 210.0)

        # Power-on suresi ve zamana bagli isi kaybi (uzun heat = daha cok kayip)
        power_on = required * charge / (sp["active_power_mw"] * 1000.0 * 0.86) * 60.0
        loss = 1.85 * power_on
        eaf_elec = required + loss
        power_on = eaf_elec * charge / (sp["active_power_mw"] * 1000.0 * 0.86) * 60.0

        tap_to_tap = FIXED_OPS_MIN + power_on + 0.35 * sp["lf_heating_min"]
        prod_tph = charge / (tap_to_tap / 60.0)

        tap_temp = sp["tap_temp_target_c"] - 6.0 + 8.0 * (0.75 - scrap_q) + 5.5 * self.rng.standard_normal()
        liquidus = LIQUIDUS_C.get(self.product_code, 1515.0)
        superheat = (tap_temp - TRANSFER_LOSS_C
                     + LF_HEAT_RATE_C_PER_MIN * sp["lf_heating_min"] * 0.55
                     - 0.9 * sp["lf_heating_min"]          # bekleme sirasinda sogume
                     - liquidus)
        superheat = superheat + 2.0 * self.rng.standard_normal()

        lf_elec = LF_ELEC_KWH_T_PER_MIN * sp["lf_heating_min"]

        # Tav firini: bolge sicakliklari + hava fazlaligi (baca kaybi)
        thickness = float(self.profile.grade(self.product_code).attrs.get("slab_thickness_mm", 220))
        reheat_gas = (
            238.0
            + 0.20 * (sp["reheat_zone1_c"] - 880.0)
            + 0.28 * (sp["reheat_zone2_c"] - 1040.0)
            + 0.34 * (sp["reheat_zone3_c"] - 1130.0)
            + 620.0 * (sp["excess_air_ratio"] - 1.02)        # bacadan giden enerji
            + 0.22 * (thickness - 220.0)
            - 0.9 * (self.ambient_cache.get("ambient_temp_c", 15.0) - 15.0)
        )
        reheat_gas = max(reheat_gas, 150.0)

        return {
            "charge_weight_t": charge,
            "eaf_elec_kwh_t": eaf_elec,
            "reheat_gas_kwh_t": reheat_gas,
            "lf_elec_kwh_t": lf_elec,
            "sec_total_kwh_t": eaf_elec + reheat_gas + lf_elec,
            "tap_temp_c": tap_temp,
            "superheat_c": superheat,
            "tap_to_tap_min": tap_to_tap,
            "production_rate_tph": prod_tph,
            "power_on_min": power_on,
        }

    # ------------------------------------------------------------------ #
    ambient_cache: Dict[str, float] = {}

    def _downtime_probability(self) -> float:
        """Plansiz durus riski: yuksek superheat + hizli dokum = breakout."""
        d = self.derived
        logit = (
            -8.6
            + 0.055 * max(0.0, d["superheat_c"] - 40.0)
            + 3.2 * max(0.0, self.sp["caster_speed_mmin"] - 1.35)
            + 2.4 * max(0.0, 25.0 - d["superheat_c"]) * 0.10   # dusuk superheat: tikanma
            + 0.020 * max(0.0, self.sp["active_power_mw"] - 82.0) * 10.0
        )
        return sigmoid(logit)

    # ------------------------------------------------------------------ #
    def step(self) -> SimStep:
        self.ts = self.ts + timedelta(minutes=1)
        self.step_count += 1
        events: List[Dict[str, Any]] = []
        quality: List[Dict[str, Any]] = []
        closed_batch: Optional[Dict[str, Any]] = None

        crew = crew_of(self.ts, self.crews)
        self.ambient_cache = self.ambient(self.ts)
        self.scrap_quality.step()
        self.scrap_moisture.step()

        running = self.downtime_left <= 0
        if not running:
            self.downtime_left -= 1
        else:
            self.heat_minute += 1.0
            if self.rng.random() < self._downtime_probability():
                code, text = DOWNTIME_REASONS[int(self.rng.integers(0, len(DOWNTIME_REASONS)))]
                self.downtime_left = int(self.rng.integers(25, 180))
                events.append(self._event("downtime", "unplanned", code, text, self.downtime_left))

        # --- heat tamamlandi -> tap ---------------------------------------- #
        if running and self.heat_minute >= self.heat_length:
            closed_batch = self._close_heat()
            quality = closed_batch.pop("_quality", [])
            self._begin_heat()

        d = self.derived
        jitter = lambda scale: scale * self.rng.standard_normal()  # noqa: E731
        values: Dict[str, float] = {name: self.sp[name] for name in self.sp}
        values.update(
            {
                "scrap_quality_index": self.scrap_quality.value,
                "scrap_moisture_pct": self.scrap_moisture.value,
                "charge_weight_t": d["charge_weight_t"],
                "slab_thickness_mm": float(
                    self.profile.grade(self.product_code).attrs.get("slab_thickness_mm", 220)
                ),
                "tap_temp_c": d["tap_temp_c"] + jitter(1.5) if running else float("nan"),
                "tap_to_tap_min": d["tap_to_tap_min"],
                "superheat_c": d["superheat_c"] + jitter(0.8) if running else float("nan"),
                "eaf_elec_kwh_t": d["eaf_elec_kwh_t"] + jitter(3.0) if running else float("nan"),
                "reheat_gas_kwh_t": d["reheat_gas_kwh_t"] + jitter(2.0) if running else float("nan"),
                "production_rate_tph": d["production_rate_tph"] if running else 0.0,
                "sec_total_kwh_t": d["sec_total_kwh_t"] + jitter(4.0) if running else float("nan"),
            }
        )
        values.update(self.ambient_cache)
        if not running:
            values["active_power_mw"] = 0.0

        return SimStep(
            ts=self.ts,
            values=values,
            product_code=self.product_code,
            batch_id=self.batch_id,
            shift_id=shift_of(self.ts),
            crew=crew.code,
            running=running,
            events=events,
            quality=quality,
            closed_batch=closed_batch,
        )

    # ------------------------------------------------------------------ #
    def _event(self, etype: str, category: str, code: str, text: str, duration_min: int) -> Dict[str, Any]:
        return {
            "event_id": f"EV-{uuid.uuid4().hex[:10].upper()}",
            "ts_start": self.ts,
            "ts_end": self.ts + timedelta(minutes=duration_min),
            "line_id": self.profile.line_id,
            "asset_id": self.profile.line_id,
            "type": etype,
            "category": category,
            "reason_code": code,
            "reason_text": text,
            "duration_s": duration_min * 60.0,
        }

    def _close_heat(self) -> Dict[str, Any]:
        d = self.derived
        grade = self.profile.grade(self.product_code)
        tap_spec = grade.spec("tap_temp_c") or (1610.0, 1665.0)
        sh_spec = grade.attrs.get("superheat_spec", [15, 45])
        scrap = max(0.0, d["charge_weight_t"] * (0.055 + 0.02 * self.rng.random()))

        samples = [
            ("tap_temperature", d["tap_temp_c"], "C", tap_spec[0], tap_spec[1]),
            ("superheat", d["superheat_c"], "C", float(sh_spec[0]), float(sh_spec[1])),
            ("carbon_pct", clamp(0.16 + 0.02 * self.rng.standard_normal(), 0.02, 0.60), "%", 0.05, 0.25),
            ("sulphur_pct", clamp(0.022 + 0.004 * self.rng.standard_normal(), 0.001, 0.08), "%", None, 0.035),
        ]
        return {
            "batch_id": self.batch_id,
            "line_id": self.profile.line_id,
            "product_code": self.product_code,
            "start_ts": self.batch_start,
            "end_ts": self.ts,
            "produced_qty": round(d["charge_weight_t"] - scrap, 3),
            "scrap_qty": round(scrap, 3),
            "uom": "ton",
            "shift_id": shift_of(self.ts),
            "crew": crew_of(self.ts, self.crews).code,
            "_quality": [
                {
                    "sample_id": f"LAB-{uuid.uuid4().hex[:8].upper()}",
                    "batch_id": self.batch_id,
                    "ts": self.ts,
                    "line_id": self.profile.line_id,
                    "property": prop,
                    "value": round(float(value), 4),
                    "unit": unit,
                    "spec_min": lo,
                    "spec_max": hi,
                }
                for prop, value, unit, lo, hi in samples
            ],
        }
