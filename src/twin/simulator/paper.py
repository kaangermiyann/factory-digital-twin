"""Kagit makinesi (PM2) simulatoru -- fizik temelli, 1 dakikalik adim.

Modellenen zincir:

    hamur -> elek (vakum) -> pres (nip) -> kurutma (buhar) -> sarici
             ucuz mekanik su alma        pahali termal su alma

Optimizasyonun kaynagi tam olarak burasi: preste alinan her kg su, kurutmada
alinmayan kg sudur. Mekanik su alma termalin ~1/10'u maliyetindedir. Operator
aliskanliktan nip/vakumu sabit tutup buharla telafi eder -> bosa giden para.

Denklemler ML modeline VERILMEZ; model bunlari veriden ogrenmek zorundadir.
"""

from __future__ import annotations

import math
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np

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

# --- fiziksel sabitler (kalibre edilmis) ------------------------------------ #
EVAP_ENERGY_KWH_PER_KG = 0.75      # suyun buharlastirilmasi (kayiplar dahil)
DRYER_EFFICIENCY = 0.88            # silindir/kondens verimi
DRYER_CAPACITY_K = 15800.0         # buhar -> buharlastirma kapasitesi olcegi
OVERSTEAM_WASTE = 0.45             # kullanilmayan kurutma kapasitesinin bosa giden orani
REEL_TONS = 25.0                   # bir tambur (reel) agirligi
FELT_LIFE_DAYS = 21.0              # keçe omru -> pres verimi zamanla duser

GRADE_BASE = {
    #            hiz   refiner  retansiyon
    "TL80":  (850.0,  95.0, 0.60),
    "TL120": (610.0,  88.0, 0.55),
    "FL45":  (1060.0, 105.0, 0.70),
}

BREAK_REASONS = [
    ("BRK-01", "Kagit kopusu - elek/pres bolgesi"),
    ("BRK-02", "Kagit kopusu - kurutma bolgesi"),
    ("BRK-03", "Sarici problemi"),
]


class PaperMachineSimulator(ProcessSimulator):
    def __init__(self, profile: Profile, seed: int = 42, start=None) -> None:
        super().__init__(profile, seed=seed, start=start)
        self.width = float(profile.extras.get("wire_width_m", 6.4))

        # --- hammadde / ortam sureclerileri (yavas, yapiskan) --------------- #
        self.freeness = OrnsteinUhlenbeck(self.rng, mean=380.0, sigma=2.2, theta=0.010, low=300, high=460)
        self.recycled = OrnsteinUhlenbeck(self.rng, mean=68.0, sigma=0.6, theta=0.008, low=40, high=95)
        self.order_pressure = OrnsteinUhlenbeck(self.rng, mean=0.0, sigma=0.9, theta=0.006, low=-18, high=18)

        # --- GOZLENMEYEN (latent) proses durumu ---------------------------- #
        # Buhar kapani kacaklari, silindir yuzey durumu, kondens tahliyesi,
        # gercek elyaf karisimi... Hicbir fabrika bunlari olcmez.
        # Modelin aciklayamayacagi artik varyans budur ve OLMASI gerekir:
        # yoksa sentetik veride model gercekte tutturulamayacak kadar iyi cikar.
        # Musteriye soylenecek cumle de buradan dogar: "bu degiskeni olcerseniz
        # modelin hatasi su kadar duser."
        self.thermal_condition = OrnsteinUhlenbeck(self.rng, mean=1.0, sigma=0.0016, theta=0.004,
                                                   low=0.93, high=1.07)
        self.elec_condition = OrnsteinUhlenbeck(self.rng, mean=1.0, sigma=0.0012, theta=0.005,
                                                low=0.94, high=1.06)

        # --- durum --------------------------------------------------------- #
        self.product_code = self.pick_grade()
        self.campaign_left = int(self.rng.integers(6 * 60, 26 * 60))
        self.transition_left = 0
        self.downtime_left = 0
        self.downtime_reason: Optional[tuple] = None
        self.felt_age_min = float(self.rng.integers(0, int(FELT_LIFE_DAYS * 1440)))

        self.batch_id = self._new_batch_id()
        self.batch_start = self.ts
        self.batch_tons = 0.0
        self.batch_scrap = 0.0
        self.batch_moisture: List[float] = []
        self.batch_bw: List[float] = []
        self.batch_tensile: List[float] = []

        # --- setpointler (operatorun elindeki kollar) ---------------------- #
        crew = crew_of(self.ts, self.crews)
        base_speed, base_ref, base_ret = GRADE_BASE[self.product_code]
        self.sp: Dict[str, float] = {
            "machine_speed_mpm": base_speed + 2.2 * crew.speed_bias,
            "headbox_consistency_pct": 0.90,
            "refiner_sec_kwh_t": base_ref,
            "press_nip_load_knm": 320.0 + crew.nip_habit,
            "vacuum_kpa": -50.0 + crew.vacuum_habit,
            "steam_g1_bar": 2.6,
            "steam_g2_bar": 3.4,
            "steam_g3_bar": 4.0,
            "hood_supply_temp_c": 105.0,
            "retention_aid_kg_t": base_ret,
        }
        # Prosesin gordugu deger != setpoint. Birinci mertebe gecikme (olu zaman).
        self.eff: Dict[str, float] = dict(self.sp)
        self.last_moisture = 5.5
        self.speed_review_in = 0

    # ------------------------------------------------------------------ #
    def _new_batch_id(self) -> str:
        return f"REEL-{self.ts:%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"

    def _grade_spec(self) -> tuple:
        return self.profile.grade(self.product_code).spec("reel_moisture_pct") or (4.8, 6.2)

    def _basis_weight(self) -> float:
        return float(self.profile.grade(self.product_code).attrs["basis_weight_gsm"])

    # ------------------------------------------------------------------ #
    # Operator davranisi -- optimizasyon bosluğunun kaynagi
    # ------------------------------------------------------------------ #
    def _operate(self, crew) -> None:
        base_speed, base_ref, base_ret = GRADE_BASE[self.product_code]
        spec_lo, spec_hi = self._grade_spec()

        # Operator spec ortasini degil, KURU tarafi hedefler ("spec disina cikmayayim").
        # Bu emniyet payi, fazladan buhar demektir -- ve tam olarak geri kazanilacak paradir.
        aim = spec_lo + 0.32 * (spec_hi - spec_lo)

        # 1) Buhar: neme gore PI benzeri tepki (olculen nem gecikmelidir)
        error = self.last_moisture - aim
        gain = crew.reaction_gain * 0.22
        adjust = gain * error + crew.steam_bias * 0.045
        for key, share in (("steam_g1_bar", 0.20), ("steam_g2_bar", 0.35), ("steam_g3_bar", 0.45)):
            var = self.profile.var(key)
            noise = (1.0 - crew.discipline) * 0.03 * self.rng.standard_normal()
            self.sp[key] = clamp(self.sp[key] + adjust * share + noise, var.op_low, var.op_high)

        # 2) Hiz: nadiren gozden gecirilir (yaklasik yarim saatte bir)
        self.speed_review_in -= 1
        if self.speed_review_in <= 0:
            self.speed_review_in = int(self.rng.integers(20, 45))
            var = self.profile.var("machine_speed_mpm")
            wanted = base_speed + 2.2 * crew.speed_bias + 1.4 * self.order_pressure.value
            self.sp["machine_speed_mpm"] = clamp(
                0.75 * self.sp["machine_speed_mpm"] + 0.25 * wanted
                + (1.0 - crew.discipline) * 2.5 * self.rng.standard_normal(),
                var.op_low, var.op_high,
            )

        # 3) Nip / vakum: ALISKANLIK. Neredeyse hic degismez.
        #    Ucuz mekanik su alma kapasitesi burada atil kalir -> ana tasarruf kalemi.
        for key, habit in (("press_nip_load_knm", 320.0 + crew.nip_habit),
                           ("vacuum_kpa", -50.0 + crew.vacuum_habit)):
            var = self.profile.var(key)
            self.sp[key] = clamp(0.995 * self.sp[key] + 0.005 * habit
                                 + 0.25 * self.rng.standard_normal() * (1.0 - crew.discipline),
                                 var.op_low, var.op_high)

        # 4) Refiner: freeness'e gore ayarlanir (dayanim icin), ekip biasi ile
        var = self.profile.var("refiner_sec_kwh_t")
        wanted = base_ref + 0.16 * (self.freeness.value - 380.0) + 4.0 * crew.steam_bias
        self.sp["refiner_sec_kwh_t"] = clamp(0.97 * self.sp["refiner_sec_kwh_t"] + 0.03 * wanted,
                                             var.op_low, var.op_high)

        # 5) Hood ve kimyasal: kabaca sabit
        var = self.profile.var("hood_supply_temp_c")
        self.sp["hood_supply_temp_c"] = clamp(
            0.98 * self.sp["hood_supply_temp_c"] + 0.02 * (102.0 + 12.0 * crew.steam_bias)
            + 0.3 * self.rng.standard_normal(), var.op_low, var.op_high)
        var = self.profile.var("retention_aid_kg_t")
        self.sp["retention_aid_kg_t"] = clamp(
            0.98 * self.sp["retention_aid_kg_t"] + 0.02 * base_ret + 0.01 * self.rng.standard_normal(),
            var.op_low, var.op_high)

        var = self.profile.var("headbox_consistency_pct")
        self.sp["headbox_consistency_pct"] = clamp(
            0.99 * self.sp["headbox_consistency_pct"] + 0.01 * (0.86 + 0.0009 * self._basis_weight()),
            var.op_low, var.op_high)

        # Setpoint -> proses: birinci mertebe gecikme (olu zaman + atalet)
        tau = {"steam_g1_bar": 6.0, "steam_g2_bar": 8.0, "steam_g3_bar": 10.0,
               "machine_speed_mpm": 3.0, "press_nip_load_knm": 2.0, "vacuum_kpa": 2.0,
               "refiner_sec_kwh_t": 12.0, "hood_supply_temp_c": 9.0,
               "retention_aid_kg_t": 5.0, "headbox_consistency_pct": 4.0}
        for key, value in self.sp.items():
            self.eff[key] += (value - self.eff[key]) / tau.get(key, 5.0)

    def _apply_grade_recipe(self, crew) -> None:
        """Grade degisiminde recete setpointleri ANINDA uygulanir.

        Gercek operator hizi kademe kademe aramaz; yeni urunun recetesindeki
        hiza gecer. Bunu modellemezsek makine saatlerce yanlis hizda calisir ve
        veri fiziksel olarak anlamsiz hale gelir.
        """
        base_speed, base_ref, base_ret = GRADE_BASE[self.product_code]
        self.sp["machine_speed_mpm"] = clamp(
            base_speed + 2.2 * crew.speed_bias,
            self.profile.var("machine_speed_mpm").op_low,
            self.profile.var("machine_speed_mpm").op_high,
        )
        self.sp["refiner_sec_kwh_t"] = base_ref
        self.sp["retention_aid_kg_t"] = base_ret
        self.speed_review_in = int(self.rng.integers(20, 45))

    # ------------------------------------------------------------------ #
    # Fizik
    # ------------------------------------------------------------------ #
    def _physics(self, ambient: Dict[str, float], running: bool) -> Dict[str, float]:
        e = self.eff
        bw = self._basis_weight()
        speed = e["machine_speed_mpm"] if running else 0.0

        # -- uretim hizi (t/h) --------------------------------------------- #
        prod_tph = speed * self.width * bw * 60.0 / 1e6

        # -- pres cikisi kuruluk (%) --------------------------------------- #
        # Nip yuku ve vakum ARTIRIR, hiz ve gramaj AZALTIR. Keçe eskidikce duser.
        felt_wear = 1.6 * (self.felt_age_min / (FELT_LIFE_DAYS * 1440.0))
        dryness = (
            40.0
            + 0.034 * (e["press_nip_load_knm"] - 250.0)
            + 0.125 * (-e["vacuum_kpa"] - 35.0)
            - 0.0072 * (e["machine_speed_mpm"] - 700.0)
            - 0.021 * (bw - 80.0)
            + 0.010 * (self.freeness.value - 380.0)      # serbest hamur daha kolay su verir
            - 0.014 * (self.recycled.value - 68.0)
            - felt_wear
            + 0.32 * self.rng.standard_normal()
        )
        dryness = clamp(dryness, 34.0, 54.0)

        # -- kurutma denklemi ---------------------------------------------- #
        dry_tph = prod_tph * 0.945                            # lif akisi (yaklasik)
        water_in_kg_h = dry_tph * 1000.0 * (100.0 / dryness - 1.0)

        # Buhar basinci -> buharlastirma kapasitesi (doyma sicakligi ~ P^0.5)
        steam_term = (
            0.25 * math.sqrt(e["steam_g1_bar"] + 1.0)
            + 0.35 * math.sqrt(e["steam_g2_bar"] + 1.0)
            + 0.40 * math.sqrt(e["steam_g3_bar"] + 1.0)
        )
        hood_factor = 1.0 + 0.0045 * (e["hood_supply_temp_c"] - 105.0)
        # Ortam nemi yuksekse hava daha az su tasir -> kapasite duser (MEVSIMSELLIK)
        ambient_factor = 1.0 - 0.0022 * (ambient["ambient_humidity_pct"] - 55.0) \
                             + 0.0016 * (ambient["ambient_temp_c"] - 18.0)
        evap_capacity = DRYER_CAPACITY_K * steam_term * hood_factor * ambient_factor

        if running and prod_tph > 0:
            floor_water = dry_tph * 1000.0 * 0.030 / 0.970     # fiziksel alt sinir (~%3 nem)
            evaporated = clamp(evap_capacity, 0.0, max(water_in_kg_h - floor_water, 0.0))
            water_out = water_in_kg_h - evaporated
            moisture = 100.0 * water_out / (dry_tph * 1000.0 + water_out)
            moisture += 0.16 * self.rng.standard_normal()
            if self.transition_left > 0:                        # grade gecisi: kontrol bozulur
                moisture += 0.55 * self.rng.standard_normal() + 0.25
            unused_capacity = max(evap_capacity - evaporated, 0.0)
        else:
            evaporated = 0.0
            moisture = float("nan")
            unused_capacity = evap_capacity                     # duruşta buhar bosa gider

        # -- enerji --------------------------------------------------------- #
        # Termal: gercek buharlastirma + kullanilmayan kapasitenin bir kismi (kayip)
        thermal_kwh_h = (evaporated + OVERSTEAM_WASTE * unused_capacity) \
            * EVAP_ENERGY_KWH_PER_KG / (DRYER_EFFICIENCY * self.thermal_condition.value)

        # Elektrik: tahrik + vakum pompalari + refiner + yardimci
        drives_kw = 380.0 + 0.62 * max(speed - 500.0, 0.0) + 0.55 * (e["press_nip_load_knm"] - 250.0)
        vacuum_kw = 220.0 + 6.6 * (-e["vacuum_kpa"] - 35.0)
        refiner_kw = e["refiner_sec_kwh_t"] * dry_tph
        aux_kw = 4200.0 + 1.6 * (e["hood_supply_temp_c"] - 80.0)
        elec_kw = ((drives_kw + vacuum_kw + refiner_kw) * (1.0 if running else 0.15)
                   + aux_kw * (1.0 if running else 0.55)) / self.elec_condition.value

        if prod_tph > 0.1:
            # Sayac/olcum gurultusu. Gercek historian verisinde bu her zaman vardir;
            # eklenmezse model neredeyse mukemmel gorunur (R2 > 0.99) ve musteriye
            # tutulamayacak bir beklenti verilir.
            elec_kwh_t = elec_kw / prod_tph * (1.0 + 0.012 * self.rng.standard_normal())
            thermal_kwh_t = thermal_kwh_h / prod_tph * (1.0 + 0.015 * self.rng.standard_normal())
        else:
            elec_kwh_t = float("nan")
            thermal_kwh_t = float("nan")

        # -- dayanim (lab ölçümü icin) -------------------------------------- #
        tensile = (
            28.0
            + 0.175 * e["refiner_sec_kwh_t"]
            - 0.045 * (self.freeness.value - 380.0)
            - 0.055 * (self.recycled.value - 68.0)
            + 3.2 * (e["retention_aid_kg_t"] - 0.6)
            + 0.020 * (bw - 80.0)
            - 0.0052 * (e["machine_speed_mpm"] - 820.0)
            + 0.55 * self.rng.standard_normal()
        )

        out = {
            "machine_speed_mpm": e["machine_speed_mpm"] if running else 0.0,
            "headbox_consistency_pct": e["headbox_consistency_pct"],
            "refiner_sec_kwh_t": e["refiner_sec_kwh_t"],
            "press_nip_load_knm": e["press_nip_load_knm"],
            "vacuum_kpa": e["vacuum_kpa"],
            "steam_g1_bar": e["steam_g1_bar"],
            "steam_g2_bar": e["steam_g2_bar"],
            "steam_g3_bar": e["steam_g3_bar"],
            "hood_supply_temp_c": e["hood_supply_temp_c"],
            "retention_aid_kg_t": e["retention_aid_kg_t"],
            "pulp_freeness_ml": self.freeness.value,
            "recycled_ratio_pct": self.recycled.value,
            "basis_weight_target_gsm": bw,
            "press_dryness_pct": dryness,
            "reel_moisture_pct": moisture,
            "reel_basis_weight_gsm": bw + 0.6 * self.rng.standard_normal() if running else float("nan"),
            "production_rate_tph": prod_tph * (1.0 + 0.004 * self.rng.standard_normal()) if running else 0.0,
            "elec_kwh_t": elec_kwh_t,
            "thermal_kwh_t": thermal_kwh_t,
            "sec_total_kwh_t": elec_kwh_t + thermal_kwh_t if running else float("nan"),
        }
        out.update(ambient)
        out["_tensile"] = tensile
        out["_evaporated_kg_h"] = evaporated
        out["_wasted_kwh_h"] = OVERSTEAM_WASTE * unused_capacity * EVAP_ENERGY_KWH_PER_KG / DRYER_EFFICIENCY
        return out

    # ------------------------------------------------------------------ #
    def _break_probability(self, values: Dict[str, float]) -> float:
        """Dakikalik kopus olasiligi.

        Yuksek hiz, asiri kuru sac (kirilgan), dusuk dayanim ve grade gecisi riski artirir.
        Bu, "hizi sonuna kadar ac" cozumunu engelleyen dogal kisittir.
        """
        moisture = values["reel_moisture_pct"]
        if math.isnan(moisture):
            return 0.0
        logit = (
            -6.9
            + 0.0075 * (values["machine_speed_mpm"] - 780.0)
            + 0.55 * max(0.0, 4.5 - moisture)                 # asiri kurutma -> kirilgan
            + 0.060 * max(0.0, 44.0 - values["_tensile"])     # zayif sac
            + 0.030 * max(0.0, values["recycled_ratio_pct"] - 75.0)
            + (0.9 if self.transition_left > 0 else 0.0)
            + 0.55 * (self.felt_age_min / (FELT_LIFE_DAYS * 1440.0))
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
        ambient = self.ambient(self.ts)
        self.freeness.step()
        self.recycled.step()
        self.order_pressure.step()
        self.thermal_condition.step()
        self.elec_condition.step()
        self.felt_age_min += 1
        if self.transition_left > 0:
            self.transition_left -= 1

        # --- planli bakim: keçe degisimi ----------------------------------- #
        if self.felt_age_min >= FELT_LIFE_DAYS * 1440 and self.downtime_left <= 0:
            self.downtime_left = int(self.rng.integers(180, 300))
            self.downtime_reason = ("MNT-01", "Planli keçe degisimi")
            self.felt_age_min = 0.0
            events.append(self._event("downtime", "planned", *self.downtime_reason, self.downtime_left))

        running = self.downtime_left <= 0
        if not running:
            self.downtime_left -= 1

        # --- grade kampanyasi ---------------------------------------------- #
        self.campaign_left -= 1
        if self.campaign_left <= 0 and running:
            old = self.product_code
            self.product_code = self.pick_grade(exclude=old)
            self.campaign_left = int(self.rng.integers(6 * 60, 26 * 60))
            if self.product_code != old:
                self.transition_left = int(self.rng.integers(20, 45))
                self._apply_grade_recipe(crew)
                events.append(self._event("grade_change", "planned", "GRD-01",
                                          f"Grade degisimi {old} -> {self.product_code}",
                                          self.transition_left))
                closed_batch = self._close_batch()

        if running:
            self._operate(crew)
        values = self._physics(ambient, running)

        if running and not math.isnan(values["reel_moisture_pct"]):
            self.last_moisture = values["reel_moisture_pct"]

        # --- kopus riski ---------------------------------------------------- #
        if running and self.rng.random() < self._break_probability(values):
            code, text = BREAK_REASONS[int(self.rng.integers(0, len(BREAK_REASONS)))]
            self.downtime_left = int(self.rng.integers(12, 55))
            self.downtime_reason = (code, text)
            events.append(self._event("downtime", "unplanned", code, text, self.downtime_left))
            self.batch_scrap += values["production_rate_tph"] * (self.downtime_left / 60.0) * 0.12

        # --- reel birikimi --------------------------------------------------- #
        if running:
            self.batch_tons += values["production_rate_tph"] / 60.0
            if not math.isnan(values["reel_moisture_pct"]):
                self.batch_moisture.append(values["reel_moisture_pct"])
                self.batch_bw.append(values["reel_basis_weight_gsm"])
                self.batch_tensile.append(values["_tensile"])
            if self.batch_tons >= REEL_TONS:
                closed_batch = self._close_batch()

        if closed_batch is not None:
            quality = closed_batch.pop("_quality", [])

        values = {k: v for k, v in values.items() if not k.startswith("_")}

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

    def _close_batch(self) -> Optional[Dict[str, Any]]:
        if self.batch_tons <= 0.01 or not self.batch_moisture:
            return None
        grade = self.profile.grade(self.product_code)
        spec_lo, spec_hi = self._grade_spec()
        samples = [
            ("moisture", float(np.mean(self.batch_moisture)), "%", spec_lo, spec_hi),
            ("basis_weight", float(np.mean(self.batch_bw)), "g/m2",
             grade.attrs["basis_weight_gsm"] * 0.97, grade.attrs["basis_weight_gsm"] * 1.03),
            ("tensile_index", float(np.mean(self.batch_tensile)), "Nm/g",
             float(grade.attrs.get("tensile_spec_min", 40)), None),
        ]
        batch = {
            "batch_id": self.batch_id,
            "line_id": self.profile.line_id,
            "product_code": self.product_code,
            "start_ts": self.batch_start,
            "end_ts": self.ts,
            "produced_qty": round(self.batch_tons, 3),
            "scrap_qty": round(self.batch_scrap, 3),
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
                    "value": round(value, 3),
                    "unit": unit,
                    "spec_min": lo,
                    "spec_max": hi,
                }
                for prop, value, unit, lo, hi in samples
            ],
        }
        self.batch_id = self._new_batch_id()
        self.batch_start = self.ts
        self.batch_tons = 0.0
        self.batch_scrap = 0.0
        self.batch_moisture, self.batch_bw, self.batch_tensile = [], [], []
        return batch
