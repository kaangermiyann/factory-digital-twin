"""Amac fonksiyonu ve kisitlar.

Tahmin bir rapordur, optimizasyon bir karardir. Bu dosya kararin verildigi yer.

Cikti birimi bilerek **TL/ton**'dur. Yonetim kWh'a degil paraya bakar; ayrica
elektrik ve termal enerji farkli fiyatlandigi icin kWh toplami zaten yaniltici
bir metriktir (1 kWh buhar ile 1 kWh elektrik ayni sey degildir).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from twin.config import Grade, Profile, get_settings

# Herhangi bir kisit ihlalinde eklenen sabit ceza (penalty_weight ile carpilir).
# Fizibil olmayan hicbir tavsiye, tasarrufu ne kadar buyuk olursa olsun kazanamaz.
INFEASIBLE_OFFSET = 0.5


@dataclass
class Economics:
    """Enerji ve urun fiyatlari. Saatlik tarife destekli."""

    electricity_try_per_kwh: float = 3.10
    thermal_try_per_kwh: float = 1.35
    peak_hours: Tuple[int, ...] = (17, 18, 19, 20, 21)
    peak_multiplier: float = 1.85
    offpeak_hours: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    offpeak_multiplier: float = 0.55
    scrap_cost_try_per_ton: float = 9800.0
    product_margin_try_per_ton: float = 2400.0

    @classmethod
    def from_settings(cls, settings: Optional[Any] = None) -> "Economics":
        cfg = (settings or get_settings()).get_path("prices", {}) or {}
        return cls(
            electricity_try_per_kwh=float(cfg.get("electricity_try_per_kwh", 3.10)),
            thermal_try_per_kwh=float(cfg.get("thermal_try_per_kwh", 1.35)),
            peak_hours=tuple(cfg.get("peak_hours", [17, 18, 19, 20, 21])),
            peak_multiplier=float(cfg.get("peak_multiplier", 1.85)),
            offpeak_hours=tuple(cfg.get("offpeak_hours", [0, 1, 2, 3, 4, 5])),
            offpeak_multiplier=float(cfg.get("offpeak_multiplier", 0.55)),
            scrap_cost_try_per_ton=float(cfg.get("scrap_cost_try_per_ton", 9800.0)),
            product_margin_try_per_ton=float(cfg.get("product_margin_try_per_ton", 2400.0)),
        )

    def electricity_price(self, hour: int) -> float:
        """Saatlik tarife. Optimizasyon bunu gorunce puant saatlerde enerji-yogun
        rejimden kacmayi KENDILIGINDEN ogrenir -- ek kod yazmadan."""
        if hour in self.peak_hours:
            return self.electricity_try_per_kwh * self.peak_multiplier
        if hour in self.offpeak_hours:
            return self.electricity_try_per_kwh * self.offpeak_multiplier
        return self.electricity_try_per_kwh

    def energy_cost_try_per_ton(self, elec_kwh_t: float, thermal_kwh_t: float, hour: int) -> float:
        return elec_kwh_t * self.electricity_price(hour) + thermal_kwh_t * self.thermal_try_per_kwh


# --------------------------------------------------------------------------- #
@dataclass
class Constraint:
    """Tek bir kisit. Ihlal, ceza terimine donusur."""

    name: str
    target: str
    low: Optional[float] = None
    high: Optional[float] = None
    weight: float = 1.0
    unit: str = ""

    def violation(self, value: float) -> float:
        """Normalize edilmis ihlal buyuklugu (0 = saglaniyor)."""
        span = max((self.high or 0.0) - (self.low or 0.0), 1e-6) if (self.low is not None and self.high is not None) else 1.0
        if self.low is not None and value < self.low:
            return (self.low - value) / span
        if self.high is not None and value > self.high:
            return (value - self.high) / span
        return 0.0

    def describe(self, value: float) -> str:
        bounds = []
        if self.low is not None:
            bounds.append(f">= {self.low:g}")
        if self.high is not None:
            bounds.append(f"<= {self.high:g}")
        status = "OK" if self.violation(value) == 0 else "IHLAL"
        return f"{self.name}: {value:.2f} {self.unit} ({' ve '.join(bounds)}) {status}"


def build_constraints(profile: Profile, grade: Optional[Grade]) -> List[Constraint]:
    """Profil + grade spec'lerinden kisit listesi uretir."""
    cons: List[Constraint] = []
    limits = profile.constraints or {}

    for target, spec in profile.targets.items():
        if spec.constraint != "spec" or grade is None:
            continue
        window = grade.spec(target)
        if window:
            lo, hi = window
            cons.append(Constraint(name=f"{target} spec", target=target, low=lo, high=hi,
                                   unit=profile.variables[target].unit if target in profile.variables else ""))

    if "min_production_rate_tph" in limits and "production_rate_tph" in profile.targets:
        cons.append(Constraint(name="min uretim", target="production_rate_tph",
                               low=float(limits["min_production_rate_tph"]), unit="t/h"))

    for target, spec in profile.targets.items():
        if spec.kind == "classification" and spec.risk_ceiling is not None:
            ceiling = float(limits.get("break_risk_ceiling", spec.risk_ceiling))
            cons.append(Constraint(name=f"{target} tavani", target=target, high=ceiling, unit=""))
    return cons


# --------------------------------------------------------------------------- #
@dataclass
class ObjectiveResult:
    cost_try_per_ton: float
    energy_cost: float
    penalty: float
    predictions: Dict[str, float]
    violations: List[str] = field(default_factory=list)


class Objective:
    """Egitilmis modelleri kullanarak bir setpoint vektorunu fiyatlandirir.

    Modeller `bundles` sozlugunde gelir. Enerji ayristirilmis (elec/thermal)
    tahmin edilemiyorsa toplam SEC uzerinden ortalama fiyat kullanilir.
    """

    def __init__(
        self,
        bundles: Dict[str, Any],
        profile: Profile,
        grade: Optional[Grade],
        economics: Optional[Economics] = None,
        penalty_weight: float = 10000.0,
    ) -> None:
        self.bundles = bundles
        self.profile = profile
        self.grade = grade
        self.economics = economics or Economics.from_settings()
        self.penalty_weight = penalty_weight
        self.constraints = build_constraints(profile, grade)

    # ------------------------------------------------------------------ #
    def predict_all(self, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {name: bundle.predict(frame) for name, bundle in self.bundles.items()}

    def evaluate(self, frame: pd.DataFrame, hour: int) -> List[ObjectiveResult]:
        """Toplu degerlendirme -- optimizasyon dongusu tek cagriyla yuzlerce
        aday puanlar; satir satir predict cagirmak 100x yavas olurdu."""
        preds = self.predict_all(frame)
        results: List[ObjectiveResult] = []

        margin = self.grade.margin_try_per_ton if self.grade else self.economics.product_margin_try_per_ton
        reference_rate = float(self.profile.constraints.get("min_production_rate_tph", 1.0)) or 1.0

        for i in range(len(frame)):
            row = {name: float(values[i]) for name, values in preds.items()}
            sec = row.get("sec_total_kwh_t", 0.0)

            if "elec_kwh_t" in row and "thermal_kwh_t" in row:
                energy_cost = self.economics.energy_cost_try_per_ton(
                    row["elec_kwh_t"], row["thermal_kwh_t"], hour
                )
            else:
                # Ayristirma yoksa: kagit/celikte termal pay ~%75, agirlikli fiyat
                blended = 0.25 * self.economics.electricity_price(hour) + 0.75 * self.economics.thermal_try_per_kwh
                energy_cost = sec * blended

            penalty = 0.0
            violations: List[str] = []
            for constraint in self.constraints:
                if constraint.target not in row:
                    continue
                magnitude = constraint.violation(row[constraint.target])
                if magnitude > 0:
                    # Sabit terim + buyume terimi. Sabit terim sart: sadece
                    # kareli ceza kullanilirsa "spec'i 0.02 asayim, 19 TL
                    # kazanayim" gibi cozumler kabul edilebilir gorunur ve
                    # optimizasyon SPEC DISI tavsiye uretir. Sabit terim bunu
                    # imkansiz kilar; kareli terim ise cozucuyu fizibil bolgeye
                    # dogru yumusakca iter.
                    penalty += self.penalty_weight * constraint.weight * (
                        INFEASIBLE_OFFSET + magnitude ** 2
                    )
                    violations.append(constraint.describe(row[constraint.target]))

            # Uretim hizi: kapasitenin uzerindeki her ton ek marj getirir.
            # (Isaret negatif: maliyeti dusurur.)
            rate = row.get("production_rate_tph")
            throughput_value = 0.0
            if rate is not None:
                throughput_value = margin * 0.02 * (rate - reference_rate) / max(reference_rate, 1e-6)

            results.append(
                ObjectiveResult(
                    cost_try_per_ton=energy_cost + penalty - throughput_value,
                    energy_cost=energy_cost,
                    penalty=penalty,
                    predictions=row,
                    violations=violations,
                )
            )
        return results
