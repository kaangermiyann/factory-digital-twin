"""Simulator altyapisi.

Amac: musteri verisi gelene kadar TUM pipeline'i (ingest -> feature -> model ->
optimizasyon -> dashboard) ayakta tutmak.

Tasarim kurali: simulator, canonical semayi uretir. Gercek veri geldiginde
`ingest` adaptoru ayni semayi uretecek; ust katmanlar farki gormeyecek.

Onemli: simulatorde BILINCLI olarak gercek bir optimizasyon boslugu birakilmistir.
  * Operatorler buhari "emniyet payi" ile fazla acar  -> bosa giden termal enerji
  * Ucuz mekanik su alma (nip/vakum) alisilmis degerde birakilir -> kullanilmayan kapasite
  * Vardiya ekipleri farkli bias tasir                -> ayni urunde farkli enerji
Model bu iliskiyi VERIDEN ogrenir; denklemleri modele soylemiyoruz.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from twin.config import Profile

MINUTE = timedelta(minutes=1)


# --------------------------------------------------------------------------- #
# Yardimcilar
# --------------------------------------------------------------------------- #
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class OrnsteinUhlenbeck:
    """Ortalamaya donen rastgele yuruyus.

    Beyaz gurultuden farki: gercek proses degiskenleri gibi 'yapiskan'dir --
    bir yone sapar, sonra yavasca geri doner. Lag/rolling ozelliklerinin
    anlamli olmasi bunu gerektirir.
    """

    def __init__(self, rng: np.random.Generator, mean: float, sigma: float, theta: float = 0.02,
                 low: Optional[float] = None, high: Optional[float] = None) -> None:
        self.rng = rng
        self.mean = mean
        self.sigma = sigma
        self.theta = theta
        self.low = low
        self.high = high
        self.value = mean

    def step(self, mean: Optional[float] = None) -> float:
        target = self.mean if mean is None else mean
        self.value += self.theta * (target - self.value) + self.sigma * self.rng.standard_normal()
        if self.low is not None or self.high is not None:
            self.value = clamp(self.value, self.low if self.low is not None else -1e9,
                               self.high if self.high is not None else 1e9)
        return self.value


@dataclass
class Crew:
    """Vardiya ekibi. Enerji tuketimindeki aciklanabilir varyansin buyuk kismi budur."""

    code: str
    steam_bias: float          # buhari ne kadar fazla acar (emniyet payi)
    speed_bias: float          # hiz istahi
    nip_habit: float           # mekanik su almayi ne kadar kullanir
    vacuum_habit: float
    reaction_gain: float       # nem sapmasina ne kadar hizli tepki verir
    discipline: float          # 0-1, setpoint kararliligi


DEFAULT_CREWS = [
    Crew("A", steam_bias=+0.34, speed_bias=-6.0, nip_habit=-18.0, vacuum_habit=+3.0,
         reaction_gain=0.55, discipline=0.80),
    Crew("B", steam_bias=+0.08, speed_bias=+11.0, nip_habit=+22.0, vacuum_habit=-4.0,
         reaction_gain=0.85, discipline=0.92),
    Crew("C", steam_bias=+0.52, speed_bias=+2.0, nip_habit=-30.0, vacuum_habit=+6.0,
         reaction_gain=0.35, discipline=0.62),
]


def shift_of(ts: datetime) -> int:
    """3 vardiya: 1 = 00-08, 2 = 08-16, 3 = 16-24."""
    return ts.hour // 8 + 1


def crew_of(ts: datetime, crews: List[Crew]) -> Crew:
    """Ekipler gunluk donusumlu vardiya alir."""
    day_index = (ts - datetime(2020, 1, 1, tzinfo=timezone.utc)).days
    return crews[(shift_of(ts) - 1 + day_index) % len(crews)]


# --------------------------------------------------------------------------- #
@dataclass
class SimStep:
    """Bir zaman adiminin tam ciktisi."""

    ts: datetime
    values: Dict[str, float]                # canonical degisken -> deger
    product_code: str
    batch_id: str
    shift_id: int
    crew: str
    running: bool = True
    events: List[Dict[str, Any]] = field(default_factory=list)
    quality: List[Dict[str, Any]] = field(default_factory=list)
    closed_batch: Optional[Dict[str, Any]] = None


class ProcessSimulator(abc.ABC):
    """Proses simulatorleri icin ortak arayuz."""

    def __init__(self, profile: Profile, seed: int = 42, start: Optional[datetime] = None) -> None:
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        self.crews = DEFAULT_CREWS
        self.ts = start or datetime.now(timezone.utc).replace(second=0, microsecond=0)
        self.step_count = 0

    # -- ortam (her iki proses icin ortak) ---------------------------------- #
    def ambient(self, ts: datetime) -> Dict[str, float]:
        """Mevsimsel + gunluk sicaklik/nem. Kurutma enerjisini dogrudan etkiler."""
        doy = ts.timetuple().tm_yday
        hour = ts.hour + ts.minute / 60.0
        seasonal = 11.0 * math.sin(2 * math.pi * (doy - 105) / 365.0)
        daily = 5.5 * math.sin(2 * math.pi * (hour - 9) / 24.0)
        temp = 15.0 + seasonal + daily + 1.2 * self.rng.standard_normal()
        # Nem sicaklikla ters, kisin daha yuksek
        rh = clamp(72.0 - 0.9 * (temp - 15.0) + 6.0 * math.sin(2 * math.pi * (doy - 20) / 365.0)
                   + 3.0 * self.rng.standard_normal(), 20.0, 98.0)
        return {"ambient_temp_c": round(temp, 2), "ambient_humidity_pct": round(rh, 2)}

    def pick_grade(self, exclude: Optional[str] = None) -> str:
        """Urun payina gore grade sec.

        `exclude`: surekli proseslerde kampanya degisimi tanimi geregi BASKA bir
        urune gecmektir; ayni grade'i yeniden secmek kampanya degisimi degildir.
        """
        codes = [c for c in self.profile.grades if c != exclude] or list(self.profile.grades)
        weights = np.array([self.profile.grades[c].share for c in codes], dtype=float)
        weights = weights / weights.sum()
        return str(self.rng.choice(codes, p=weights))

    @abc.abstractmethod
    def step(self) -> SimStep:
        """Bir zaman adimi ilerlet ve ciktiyi dondur."""

    def run(self, minutes: int) -> List[SimStep]:
        return [self.step() for _ in range(minutes)]
