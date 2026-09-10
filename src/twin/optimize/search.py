"""Kisitli optimizasyon -- dijital ikizin karar veren katmani.

Neden turevsiz cozucu?
    RandomForest cikti yuzeyi parcali-sabittir; gradyan neredeyse her yerde
    sifirdir. SLSQP/L-BFGS gibi gradyan tabanli cozuculer bu yuzeyde ilk
    noktada takilir. Differential Evolution turev istemez ve basamakli
    yuzeylerde saglam calisir.

Neden kutu kisitlari egitim zarfiyla kesisiyor?
    RandomForest egitim araliginin DISINA ekstrapole edemez; disarida sabit
    deger basar. Bu, optimizasyonu kandirmak icin bire birdir: cozucu
    "buhari kapat, enerji sifir" der. Arama uzayini egitim verisinin gordugu
    yuzdeliklerle sinirlamak bu tuzagi kapatir.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from twin.config import Grade, Profile, Variable, get_settings
from twin.features.build import derived_for
from twin.optimize.objective import Economics, Objective

log = logging.getLogger(__name__)

_SUFFIX = re.compile(r"__(lag|rmean|rstd|d)\d+$")


def base_name(column: str) -> str:
    return _SUFFIX.sub("", column)


# --------------------------------------------------------------------------- #
def _assign_with_dynamics(frame: pd.DataFrame, name: str, values) -> None:
    """Bir degiskeni ve onun tureyen kolonlarini birlikte gunceller.

    KARARLI HAL VARSAYIMI: "bu setpointi koyup beklersem ne olur?" sorusunu
    soruyoruz. Dolayisiyla lag ve rolling-mean kopyalari da yeni degere set
    edilir, degisim (delta) sifirlanir.

    `rolling_std` DEGISTIRILMEZ: prosesin kararliligi setpoint degistirmekle
    aninda degismez; mevcut kararlilik seviyesi korunur.
    """
    for col in frame.columns:
        if base_name(col) != name:
            continue
        if col == name or "__lag" in col or "__rmean" in col:
            frame[col] = values
        elif "__d" in col:
            frame[col] = 0.0


def recompute_derived(frame: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """El yapimi ozellikleri yeni setpointlere gore yeniden hesaplar.

    Bu adim atlanirsa model fiziksel olarak imkansiz cevaplar verir: hizi 80
    m/dk artirirsiniz ama `fiber_flux` eski degerinde kaldigi icin model
    "uretim degismedi" der. Turetilmis ozelliklerin tanimi features/build.py
    icinde TEK YERDE durur; burada ayni sozluk kullanilir.
    """
    for name, spec in derived_for(profile.name).items():
        if name not in frame.columns and not any(base_name(c) == name for c in frame.columns):
            continue
        if not set(spec["inputs"]).issubset(frame.columns):
            continue
        _assign_with_dynamics(frame, name, spec["fn"](frame).values)
    return frame


def apply_setpoints(
    row: pd.Series, setpoints: Dict[str, float], profile: Optional[Profile] = None
) -> pd.Series:
    """Tek bir satira setpoint degisikligi uygular (what-if icin).

    `make_candidate_frame` ile AYNI yardimciyi kullanir; iki yolun ayrisip
    farkli sonuc uretmesi tam olarak bu projede avlamaya calistigimiz sessiz
    hata turudur.
    """
    frame = pd.DataFrame([row.values], columns=row.index)
    for name, value in setpoints.items():
        _assign_with_dynamics(frame, name, value)
    if profile is not None:
        recompute_derived(frame, profile)
    return pd.Series(frame.iloc[0].values, index=row.index, name=row.name)


def make_candidate_frame(
    row: pd.Series, names: List[str], matrix: np.ndarray, profile: Optional[Profile] = None
) -> pd.DataFrame:
    """(n_aday, n_degisken) matrisini tam ozellik tablosuna genisletir.

    Optimizasyon dongusu tek cagriyla yuzlerce adayi puanlar; satir satir
    predict cagirmak 100x yavas olurdu.
    """
    frame = pd.DataFrame([row.values] * len(matrix), columns=row.index)
    for j, name in enumerate(names):
        _assign_with_dynamics(frame, name, matrix[:, j])
    if profile is not None:
        recompute_derived(frame, profile)
    return frame


# --------------------------------------------------------------------------- #
@dataclass
class SearchSpace:
    """Aranacak degiskenler ve -- uc kez kesilmis -- sinirlari."""

    names: List[str]
    lower: np.ndarray
    upper: np.ndarray
    current: np.ndarray
    notes: List[str] = field(default_factory=list)

    def as_bounds(self) -> List[Tuple[float, float]]:
        return list(zip(self.lower.tolist(), self.upper.tolist()))


def build_search_space(
    profile: Profile,
    row: pd.Series,
    reference_bundle: Any,
    variables: Optional[Sequence[str]] = None,
    respect_rate_limits: bool = True,
) -> SearchSpace:
    """Sinirlari uc kaynagin KESISIMI olarak kurar:

        1. Emniyet araligi  (sensor_registry / profile.yaml op_low..op_high)
        2. Degisim hizi limiti (|x - mevcut| <= max_delta)
        3. Egitim veri zarfi   (p01..p99) -- ekstrapolasyon yasagi

    Zarf sinirlari egitim aninda `feature_stats` icine yazilan p01/p99'dur;
    burada yeniden hesaplanmaz (model neyi gordugunu ancak kendi kaydindan
    bilebilir).
    """
    names: List[str] = []
    lows: List[float] = []
    highs: List[float] = []
    currents: List[float] = []
    notes: List[str] = []

    candidates: List[Variable] = [
        v for v in profile.controllables
        if (variables is None or v.name in variables) and v.name in row.index
    ]

    for var in candidates:
        low, high = var.bounds
        current = float(row[var.name])

        if respect_rate_limits and var.max_delta:
            low = max(low, current - float(var.max_delta))
            high = min(high, current + float(var.max_delta))

        # Zarf, degiskenin TUM ailesi uzerinden kesilir: ham deger + lag'leri +
        # rolling ortalamalari. Kararli hal varsayimi geregi bu kolonlarin
        # hepsini ayni degere set ediyoruz; dolayisiyla hepsi kendi egitim
        # zarfinin icinde kalmali.
        #
        # Sadece ham degiskeni kesmek TUTARSIZLIK uretir: arama uzayi bir nokta
        # onerir, guven kontrolu ayni noktaya "ekstrapolasyon" der. Rolling
        # ortalamalarin dagilimi ani degerden DAHA DARDIR ("bu hizi 60 dakika
        # boyunca hic surdurmediniz"), yani asil kisitlayici olan odur.
        stats_by_column = reference_bundle.feature_stats or {}
        family = [
            col for col in row.index
            if base_name(col) == var.name and "__rstd" not in col and "__d" not in col
        ]
        trained_low, trained_high = -np.inf, np.inf
        for col in family:
            col_stats = stats_by_column.get(col)
            if col_stats:
                trained_low = max(trained_low, col_stats["p01"])
                trained_high = min(trained_high, col_stats["p99"])
        if np.isfinite(trained_low) and np.isfinite(trained_high):
            if low < trained_low or high > trained_high:
                notes.append(
                    f"{var.name}: arama araligi egitim zarfiyla kisitlandi "
                    f"[{max(low, trained_low):.1f}, {min(high, trained_high):.1f}]"
                )
            low = max(low, trained_low)
            high = min(high, trained_high)

        if high - low < 1e-6:
            continue
        names.append(var.name)
        lows.append(low)
        highs.append(high)
        currents.append(float(np.clip(current, low, high)))

    return SearchSpace(names, np.array(lows), np.array(highs), np.array(currents), notes)


# --------------------------------------------------------------------------- #
@dataclass
class OptimizationResult:
    setpoints: Dict[str, float]
    current_setpoints: Dict[str, float]
    predicted: Dict[str, float]
    predicted_current: Dict[str, float]
    cost: float
    cost_current: float
    saving_try_per_ton: float
    confidence: str
    violations: List[str]
    notes: List[str]
    n_evaluations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "setpoints": {k: round(v, 4) for k, v in self.setpoints.items()},
            "current_setpoints": {k: round(v, 4) for k, v in self.current_setpoints.items()},
            "predicted": {k: round(v, 4) for k, v in self.predicted.items()},
            "predicted_current": {k: round(v, 4) for k, v in self.predicted_current.items()},
            "cost_try_per_ton": round(self.cost, 2),
            "cost_current_try_per_ton": round(self.cost_current, 2),
            "saving_try_per_ton": round(self.saving_try_per_ton, 2),
            "confidence": self.confidence,
            "violations": self.violations,
            "notes": self.notes,
            "n_evaluations": self.n_evaluations,
        }


# --------------------------------------------------------------------------- #
def assess_confidence(
    bundles: Dict[str, Any],
    frame: pd.DataFrame,
    min_support: int,
    decision_variables: Optional[Sequence[str]] = None,
) -> Tuple[str, List[str]]:
    """Tavsiyeye ne kadar guvenilir?

    Iki bagimsiz kontrol:

      1. KUTU: onerdigimiz setpointler egitim zarfinin icinde mi?
      2. YOGUNLUK: bu calisma noktasina BENZER bir nokta egitimde kac kez
         gorulmus? Kutu kontrolu yetmez -- her degisken tek tek araliginda
         olup birlesimleri hic gorulmemis olabilir. Optimizasyonu kandiran
         nokta tam olarak orasidir.

    DIKKAT -- neden sadece karar degiskenlerine bakiyoruz: 200+ ozelligin
    HEPSINI p01-p99 araliginda istemek istatistiksel olarak imkansizdir
    (0.98^200 ~ %2). Boyle bir kontrol her tavsiyeyi "dusuk guven" isaretler
    ve hicbir sey ifade etmez. Ekstrapolasyon riski bizim OYNATTIGIMIZ
    degiskenlerden dogar; baglam degiskenleri zaten olculen gercekliktir.
    """
    notes: List[str] = []
    columns = None
    if decision_variables:
        wanted = set(decision_variables)
        columns = [c for c in frame.columns if base_name(c) in wanted]

    if not all(bundle.in_domain(frame, columns=columns).all() for bundle in bundles.values()):
        notes.append("Onerilen setpointler egitim verisinin gordugu araligin disinda -- EKSTRAPOLASYON.")
        return "low", notes

    measured = [
        float(np.min(values)) for values in (bundle.support(frame) for bundle in bundles.values())
        if np.isfinite(values).all()
    ]
    if not measured:
        # Olculemedi. "Yuksek" demek, olcmedigimiz bir seyi iddia etmektir.
        notes.append(
            "Yogunluk kontrolu yapilamadi (model paketinde egitim ornegi yok). "
            "Guven en fazla ORTA olarak raporlanir -- modeli yeniden egitmek bu kontrolu aktive eder."
        )
        return "medium", notes

    support = min(measured)
    if support < min_support:
        notes.append(
            f"Benzer rejim egitim verisinde az gozlenmis ({support:.0f} benzer nokta "
            f"< {min_support}). Tavsiye deneysel sayilmali."
        )
        return "medium", notes
    notes.append(f"Benzer rejim egitim verisinde bol gozlenmis ({support:.0f} benzer nokta).")
    return "high", notes


# --------------------------------------------------------------------------- #
def optimize_row(
    row: pd.Series,
    bundles: Dict[str, Any],
    profile: Profile,
    grade: Optional[Grade],
    hour: Optional[int] = None,
    variables: Optional[Sequence[str]] = None,
    respect_rate_limits: bool = True,
    settings: Optional[Any] = None,
) -> OptimizationResult:
    settings = settings or get_settings()
    cfg = settings.get_path("optimize", {}) or {}
    hour = hour if hour is not None else int(pd.Timestamp(row.name).hour) if row.name is not None else 12

    reference = bundles.get("sec_total_kwh_t") or next(iter(bundles.values()))
    space = build_search_space(profile, row, reference, variables, respect_rate_limits)
    objective = Objective(
        bundles, profile, grade,
        economics=Economics.from_settings(settings),
        penalty_weight=float(cfg.get("penalty_weight", 10000.0)),
    )

    if not space.names:
        raise ValueError("Optimize edilebilir degisken yok (setpoint tanimi veya veri eksik).")

    counter = {"n": 0}

    def cost_of(matrix: np.ndarray) -> np.ndarray:
        """DE `vectorized=True` ile (n_degisken, n_aday) verir -> toplu puanla."""
        matrix = np.atleast_2d(matrix)
        if matrix.shape[0] == len(space.names) and matrix.shape[1] != len(space.names):
            matrix = matrix.T
        counter["n"] += len(matrix)
        frame = make_candidate_frame(row, space.names, matrix, profile)
        return np.array([r.cost_try_per_ton for r in objective.evaluate(frame, hour)])

    solver = str(cfg.get("solver", "differential_evolution"))
    if solver == "differential_evolution":
        outcome = differential_evolution(
            cost_of,
            bounds=space.as_bounds(),
            maxiter=int(cfg.get("max_iter", 60)),
            popsize=int(cfg.get("population", 24)),
            seed=int(cfg.get("seed", 7)),
            polish=False,            # RF yuzeyinde gradyanli polish anlamsiz
            vectorized=True,
            updating="deferred",   # vectorized ile zorunlu; acikca yazip uyariyi susturuyoruz
            init="sobol",            # arama uzayini duzgun tarar
            tol=0.01,
        )
        best_x = outcome.x
    else:  # random_search -- bagimliliksiz yedek
        rng = np.random.default_rng(int(cfg.get("seed", 7)))
        n = int(cfg.get("max_iter", 60)) * int(cfg.get("population", 24))
        samples = rng.uniform(space.lower, space.upper, size=(n, len(space.names)))
        samples = np.vstack([space.current[None, :], samples])
        best_x = samples[int(np.argmin(cost_of(samples)))]

    # --- mevcut durum ile karsilastir --------------------------------------- #
    comparison = make_candidate_frame(row, space.names, np.vstack([space.current, best_x]), profile)
    current_result, best_result = objective.evaluate(comparison, hour)

    # Optimizasyon mevcut durumdan kotu cikarsa mevcut durumu koru (guvenli varsayilan)
    notes = list(space.notes)
    reverted = best_result.cost_try_per_ton >= current_result.cost_try_per_ton
    if reverted:
        notes.append("Mevcut ayarlardan daha iyi ve kisitlari saglayan bir nokta bulunamadi.")
        best_x = space.current
        best_result = current_result

    # Guven, GERCEKTEN tavsiye edilen nokta uzerinden olculur; mevcut ayarlara
    # geri donduysek reddedilen adayin guveni raporlanmamali.
    confidence, conf_notes = assess_confidence(
        bundles, comparison.iloc[[0 if reverted else 1]],
        int(cfg.get("min_support", 20)),
        decision_variables=space.names,
    )
    notes.extend(conf_notes)

    return OptimizationResult(
        setpoints=dict(zip(space.names, map(float, best_x))),
        current_setpoints=dict(zip(space.names, map(float, space.current))),
        predicted=best_result.predictions,
        predicted_current=current_result.predictions,
        cost=best_result.cost_try_per_ton,
        cost_current=current_result.cost_try_per_ton,
        saving_try_per_ton=current_result.energy_cost - best_result.energy_cost,
        confidence=confidence,
        violations=best_result.violations,
        notes=notes,
        n_evaluations=counter["n"],
    )


# --------------------------------------------------------------------------- #
def what_if(
    row: pd.Series,
    bundles: Dict[str, Any],
    profile: Profile,
    grade: Optional[Grade],
    setpoints: Dict[str, float],
    hour: Optional[int] = None,
) -> Dict[str, Any]:
    """Senaryo analizi: 'hizi 900'e cikarirsam ne olur?'

    En kolay satilan ozellik; ayni zamanda operator icin bir egitim araci.
    """
    hour = hour if hour is not None else int(pd.Timestamp(row.name).hour) if row.name is not None else 12
    objective = Objective(bundles, profile, grade)
    frame = pd.DataFrame(
        [row.values, apply_setpoints(row, setpoints, profile).values], columns=row.index
    )
    base, scenario = objective.evaluate(frame, hour)
    return {
        "baseline": base.predictions,
        "scenario": scenario.predictions,
        "delta": {k: scenario.predictions[k] - base.predictions[k] for k in scenario.predictions},
        "violations": scenario.violations,
        "cost_delta_try_per_ton": scenario.energy_cost - base.energy_cost,
    }
