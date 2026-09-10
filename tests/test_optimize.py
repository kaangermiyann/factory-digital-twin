"""Optimizasyon katmani testleri.

Bu testlerin varlik sebebi: optimizasyon SESSIZCE saçma cevap verebilir.
Model egitim araliginin disina ekstrapole edemedigi icin cozucu "buhari kapat,
enerji sifir" gibi fiziksel olmayan noktalara kacar ve bu hicbir hata mesaji
uretmez. Asagidaki testler o kapiyi kilitli tutar.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from twin.config import get_profile
from twin.optimize.objective import Economics, build_constraints
from twin.optimize.search import (
    apply_setpoints,
    base_name,
    build_search_space,
    make_candidate_frame,
    optimize_row,
    recompute_derived,
    what_if,
)


@pytest.fixture(scope="module")
def profile():
    return get_profile("paper")


class FakeBundle:
    """Bilinen bir fonksiyonu taklit eden sahte model.

    Gercek modelle test etmek yavas ve belirsizdir; burada optimizasyonun
    DOGRU seyi arayip aramadigini kesin olarak olcebiliyoruz.
    """

    def __init__(self, target: str, fn, kind: str = "regression",
                 stats: Dict[str, Dict[str, float]] = None, features: List[str] = None):
        self.target = target
        self.kind = kind
        self.fn = fn
        self.features = features or []
        self.feature_stats = stats or {}
        self.algorithm = "fake"
        self.version = "test"
        self.metrics = {}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([self.fn(row) for _, row in frame.iterrows()], dtype=float)

    def in_domain(self, frame, tolerance=0.0, columns=None):
        checked = self.feature_stats if columns is None else {
            c: self.feature_stats[c] for c in columns if c in self.feature_stats
        }
        ok = pd.Series(True, index=frame.index)
        for col, stats in checked.items():
            if col in frame.columns:
                ok &= frame[col].between(stats["p01"], stats["p99"])
        return ok

    def support(self, frame, radius=0.15):
        return np.full(len(frame), 1000.0)


@pytest.fixture
def row(profile):
    """Nominal degerlerde, lag/rolling kopyalari olan bir ozellik vektoru."""
    values = {}
    for var in profile.variables.values():
        values[var.name] = float(var.nominal if var.nominal is not None else 1.0)
    values["ambient_temp_c"] = 12.0
    values["ambient_humidity_pct"] = 70.0
    values["pulp_freeness_ml"] = 380.0
    values["basis_weight_target_gsm"] = 80.0
    values["press_dryness_pct"] = 42.0
    # el yapimi (turetilmis) ozellikler -- setpoint degisince yeniden hesaplanmali
    values["fiber_flux"] = values["machine_speed_mpm"] * values["basis_weight_target_gsm"]
    values["steam_total_bar"] = (values["steam_g1_bar"] + values["steam_g2_bar"]
                                 + values["steam_g3_bar"])
    for name in ("steam_g3_bar", "machine_speed_mpm", "press_nip_load_knm",
                 "fiber_flux", "steam_total_bar"):
        values[f"{name}__lag10"] = values[name]
        values[f"{name}__rmean15"] = values[name]
        values[f"{name}__rstd15"] = 0.05
        values[f"{name}__d15"] = 0.0
    return pd.Series(values, name=pd.Timestamp("2026-03-04T03:00:00Z"))


def stats_for(profile) -> Dict[str, Dict[str, float]]:
    """Emniyet araliginin tamamini gormus bir egitim seti taklidi."""
    out = {}
    for var in profile.controllables:
        out[var.name] = {"min": var.op_low, "p01": var.op_low, "median": var.nominal,
                         "p99": var.op_high, "max": var.op_high}
    return out


# --------------------------------------------------------------------------- #
def test_base_name_strips_dynamic_suffixes():
    assert base_name("steam_g3_bar__lag10") == "steam_g3_bar"
    assert base_name("steam_g3_bar__rmean60") == "steam_g3_bar"
    assert base_name("steam_g3_bar__rstd15") == "steam_g3_bar"
    assert base_name("steam_g3_bar__d15") == "steam_g3_bar"
    assert base_name("steam_g3_bar") == "steam_g3_bar"


def test_apply_setpoints_propagates_to_lags(row):
    """Kararli hal varsayimi: lag ve rolling-mean kopyalari da guncellenir,
    delta sifirlanir, rolling-std KORUNUR."""
    before_std = row["steam_g3_bar__rstd15"]
    updated = apply_setpoints(row, {"steam_g3_bar": 2.0})

    assert updated["steam_g3_bar"] == 2.0
    assert updated["steam_g3_bar__lag10"] == 2.0
    assert updated["steam_g3_bar__rmean15"] == 2.0
    assert updated["steam_g3_bar__d15"] == 0.0
    assert updated["steam_g3_bar__rstd15"] == before_std, "kararlilik yanlislikla sifirlandi"
    assert updated["machine_speed_mpm"] == row["machine_speed_mpm"], "ilgisiz degisken bozuldu"


def test_candidate_frame_matches_apply_setpoints(row):
    """Toplu genisletme ile tekil uygulama ayni sonucu vermeli."""
    matrix = np.array([[2.0], [3.0]])
    frame = make_candidate_frame(row, ["steam_g3_bar"], matrix)
    single = apply_setpoints(row, {"steam_g3_bar": 3.0})
    assert frame.iloc[1]["steam_g3_bar__rmean15"] == single["steam_g3_bar__rmean15"]
    assert frame.iloc[0]["steam_g3_bar"] == 2.0


# --------------------------------------------------------------------------- #
def test_search_space_respects_rate_limits(profile, row):
    bundle = FakeBundle("sec_total_kwh_t", lambda r: 0.0, stats=stats_for(profile))
    space = build_search_space(profile, row, bundle, respect_rate_limits=True)

    for i, name in enumerate(space.names):
        var = profile.var(name)
        assert space.upper[i] - space.current[i] <= var.max_delta + 1e-9
        assert space.current[i] - space.lower[i] <= var.max_delta + 1e-9


def test_search_space_clipped_to_training_envelope(profile, row):
    """EN KRITIK KORUMA: egitimde gorulmemis bolge aranmaz."""
    narrow = stats_for(profile)
    narrow["steam_g3_bar"] = {"min": 3.8, "p01": 3.8, "median": 4.0, "p99": 4.2, "max": 4.2}
    bundle = FakeBundle("sec_total_kwh_t", lambda r: 0.0, stats=narrow)

    space = build_search_space(profile, row, bundle, respect_rate_limits=False)
    i = space.names.index("steam_g3_bar")
    assert space.lower[i] >= 3.8 - 1e-9
    assert space.upper[i] <= 4.2 + 1e-9
    assert any("egitim zarfiyla" in note for note in space.notes), "kisitlama raporlanmadi"


# --------------------------------------------------------------------------- #
def test_optimizer_finds_the_known_optimum(profile, row):
    """Enerji buharla artiyorsa cozucu buhari DUSURMELI -- kalite kisiti izin
    verdigi olcude."""
    stats = stats_for(profile)
    bundles = {
        "sec_total_kwh_t": FakeBundle(
            "sec_total_kwh_t",
            lambda r: 900.0 + 120.0 * r["steam_g3_bar"] + 0.15 * r["machine_speed_mpm"],
            stats=stats),
        "reel_moisture_pct": FakeBundle(
            "reel_moisture_pct",
            lambda r: 11.0 - 1.45 * r["steam_g3_bar"],       # buhar dusunce nem ARTAR
            stats=stats),
        "production_rate_tph": FakeBundle(
            "production_rate_tph", lambda r: 0.031 * r["machine_speed_mpm"], stats=stats),
    }
    grade = profile.grade("TL80")           # nem spec 4.8 - 6.2

    result = optimize_row(row, bundles, profile, grade, hour=3, respect_rate_limits=False)

    assert result.setpoints["steam_g3_bar"] < row["steam_g3_bar"], "buhar dusurulmedi"
    # Nem spec'in ustune cikmamali: 11 - 1.45*P <= 6.2  ->  P >= 3.31
    assert result.setpoints["steam_g3_bar"] >= 3.25, "kalite kisiti ihlal edildi"
    assert result.saving_try_per_ton > 0
    assert result.confidence in ("high", "medium", "low")


def test_optimizer_never_returns_worse_than_current(profile, row):
    """Kisitlar cozumsuzse mevcut ayarlar korunur -- kotu tavsiye uretilmez."""
    stats = stats_for(profile)
    bundles = {
        "sec_total_kwh_t": FakeBundle("sec_total_kwh_t", lambda r: 1200.0, stats=stats),
        "reel_moisture_pct": FakeBundle("reel_moisture_pct", lambda r: 5.4, stats=stats),
        "production_rate_tph": FakeBundle("production_rate_tph", lambda r: 25.0, stats=stats),
    }
    result = optimize_row(row, bundles, profile, profile.grade("TL80"), hour=12)
    assert result.cost <= result.cost_current + 1e-6
    assert result.saving_try_per_ton >= -1e-6


def test_constraint_violation_is_reported(profile, row):
    """Kisit saglanamiyorsa sessiz kalinmaz."""
    stats = stats_for(profile)
    bundles = {
        "sec_total_kwh_t": FakeBundle("sec_total_kwh_t", lambda r: 1300.0, stats=stats),
        "reel_moisture_pct": FakeBundle("reel_moisture_pct", lambda r: 9.9, stats=stats),  # spec disi
        "production_rate_tph": FakeBundle("production_rate_tph", lambda r: 25.0, stats=stats),
    }
    result = optimize_row(row, bundles, profile, profile.grade("TL80"), hour=12)
    assert result.violations, "spec ihlali raporlanmadi"


# --------------------------------------------------------------------------- #
def test_what_if_reports_direction_and_cost(profile, row):
    stats = stats_for(profile)
    bundles = {
        "sec_total_kwh_t": FakeBundle(
            "sec_total_kwh_t", lambda r: 900.0 + 120.0 * r["steam_g3_bar"], stats=stats),
    }
    outcome = what_if(row, bundles, profile, profile.grade("TL80"),
                      {"steam_g3_bar": row["steam_g3_bar"] + 1.0}, hour=3)
    assert outcome["delta"]["sec_total_kwh_t"] == pytest.approx(120.0, rel=1e-6)
    assert outcome["cost_delta_try_per_ton"] > 0


# --------------------------------------------------------------------------- #
def test_tariff_makes_peak_hours_more_expensive():
    economics = Economics(electricity_try_per_kwh=3.0, peak_multiplier=2.0,
                          offpeak_multiplier=0.5)
    assert economics.electricity_price(19) == pytest.approx(6.0)
    assert economics.electricity_price(3) == pytest.approx(1.5)
    assert economics.electricity_price(11) == pytest.approx(3.0)


def test_constraints_are_derived_from_grade_specs(profile):
    constraints = build_constraints(profile, profile.grade("TL80"))
    names = {c.target for c in constraints}
    assert "reel_moisture_pct" in names
    assert "production_rate_tph" in names

    moisture = next(c for c in constraints if c.target == "reel_moisture_pct")
    assert moisture.violation(5.5) == 0.0
    assert moisture.violation(7.0) > 0.0
    assert moisture.violation(3.0) > 0.0


# --------------------------------------------------------------------------- #
# Turetilmis ozelliklerin yeniden hesaplanmasi
# --------------------------------------------------------------------------- #
def test_derived_features_are_recomputed(profile, row):
    """Setpoint degisince el yapimi ozellikler de degismeli.

    Bu adim atlanirsa model "hizi 80 m/dk artirdim ama lif akisi ayni kaldi"
    durumunu gorur ve fiziksel olarak imkansiz bir cevap uretir -- hicbir hata
    mesaji da vermez. Sessiz yanlislik oldugu icin testi var.
    """
    new_speed = row["machine_speed_mpm"] + 80.0
    updated = apply_setpoints(row, {"machine_speed_mpm": new_speed}, profile)

    expected = new_speed * row["basis_weight_target_gsm"]
    assert updated["fiber_flux"] == pytest.approx(expected)
    assert updated["fiber_flux__lag10"] == pytest.approx(expected)
    assert updated["fiber_flux__d15"] == 0.0
    assert updated["fiber_flux__rstd15"] == row["fiber_flux__rstd15"], "kararlilik bozuldu"


def test_derived_recompute_is_vectorised(profile, row):
    """Toplu genisletme, tekil uygulamayla ayni sonucu vermeli."""
    speeds = np.array([[700.0], [900.0]])
    frame = make_candidate_frame(row, ["machine_speed_mpm"], speeds, profile)
    single = apply_setpoints(row, {"machine_speed_mpm": 900.0}, profile)
    assert frame.iloc[1]["fiber_flux"] == pytest.approx(single["fiber_flux"])
    assert frame.iloc[0]["fiber_flux"] == pytest.approx(700.0 * row["basis_weight_target_gsm"])


def test_recompute_derived_updates_steam_total(profile, row):
    frame = pd.DataFrame([row.values], columns=row.index)
    frame["steam_g1_bar"] = 1.0
    frame["steam_g2_bar"] = 2.0
    frame["steam_g3_bar"] = 3.0
    recompute_derived(frame, profile)
    assert frame.iloc[0]["steam_total_bar"] == pytest.approx(6.0)
    assert frame.iloc[0]["steam_total_bar__rmean15"] == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
def test_infeasible_point_never_beats_a_feasible_one(profile, row):
    """Kucuk bir spec ihlali, buyuk bir tasarrufla 'satin alinamaz'.

    Sadece kareli ceza kullanilsaydi cozucu "spec'i 0.02 asip 19 TL kazanayim"
    derdi ve SPEC DISI tavsiye uretirdi.
    """
    stats = stats_for(profile)
    # Buhar dustukce enerji ucuzlar ama nem yukselir; 4.5 bar'in altinda spec disi.
    bundles = {
        "sec_total_kwh_t": FakeBundle(
            "sec_total_kwh_t", lambda r: 100.0 * r["steam_g3_bar"], stats=stats),
        "reel_moisture_pct": FakeBundle(
            "reel_moisture_pct", lambda r: 6.2 + 2.0 * (4.5 - r["steam_g3_bar"]), stats=stats),
        "production_rate_tph": FakeBundle(
            "production_rate_tph", lambda r: 25.0, stats=stats),
    }
    result = optimize_row(row, bundles, profile, profile.grade("TL80"),
                          hour=3, variables=["steam_g3_bar"], respect_rate_limits=False)
    # spec ust siniri 6.2 -> buhar 4.5'in altina inemez
    assert result.setpoints["steam_g3_bar"] >= 4.5 - 1e-3
    assert not result.violations


# --------------------------------------------------------------------------- #
# Yogunluk (joint support) kontrolu
# --------------------------------------------------------------------------- #
def test_support_counts_similar_training_points(profile):
    """Kutu kontrolu yetmez: her degisken tek tek araliginda olup birlesimleri
    hic gorulmemis olabilir. Yogunluk kontrolu tam olarak bunu olcer."""
    from twin.models.registry import ModelBundle, compute_feature_stats

    rng = np.random.default_rng(0)
    # Egitim: hiz ve buhar KORELE gitmis (yuksek hiz -> yuksek buhar)
    speed = rng.uniform(500, 700, 500)
    steam = speed / 200.0 + rng.normal(0, 0.05, 500)
    X = pd.DataFrame({"machine_speed_mpm": speed, "steam_g3_bar": steam})

    bundle = ModelBundle(target="t", kind="regression", model=None,
                         features=list(X.columns), version="v", profile="paper", line_id="PM2",
                         feature_stats=compute_feature_stats(X))
    bundle.attach_support(X)

    # Korelasyonu izleyen nokta: bol gozlenmis
    on_manifold = pd.DataFrame([{"machine_speed_mpm": 600.0, "steam_g3_bar": 3.0}])
    # Her ikisi de kendi araliginda ama BIRLESIMI hic gorulmemis
    off_manifold = pd.DataFrame([{"machine_speed_mpm": 690.0, "steam_g3_bar": 2.55}])

    assert bundle.in_domain(off_manifold).all(), "kutu kontrolu bu noktayi gecirir -- test anlamsizlasir"
    assert bundle.support(on_manifold)[0] > 20
    assert bundle.support(off_manifold)[0] == 0, "kutu ici ama rejim disi nokta yakalanamadi"


def test_support_returns_nan_when_not_measurable(profile):
    """Olculemeyeni 'yuksek' diye raporlamak, hic raporlamamaktan kotudur."""
    from twin.models.registry import ModelBundle

    bundle = ModelBundle(target="t", kind="regression", model=None, features=["a"],
                         version="v", profile="paper", line_id="PM2")
    values = bundle.support(pd.DataFrame([{"a": 1.0}]))
    assert np.isnan(values).all()


def test_confidence_caps_at_medium_without_support(profile, row):
    """Yogunluk olculemiyorsa guven en fazla ORTA olmali."""
    from twin.optimize.search import assess_confidence

    class NoSupport(FakeBundle):
        def support(self, frame, radius=0.15):
            return np.full(len(frame), np.nan)

    bundles = {"sec_total_kwh_t": NoSupport("sec_total_kwh_t", lambda r: 1.0,
                                            stats=stats_for(profile))}
    frame = pd.DataFrame([row.values], columns=row.index)
    level, notes = assess_confidence(bundles, frame, min_support=20,
                                     decision_variables=["steam_g3_bar"])
    assert level == "medium"
    assert any("olcul" in n.lower() or "yapilamadi" in n.lower() for n in notes)
