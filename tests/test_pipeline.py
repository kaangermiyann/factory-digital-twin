"""Uctan uca boru hatti testleri.

Odak: "calisiyor mu" degil, **sessizce yanlis calisabilecek** yerler:
sizinti korumasi, zaman sirasi, ekstrapolasyon kilidi, birim donusumu.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from twin.config import get_profile
from twin.features.build import BREAK_LABEL, DOWNTIME_FLAG, build_feature_table, select_features
from twin.schema import Dataset, QualitySample, TelemetryPoint
from twin.storage.parquet import ParquetRepository


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("twin-data")
    repository = ParquetRepository(root=str(root))
    repository.ensure_schema()
    return repository


@pytest.fixture(scope="module")
def profile():
    return get_profile("paper")


@pytest.fixture(scope="module")
def populated(repo, profile):
    """3 gunluk simulasyonu test deposuna yazar."""
    from twin.simulator.paper import PaperMachineSimulator

    start = datetime.now(timezone.utc) - timedelta(days=3)
    sim = PaperMachineSimulator(profile, seed=11, start=start)

    telemetry, batches, quality, events = [], [], [], []
    for _ in range(3 * 24 * 60):
        step = sim.step()
        for name, value in step.values.items():
            if name not in profile.variables or not np.isfinite(value):
                continue
            var = profile.var(name)
            telemetry.append(TelemetryPoint(
                ts=step.ts, asset_id=var.asset, line_id=profile.line_id, tag=var.tag,
                variable=name, value=float(value), unit=var.unit, role=var.role,
                batch_id=step.batch_id,
            ).to_dict())
        for event in step.events:
            events.append({k: (v.isoformat() if isinstance(v, datetime) else v)
                           for k, v in event.items()})
        for sample in step.quality:
            row = dict(sample)
            row["ts"] = row["ts"].isoformat()
            quality.append(QualitySample(**row).to_dict())
        if step.closed_batch:
            batch = dict(step.closed_batch)
            batch["start_ts"] = batch["start_ts"].isoformat()
            batch["end_ts"] = batch["end_ts"].isoformat()
            batches.append(batch)

    repo.write(Dataset.TELEMETRY, telemetry)
    repo.write(Dataset.BATCHES, batches)
    repo.write(Dataset.QUALITY, quality)
    repo.write(Dataset.EVENTS, events)
    return repo


# --------------------------------------------------------------------------- #
# Konfigurasyon / profil
# --------------------------------------------------------------------------- #
def test_profile_roles_are_complete(profile):
    """Optimizasyon setpoint/olcum ayrimina bagli; ayrim yoksa proje calismaz."""
    assert profile.controllables, "Hicbir kontrol edilebilir degisken yok"
    for var in profile.controllables:
        assert var.op_low < var.op_high, f"{var.name}: gecersiz emniyet araligi"
        assert var.max_delta and var.max_delta > 0, f"{var.name}: degisim hizi limiti yok"


def test_every_target_exists_as_variable(profile):
    for target in profile.targets:
        if target == BREAK_LABEL:
            continue                     # turetilmis etiket, sensor degil
        assert target in profile.variables, f"Hedef profilde tanimli degil: {target}"


def test_grade_specs_are_ordered(profile):
    for code, grade in profile.grades.items():
        window = grade.spec("reel_moisture_pct")
        assert window and window[0] < window[1], f"{code}: gecersiz spec araligi"


# --------------------------------------------------------------------------- #
# Depolama
# --------------------------------------------------------------------------- #
def test_repository_roundtrip(repo):
    now = datetime.now(timezone.utc)
    rows = [
        {"ts": (now - timedelta(minutes=i)).isoformat(), "line_id": "T1", "asset_id": "A",
         "tag": "X", "variable": "v", "value": float(i), "unit": "u", "role": "measurement",
         "quality": 100, "batch_id": None}
        for i in range(10)
    ]
    assert repo.write(Dataset.TELEMETRY, rows) == 10
    got = repo.read(Dataset.TELEMETRY, filters={"line_id": "T1"})
    assert len(got) == 10
    assert pd.api.types.is_datetime64_any_dtype(got["ts"]), "ts datetime olarak donmedi"
    assert got["ts"].is_monotonic_increasing, "okuma zaman sirali degil"


def test_latest_returns_newest_first(repo):
    latest = repo.latest(Dataset.TELEMETRY, n=3, filters={"line_id": "T1"})
    assert len(latest) == 3
    assert latest["ts"].is_monotonic_decreasing


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
def test_simulation_is_physically_plausible(populated, profile):
    frame = build_feature_table(profile=profile, repo=populated)
    running = frame[frame[DOWNTIME_FLAG] == 0]

    assert running["reel_moisture_pct"].median() == pytest.approx(5.2, abs=1.5), \
        "Nem medyani fiziksel araligin disinda -- kurutma kalibrasyonu bozuk"
    assert running["press_dryness_pct"].between(30, 60).mean() > 0.95
    assert running["production_rate_tph"].median() > 5
    # Termal enerji elektrikten belirgin sekilde buyuk olmali (kagit makinesi)
    assert running["thermal_kwh_t"].median() > running["elec_kwh_t"].median()


def test_simulation_produces_variance_to_learn_from(populated, profile):
    """Hic degismeyen setpoint = ogrenilecek bilgi yok = optimize edilemez."""
    frame = build_feature_table(profile=profile, repo=populated)
    for var in profile.controllables:
        if var.name not in frame.columns:
            continue
        assert frame[var.name].std() > 1e-6, f"{var.name} hic degismemis"


def test_downtime_and_break_labels_exist(populated, profile):
    frame = build_feature_table(profile=profile, repo=populated)
    assert 0 < frame[DOWNTIME_FLAG].mean() < 0.5, "Durus orani gercekci degil"
    assert frame[BREAK_LABEL].sum() > 0, "Hic kopus etiketi uretilmemis"


# --------------------------------------------------------------------------- #
# Feature / sizinti
# --------------------------------------------------------------------------- #
def test_no_leakage_in_feature_selection(populated, profile):
    """En kritik test: hedefin kendisi ve bilesenleri ozellik olamaz.

    Bu kontrol dusersse model mukemmel gorunur ve tamamen ise yaramaz olur.
    """
    frame = build_feature_table(profile=profile, repo=populated)
    features = select_features(frame, profile, "sec_total_kwh_t")

    forbidden = {"sec_total_kwh_t", "elec_kwh_t", "thermal_kwh_t"}
    for name in forbidden:
        assert name not in features, f"Hedefin bileseni ozellik listesinde: {name}"
        # ...ama gecmis degerleri mesrudur
    assert all(f.split("__")[0] not in forbidden or "__" in f for f in features)

    for other in profile.targets:
        if other != "sec_total_kwh_t":
            assert other not in features, f"Baska bir hedef ozellik olmus: {other}"


def test_lab_results_do_not_leak_from_the_future(populated, profile):
    """Lab numunesi partinin SONUNDA alinir.

    Partinin ortasindaki bir satira o partinin lab sonucunu vermek, modele
    gelecegi gostermektir. Ozellik tablosu yalnizca BIR ONCEKI partinin lab
    sonucunu tasimali (`lab_prev_*`) ve guncel partininki hic bulunmamali.
    """
    frame = build_feature_table(profile=profile, repo=populated)
    lab_columns = [c for c in frame.columns if c.startswith("lab")]
    assert lab_columns, "lab kolonlari hic uretilmemis"
    assert all(c.startswith("lab_prev_") for c in lab_columns), \
        f"Guncel partinin lab sonucu sizmis: {lab_columns}"

    # Degerler gercekten bir onceki partiye mi ait?
    quality = populated.read(Dataset.QUALITY, filters={"line_id": profile.line_id})
    batches = populated.read(Dataset.BATCHES, filters={"line_id": profile.line_id})
    order = batches.drop_duplicates("batch_id").sort_values("start_ts")["batch_id"].tolist()
    truth = (
        quality[quality["property"] == "moisture"]
        .groupby("batch_id")["value"].mean().reindex(order)
    )
    sample = frame.dropna(subset=["lab_prev_moisture"]).iloc[len(frame) // 2]
    position = order.index(sample["batch_id"])
    assert position > 0
    assert sample["lab_prev_moisture"] == pytest.approx(truth.iloc[position - 1], abs=1e-6), \
        "lab_prev_* bir onceki partinin degeri degil"


def test_previous_batch_lab_is_a_legitimate_feature(populated, profile):
    """Bir onceki partinin lab sonucu MESRU bir ozelliktir, yasaklanmamali.

    Operator tahmin anında gercekten o bilgiye sahiptir. Yasaklanmasi gereken,
    tahmin edilen partinin KENDI lab sonucudur -- ve o zaten tabloda yok
    (bkz. test_lab_results_do_not_leak_from_the_future).
    """
    frame = build_feature_table(profile=profile, repo=populated)
    features = select_features(frame, profile, "reel_moisture_pct")

    assert "lab_prev_moisture" in features, "gecmis lab sonucu gereksiz yere dislanmis"
    # Guncel partinin lab sonucu hicbir isimle tabloda bulunmamali
    assert not any(c.startswith("lab_") and not c.startswith("lab_prev_") for c in frame.columns)


def test_dynamics_only_look_backwards(populated, profile):
    """Rolling/lag pencereleri gelecege bakmamali.

    Lag izgarasi config'ten OKUNUR, teste gomulmez: izgara degistiginde test
    sessizce atlanirsa koruma devre disi kalir ve kimse fark etmez.
    """
    from twin.config import get_settings

    settings = get_settings()
    frame = build_feature_table(profile=profile, repo=populated).reset_index(drop=True)
    step = int(pd.Timedelta(settings.get_path("features.resample", "5min")).total_seconds() // 60)
    lags = [lag for lag in settings.get_path("features.lags_min", []) if lag > 0]
    assert lags, "config'te lag tanimli degil -- olu zaman modellemesi kapali"

    checked = 0
    for col in ("steam_g3_bar", "machine_speed_mpm"):
        for lag in lags:
            name = f"{col}__lag{lag}"
            assert name in frame.columns, f"beklenen lag ozelligi uretilmemis: {name}"
            expected = frame[col].shift(max(1, lag // step))
            valid = expected.notna() & frame[name].notna()
            assert valid.any(), f"{name} tamamen bos"
            assert np.allclose(frame.loc[valid, name], expected[valid]), f"{name} gelecege bakiyor"
            checked += 1
    assert checked == 2 * len(lags)


def test_feature_table_is_time_ordered(populated, profile):
    frame = build_feature_table(profile=profile, repo=populated)
    assert frame["ts"].is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
def test_wide_csv_maps_to_canonical(profile, tmp_path):
    """Musteri genis CSV verir; tag_map ile canonical'a cevrilmeli."""
    from twin.ingest.loader import wide_to_canonical

    speed = profile.var("machine_speed_mpm")
    steam = profile.var("steam_g3_bar")
    raw = pd.DataFrame({
        "TimeStamp": pd.date_range("2026-01-01", periods=5, freq="1min"),
        speed.tag: [820, 825, 830, 828, 826],
        steam.tag: [3.8, 3.9, 3.9, 4.0, 4.0],
        "BILINMEYEN.TAG": [1, 2, 3, 4, 5],       # eslesmeyen kolon atlanmali
    })
    canonical = wide_to_canonical(raw, profile, source_tz="Europe/Istanbul")

    assert set(canonical["variable"]) == {"machine_speed_mpm", "steam_g3_bar"}
    assert str(canonical["ts"].dt.tz) == "UTC", "Zaman UTC'ye cevrilmedi"
    assert canonical["role"].isin(["setpoint", "measurement", "context"]).all()
    assert len(canonical) == 10


def test_unit_conversion():
    from twin.ingest.loader import convert_unit

    assert convert_unit(pd.Series([400.0]), "kPa", "bar").iloc[0] == pytest.approx(4.0)
    assert convert_unit(pd.Series([212.0]), "F", "C").iloc[0] == pytest.approx(100.0)
    # Tanimsiz donusumde deger BOZULMAMALI (sessiz veri hatasi olmasin)
    assert convert_unit(pd.Series([5.0]), "widget", "bar").iloc[0] == 5.0


def test_validation_flags_frozen_setpoint(profile):
    from twin.ingest.validate import validate_telemetry

    stamps = pd.date_range("2026-01-01", periods=400, freq="1min", tz="UTC")
    frame = pd.DataFrame({
        "ts": stamps,
        "line_id": profile.line_id,
        "asset_id": "A",
        "tag": profile.var("steam_g3_bar").tag,
        "variable": "steam_g3_bar",
        "value": 3.5,                                  # hic degismiyor
        "unit": "bar", "role": "setpoint", "quality": 100, "batch_id": None,
    })
    _, _, report = validate_telemetry(frame, profile)
    statuses = {r["degisken"]: r["durum"] for r in report.per_variable}
    assert statuses["steam_g3_bar"] in ("DONUK SENSOR", "SABIT (varyans yok)",
                                        "SETPOINT HIC DEGISMEMIS")
    assert any("setpoint" in w.lower() or "donuk" in w.lower() for w in report.warnings) \
        or report.per_variable, "Sorun raporlanmadi"


def test_validation_quarantines_out_of_range(profile):
    from twin.ingest.validate import validate_telemetry

    stamps = pd.date_range("2026-01-01", periods=20, freq="1min", tz="UTC")
    values = [850.0] * 19 + [99999.0]                  # son deger sensor arizasi
    frame = pd.DataFrame({
        "ts": stamps, "line_id": profile.line_id, "asset_id": "A",
        "tag": "T", "variable": "machine_speed_mpm", "value": values,
        "unit": "m/min", "role": "setpoint", "quality": 100, "batch_id": None,
    })
    accepted, rejected, report = validate_telemetry(frame, profile)
    assert len(rejected) == 1
    assert "aralik" in "".join(report.rejections).lower()
    assert len(accepted) == 19


# --------------------------------------------------------------------------- #
# Model secimi
# --------------------------------------------------------------------------- #
def test_linear_baseline_exists_for_both_problem_types():
    """Her iki problem tipinde de bir DOGRUSAL taban cizgisi egitilmeli.

    Regresyonda adi `ridge`, siniflandirmada `logistic`. Konfigurasyon sadece
    `ridge` yaziyor diye siniflandirma modelleri dogrusal referanssiz kalirsa
    "modelimiz dogrusaldan iyi" iddiasi kanitsiz kalir -- ve bu sessizce olur.
    """
    from twin.config import get_settings
    from twin.models.train import selected_model_names

    cfg = get_settings().get_path("training", {})
    for kind, linear in (("regression", "ridge"), ("classification", "logistic")):
        names = selected_model_names(kind, cfg)
        assert linear in names, f"{kind} icin dogrusal taban cizgisi yok: {names}"
        assert "naive" in names, f"{kind} icin naive taban cizgisi yok"
        assert any(n in names for n in ("random_forest", "gradient_boosting"))


def test_validation_names_the_unit_error(profile):
    """Birim hatasi, veri kesfi fazinin en sik ve en pahali bulgusudur.

    "4.570 kayit aralik disi" demek aksiyona donusmez; "steam_g3_bar tag'inin
    %100'u aralik disi -> birim hatasi" demek tek satirlik bir duzeltmeye
    donusur. Rapor bu ayrimi yapmak zorunda.
    """
    from twin.ingest.validate import validate_telemetry

    stamps = pd.date_range("2026-01-01", periods=60, freq="1min", tz="UTC")
    var = profile.var("steam_g3_bar")
    frame = pd.DataFrame({
        "ts": stamps, "line_id": profile.line_id, "asset_id": var.asset,
        "tag": var.tag, "variable": "steam_g3_bar",
        "value": np.linspace(280.0, 420.0, 60),      # bar yerine kPa gonderilmis
        "unit": "bar", "role": "setpoint", "quality": 100, "batch_id": None,
    })
    _, rejected, report = validate_telemetry(frame, profile)

    assert len(rejected) == 60
    top = report.rejected_by_variable[0]
    assert top["degisken"] == "steam_g3_bar"
    assert top["oran"] == pytest.approx(100.0)
    assert "BIRIM" in top["ipucu"], "birim hatasi ipucu verilmedi"


def test_recipe_context_is_not_called_a_stuck_sensor(profile):
    """Hedef gramaj kampanya boyunca sabit kalir -- bu ariza degil, prosesin dogasi."""
    from twin.ingest.validate import validate_telemetry

    # Kampanyalar donukluk penceresinden (120 dk) uzun olmali; gercek bir
    # kagit makinesinde grade kampanyasi zaten 6-26 saat surer.
    stamps = pd.date_range("2026-01-01", periods=1200, freq="1min", tz="UTC")
    values = np.repeat([80.0, 120.0, 45.0], 400)     # uc kampanya
    frame = pd.DataFrame({
        "ts": stamps, "line_id": profile.line_id, "asset_id": "PM2",
        "tag": profile.var("basis_weight_target_gsm").tag,
        "variable": "basis_weight_target_gsm", "value": values,
        "unit": "g/m2", "role": "context", "quality": 100, "batch_id": None,
    })
    _, _, report = validate_telemetry(frame, profile)
    status = {r["degisken"]: r["durum"] for r in report.per_variable}
    assert status["basis_weight_target_gsm"] == "KADEMELI (recete degeri)"


def test_categorical_codes_exist_in_both_paths(populated, profile):
    """Kategorik kodlar feature tablosunda uretilmeli, egitimde uydurulmamali.

    Kodlama sadece egitim tarafinda yapilirsa canli tahminde o kolon hic
    bulunmaz, medyanla doldurulur ve ozellik SESSIZCE olu kalir. Ekip etkisi
    bu projenin ana bulgularindan biri; boyle kaybedilemez.
    """
    from twin.features.build import make_supervised

    frame = build_feature_table(profile=profile, repo=populated)
    assert {"crew_id", "product_code_id"}.issubset(frame.columns)

    X, _, features = make_supervised(frame, profile, "sec_total_kwh_t")
    assert "crew_id" in features and "product_code_id" in features
    # Egitim matrisindeki her ozellik, canli feature tablosunda da bulunmali
    assert set(features) <= set(frame.columns), \
        f"Egitimde olup canlida olmayan ozellik: {sorted(set(features) - set(frame.columns))}"


def test_categorical_mapping_is_window_independent(populated, profile):
    """Ayni ekip, 6 saatlik pencerede de tum gecmiste de ayni kodu almali."""
    full = build_feature_table(profile=profile, repo=populated)
    cutoff = full["ts"].max() - timedelta(hours=6)
    window = build_feature_table(start=cutoff, profile=profile, repo=populated)

    pairs_full = full[["crew", "crew_id"]].drop_duplicates().set_index("crew")["crew_id"]
    pairs_window = window[["crew", "crew_id"]].drop_duplicates().set_index("crew")["crew_id"]
    shared = pairs_full.index.intersection(pairs_window.index)
    assert len(shared) > 0
    assert (pairs_full[shared] == pairs_window[shared]).all(), "kodlama pencereye gore degisiyor"


def test_tree_model_preferred_when_statistically_tied():
    """Secilen model, optimizasyonun uzerinde arama yaptigi VEKIL YUZEYDIR.

    Dogrusal model binde birlik bir dogruluk farkiyla onde olsa bile, sinirsiz
    ekstrapole ettigi icin cozucuye sahte kazanc gosterebilir. Agac modeli
    egitim destegi disinda doyuma gider. Bu yuzden yeterince yakinsa agac
    tercih edilmeli -- ve fark buyukse EDILMEMELI.
    """
    from twin.models.train import _apply_optimizer_safe_tiebreak

    def rows(ridge_mae, tree_mae):
        return [{"model": "ridge", "mae": ridge_mae},
                {"model": "gradient_boosting", "mae": tree_mae},
                {"model": "naive", "mae": 90.0}]

    # %0.5 fark -> agac tercih edilir
    picked, note = _apply_optimizer_safe_tiebreak(rows(19.25, 19.36), "mae", False, 0.03)
    assert picked["model"] == "gradient_boosting"
    assert note and "guvenli" in note.lower()

    # %20 fark -> dogrusal korunur, dogruluk feda edilmez
    picked, note = _apply_optimizer_safe_tiebreak(rows(10.0, 12.0), "mae", False, 0.03)
    assert picked["model"] == "ridge"
    assert note is None

    # Zaten agac kazandiysa dokunma
    picked, note = _apply_optimizer_safe_tiebreak(rows(25.0, 19.0), "mae", False, 0.03)
    assert picked["model"] == "gradient_boosting"
    assert note is None


def test_scrap_ratio_does_not_leak_from_the_future(populated, profile):
    """Iskarta orani parti KAPANDIGINDA bilinir -- ve kopus, iskartayi URETEN seydir.

    Partinin ortasindaki satira o partinin iskarta oranini vermek, modele olacak
    durusun izini gostermektir. Model "iskarta yuksekse kopus gelecek" diye
    ogrenir; holdout skoru parlar, sahada hicbir ise yaramaz. Bu, projedeki en
    sinsi sizinti turudur cunku kolon adi masum gorunur.
    """
    frame = build_feature_table(profile=profile, repo=populated)
    scrap_columns = [c for c in frame.columns if "scrap" in c]
    assert scrap_columns, "iskarta kolonu hic uretilmemis"
    assert all(c.startswith("prev_") for c in scrap_columns), \
        f"Guncel partinin iskartasi sizmis: {scrap_columns}"

    # Deger gercekten bir onceki partiye mi ait?
    batches = populated.read(Dataset.BATCHES, filters={"line_id": profile.line_id})
    order = batches.drop_duplicates("batch_id").sort_values("start_ts").reset_index(drop=True)
    truth = (order["scrap_qty"] / order["produced_qty"].replace(0, np.nan)).fillna(0.0)

    sample = frame.dropna(subset=["prev_batch_scrap_ratio"]).iloc[len(frame) // 2]
    position = order.index[order["batch_id"] == sample["batch_id"]][0]
    assert position > 0
    assert sample["prev_batch_scrap_ratio"] == pytest.approx(truth.iloc[position - 1], abs=1e-9), \
        "prev_batch_scrap_ratio bir onceki partinin degeri degil"


def test_no_batch_level_aggregate_leaks(populated, profile):
    """YAPISAL DEGISMEZ -- tek tek sizinti avlamak yerine kurali test et.

    Parti seviyesinde TOPLANAN her sey (iskarta, uretim miktari, lab sonucu,
    spec uyumu) ancak parti KAPANDIGINDA bilinir. Partinin ortasindaki bir
    satira bu degerleri vermek, modele gelecegi gostermektir.

    Bu projede ayni sinifta uc sizinti yakalandi: lab sonuclari, iskarta orani
    ve turetilmis `water_load` (uretim hizini iceriyordu). Ucu de kolon adi
    masum gorundugu icin gozden kacti. Bu test, YENI eklenen her parti-bazli
    ozelligi otomatik yakalar: ya `prev_` onekiyle gelecek ya da testi dusurecek.
    """
    frame = build_feature_table(profile=profile, repo=populated)

    batch_aggregate_markers = ("scrap", "lab_", "produced_qty", "all_passed")
    suspicious = [
        col for col in frame.columns
        if any(marker in col for marker in batch_aggregate_markers)
        and not (col.startswith("prev_") or col.startswith("lab_prev_"))
    ]
    assert not suspicious, (
        "Parti kapaninca bilinen degerler `prev_` oneki olmadan ozellik tablosunda: "
        f"{suspicious}. Bunlar tahmin aninda MEVCUT DEGILDIR."
    )


def test_predictions_are_deduplicated_on_read():
    """`make score` iki kez kosunca grafik ust uste cizmemeli.

    Tahmin deposu eklemelidir; ayni (zaman, hedef) icin birden fazla satir
    olusabilir. Metrikler bozulmaz ama grafik, tablo ve "kac tahmin var"
    sayisi yaniltici olur -- ve bu sessizce olur.
    """
    from twin.dashboard.data import deduplicate_predictions

    stamps = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    rows = pd.DataFrame({
        "ts": list(stamps) * 2,
        "target": ["sec_total_kwh_t"] * 6,
        "model_version": ["20260101T000000"] * 3 + ["20260202T000000"] * 3,
        "y_pred": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
        "y_true": [1.1, 2.1, 3.1, 1.1, 2.1, 3.1],
    })
    unique = deduplicate_predictions(rows)

    assert len(unique) == 3
    assert not unique.duplicated(subset=["ts", "target"]).any()
    # En YENI model surumu kazanmali
    assert unique["model_version"].unique().tolist() == ["20260202T000000"]
    assert unique["y_pred"].tolist() == [1.5, 2.5, 3.5]
