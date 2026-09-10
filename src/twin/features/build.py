"""Feature engineering.

long telemetri -> wide model tablosu.

Bu katmanin tasidigi uc kritik karar:

1. OLU ZAMAN (dead time). Kurutma grubundaki buhar degisimi sariciya 8-15 dk
   sonra yansir. Ayni andaki degerlerle model kurmak fiziksel olarak yanlistir
   -> lag / rolling / delta ozellikleri.

2. KARARLILIK. rolling_std, ortalama kadar degerlidir: kararsiz proses = yuksek
   enerji + dusuk kalite. Modelin en cok kullandigi ozelliklerden biri cikar.

3. SIZINTI (leakage). Hedefin kendisinden turemis kolonlar dislanmali.
   `profile.targets[...].exclude_features` bu isi yapar; unutulursa model
   mukemmel gorunur ve ise yaramaz.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from twin.config import Profile, get_profile, get_settings
from twin.schema import Dataset
from twin.storage import Repository, get_repository

log = logging.getLogger(__name__)

DOWNTIME_FLAG = "downtime_flag"
BREAK_LABEL = "break_next_30m"


# --------------------------------------------------------------------------- #
# 1. Ham veriyi wide tabloya cevir
# --------------------------------------------------------------------------- #
def load_wide_telemetry(
    repo: Repository,
    profile: Profile,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    resample: str = "5min",
) -> pd.DataFrame:
    """long (ts, variable, value) -> wide (ts, var1, var2, ...), yeniden orneklenmis."""
    raw = repo.read(
        Dataset.TELEMETRY,
        start=start,
        end=end,
        filters={"line_id": profile.line_id},
        columns=["ts", "variable", "value", "batch_id"],
    )
    if raw.empty:
        return pd.DataFrame()

    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    wide = raw.pivot_table(index="ts", columns="variable", values="value", aggfunc="mean")
    wide = wide.resample(resample).mean()

    # batch_id sayisal degil -> ayri, mod (en sik) ile
    batch = (
        raw.dropna(subset=["batch_id"])
        .set_index("ts")["batch_id"]
        .resample(resample)
        .agg(lambda s: s.mode().iloc[0] if len(s) else np.nan)
    )
    wide["batch_id"] = batch
    wide["line_id"] = profile.line_id
    return wide.reset_index()


# --------------------------------------------------------------------------- #
# 2. Baglam veri kumelerini birlestir
# --------------------------------------------------------------------------- #
def join_batches(frame: pd.DataFrame, repo: Repository, profile: Profile) -> pd.DataFrame:
    """Urun kodu / vardiya / ekip bilgisini batch tablosundan getir."""
    batches = repo.read(Dataset.BATCHES, filters={"line_id": profile.line_id})
    if batches.empty:
        frame["product_code"] = "UNKNOWN"
        frame["shift_id"] = ((frame["ts"].dt.hour // 8) + 1).astype(int)
        frame["crew"] = "?"
        frame["product_code_id"] = -1
        frame["crew_id"] = -1
        return frame

    cols = ["batch_id", "product_code", "shift_id", "crew", "produced_qty", "scrap_qty"]
    slim = batches[[c for c in cols if c in batches.columns]].drop_duplicates("batch_id")
    merged = frame.merge(slim, on="batch_id", how="left")
    merged["product_code"] = merged["product_code"].ffill().fillna("UNKNOWN")
    merged["shift_id"] = ((merged["ts"].dt.hour // 8) + 1).astype(int)
    merged["crew"] = merged["crew"].ffill().fillna("?")

    # Kategorik -> sayisal kodlama BURADA yapilir, egitimde degil.
    #
    # Neden onemli: kodlama egitim tarafinda `cat.codes` ile uretilirse canli
    # tarafta hic uretilmez; model o ozelligi ogrenir ama serviste HER ZAMAN
    # medyanla doldurulmus gorur. Hicbir hata mesaji cikmaz, ozellik olu kalir.
    # Ekip etkisi bu projenin satis argumanlarindan biri oldugu icin bu sessiz
    # kayip ozellikle pahaliya mal olur.
    #
    # Esleme pencereye degil, TUM gecmise gore kurulur; boylece 6 saatlik bir
    # pencerede de 90 gunluk egitimde de ayni ekip ayni kodu alir.
    merged["product_code_id"] = merged["product_code"].map(
        {code: i for i, code in enumerate(sorted(profile.grades))}
    ).fillna(-1).astype(int)
    crews = sorted(batches["crew"].dropna().unique()) if "crew" in batches.columns else []
    merged["crew_id"] = merged["crew"].map(
        {crew: i for i, crew in enumerate(crews)}
    ).fillna(-1).astype(int)
    # --- iskarta orani: BIR ONCEKI partininki ------------------------------- #
    #
    # ZAMAN SIZINTISI: bir partinin iskartasi ancak parti KAPANDIGINDA bilinir.
    # Dahasi, kopus/durus iskartayi URETEN seydir. Partinin ortasindaki bir
    # satira o partinin iskarta oranini vermek, modele "olacak kopusun izini"
    # gostermektir: model "iskarta yuksekse kopus gelecek" diye ogrenir, holdout
    # skoru parlar ve sahada HICBIR ISE YARAMAZ (canli veride o sayi yok).
    #
    # Operatorun tahmin aninda sahip oldugu bilgi, bir onceki reel/heat'in
    # iskarta oranidir. Ad da bunu soylesin ki ileride yanlislikla "guncel"
    # gibi kullanilmasin.
    if {"produced_qty", "scrap_qty"}.issubset(slim.columns):
        order = (
            batches.dropna(subset=["batch_id"])
            .drop_duplicates("batch_id")
            .sort_values("start_ts")
        )
        denom = order["produced_qty"].replace(0, np.nan)
        ratio = (order["scrap_qty"] / denom).fillna(0.0)
        previous = pd.DataFrame({
            "batch_id": order["batch_id"].to_numpy(),
            "prev_batch_scrap_ratio": ratio.shift(1).to_numpy(),
        })
        merged = merged.merge(previous, on="batch_id", how="left")
    merged = merged.drop(columns=[c for c in ("produced_qty", "scrap_qty") if c in merged.columns])
    return merged


def join_events(frame: pd.DataFrame, repo: Repository, profile: Profile, horizon_min: int = 30) -> pd.DataFrame:
    """Durus bayragi + 'onumuzdeki N dk icinde plansiz durus' etiketi.

    Etiket, olayin BASLANGICINA gore kurulur; durusun kendi suresi etiketlenmez.
    Aksi halde model "hiz 0 -> durus var" gibi ise yaramaz bir kural ogrenir.
    """
    frame[DOWNTIME_FLAG] = 0
    frame[BREAK_LABEL] = 0
    events = repo.read(Dataset.EVENTS, filters={"line_id": profile.line_id})
    if events.empty:
        return frame

    events["ts_start"] = pd.to_datetime(events["ts_start"], utc=True)
    events["ts_end"] = pd.to_datetime(events["ts_end"], utc=True)
    ts = frame["ts"]

    for _, ev in events.iterrows():
        if ev.get("type") != "downtime":
            continue
        frame.loc[(ts >= ev["ts_start"]) & (ts < ev["ts_end"]), DOWNTIME_FLAG] = 1
        if ev.get("category") == "unplanned":
            window_start = ev["ts_start"] - timedelta(minutes=horizon_min)
            frame.loc[(ts >= window_start) & (ts < ev["ts_start"]), BREAK_LABEL] = 1
    return frame


def join_quality(frame: pd.DataFrame, repo: Repository, profile: Profile) -> pd.DataFrame:
    """Lab sonuclarini bagla -- ama BIR ONCEKI partinin sonuclarini.

    ZAMAN SIZINTISI: Laboratuvar numunesi partinin SONUNDA alinir. Partinin
    ortasindaki bir satira o partinin lab sonucunu ozellik olarak vermek,
    modele gelecegi gostermektir; model mukemmel gorunur ve sahada cuvallar.

    Gercek sistemde operatorun elinde olan bilgi, BIR ONCEKI reel/heat'in lab
    sonucudur. Burada da o modellenir: partiler zaman sirasina dizilir ve lab
    degerleri bir parti kaydirilir. Kolon adlari `lab_prev_*` -- anlam adin
    icinde olsun ki ileride yanlislikla "guncel lab" gibi kullanilmasin.
    """
    if "batch_id" not in frame.columns:
        return frame
    quality = repo.read(Dataset.QUALITY, filters={"line_id": profile.line_id})
    batches = repo.read(Dataset.BATCHES, filters={"line_id": profile.line_id})
    if quality.empty or batches.empty:
        return frame

    order = (
        batches.dropna(subset=["batch_id"])
        .drop_duplicates("batch_id")
        .sort_values("start_ts")["batch_id"]
        .tolist()
    )
    pivot = quality.pivot_table(index="batch_id", columns="property", values="value", aggfunc="mean")
    passed = quality.groupby("batch_id")["passed"].min().rename("all_passed")
    lab = pivot.join(passed).reindex(order)

    # Kaydirma: her parti, KENDINDEN ONCEKI partinin lab sonucunu gorur.
    lab = lab.shift(1)
    lab.columns = [f"lab_prev_{c}" for c in lab.columns]
    lab.index.name = "batch_id"
    return frame.merge(lab.reset_index(), on="batch_id", how="left")


# --------------------------------------------------------------------------- #
# 3. Turetilmis ozellikler
# --------------------------------------------------------------------------- #
def add_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    ts = frame["ts"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["dayofweek"] = ts.dt.dayofweek
    frame["is_weekend"] = (frame["dayofweek"] >= 5).astype(int)
    return frame


def add_dynamics(
    frame: pd.DataFrame,
    columns: List[str],
    lags_min: List[int],
    rolling_min: List[int],
    delta_min: List[int],
    step_min: int,
) -> pd.DataFrame:
    """Lag / rolling mean / rolling std / delta.

    Butun pencereler GECMISE bakar (`shift(1)` sonrasi rolling), gelecege sizmaz.
    """
    new: Dict[str, pd.Series] = {}
    for col in columns:
        if col not in frame.columns:
            continue
        series = frame[col]
        for lag in lags_min:
            if lag == 0:
                continue
            new[f"{col}__lag{lag}"] = series.shift(max(1, lag // step_min))
        for window in rolling_min:
            periods = max(2, window // step_min)
            shifted = series.shift(1)
            new[f"{col}__rmean{window}"] = shifted.rolling(periods, min_periods=2).mean()
            new[f"{col}__rstd{window}"] = shifted.rolling(periods, min_periods=2).std()
        for delta in delta_min:
            new[f"{col}__d{delta}"] = series - series.shift(max(1, delta // step_min))
    return pd.concat([frame, pd.DataFrame(new, index=frame.index)], axis=1)


# --------------------------------------------------------------------------- #
# Turetilmis (el yapimi) ozellikler
#
# Her tanim GIRDILERINI acikca bildirir. Bu iki sey icin sart:
#
#   1. SIZINTI: bir turetilmis ozellik, hedefin kendisinden hesaplanmissa
#      hedefi gizlice ozellik listesine sokar. `inputs` sayesinde
#      `select_features` bunu yakalayip diskaliye eder.
#   2. OPTIMIZASYON: setpoint degisince turetilmis ozellik de degismek
#      zorundadir. Yeniden hesaplamazsak model "hizi 80 artirdim ama uretim
#      degismedi" gibi fiziksel olarak imkansiz cevaplar verir.
#
# Bu yuzden tanimlar TEK YERDE durur ve hem feature pipeline'i hem optimizasyon
# ayni sozlugu kullanir.
# --------------------------------------------------------------------------- #
DERIVED: Dict[str, Dict[str, Any]] = {
    "steam_total_bar": {
        "profile": "paper",
        "inputs": ["steam_g1_bar", "steam_g2_bar", "steam_g3_bar"],
        "fn": lambda f: f["steam_g1_bar"] + f["steam_g2_bar"] + f["steam_g3_bar"],
        "desc": "Toplam kurutma buhari",
    },
    "steam_profile_ratio": {
        "profile": "paper",
        "inputs": ["steam_g1_bar", "steam_g3_bar"],
        "fn": lambda f: f["steam_g3_bar"] / f["steam_g1_bar"].replace(0, np.nan),
        "desc": "Kurutma profilinin dikligi (son grup / ilk grup)",
    },
    "fiber_flux": {
        "profile": "paper",
        "inputs": ["machine_speed_mpm", "basis_weight_target_gsm"],
        "fn": lambda f: f["machine_speed_mpm"] * f["basis_weight_target_gsm"],
        "desc": "Lif akisi (hiz x gramaj) -- uretim hizinin surucusu",
    },
    "water_load_index": {
        "profile": "paper",
        # DIKKAT: production_rate_tph KULLANILMAZ -- o bir hedef, kullanilsaydi
        # SEC modeline sizardi. Ayni fizik hiz x gramaj uzerinden kuruluyor.
        "inputs": ["machine_speed_mpm", "basis_weight_target_gsm", "press_dryness_pct"],
        "fn": lambda f: (
            f["machine_speed_mpm"] * f["basis_weight_target_gsm"]
            * (100.0 / f["press_dryness_pct"].clip(lower=1.0) - 1.0) / 1e4
        ),
        "desc": "Kurutmaya giren su yuku -- termal enerjinin gercek surucusu",
    },
    "chemical_energy_index": {
        "profile": "steel",
        "inputs": ["oxygen_nm3_t", "carbon_inject_kg_t"],
        "fn": lambda f: 3.5 * f["oxygen_nm3_t"] + 2.4 * f["carbon_inject_kg_t"],
        "desc": "Kimyasal enerji katkisi (elektrigi ikame eder)",
    },
    "reheat_mean_c": {
        "profile": "steel",
        "inputs": ["reheat_zone1_c", "reheat_zone2_c", "reheat_zone3_c"],
        "fn": lambda f: (f["reheat_zone1_c"] + f["reheat_zone2_c"] + f["reheat_zone3_c"]) / 3.0,
        "desc": "Tav firini ortalama bolge sicakligi",
    },
    "stack_loss_index": {
        "profile": "steel",
        "inputs": ["excess_air_ratio", "reheat_zone1_c", "reheat_zone2_c", "reheat_zone3_c"],
        "fn": lambda f: (f["excess_air_ratio"] - 1.0)
        * (f["reheat_zone1_c"] + f["reheat_zone2_c"] + f["reheat_zone3_c"]) / 3.0,
        "desc": "Bacadan giden enerji gostergesi",
    },
}


def derived_for(profile_name: str) -> Dict[str, Dict[str, Any]]:
    return {k: v for k, v in DERIVED.items() if v["profile"] == profile_name}


def add_domain_ratios(frame: pd.DataFrame, profile: Profile) -> pd.DataFrame:
    """Proses bilgisiyle el yapimi birkac oran. Az ama oz."""
    for name, spec in derived_for(profile.name).items():
        if set(spec["inputs"]).issubset(frame.columns):
            frame[name] = spec["fn"](frame)
    return frame


# --------------------------------------------------------------------------- #
# 4. Ana giris noktasi
# --------------------------------------------------------------------------- #
def build_feature_table(
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    profile: Optional[Profile] = None,
    repo: Optional[Repository] = None,
    with_dynamics: bool = True,
) -> pd.DataFrame:
    settings = get_settings()
    profile = profile or get_profile()
    repo = repo or get_repository()

    resample = settings.get_path("features.resample", "5min")
    step_min = int(pd.Timedelta(resample).total_seconds() // 60)

    frame = load_wide_telemetry(repo, profile, start, end, resample)
    if frame.empty:
        log.warning("Telemetri bulunamadi -- once simulatoru calistirin.")
        return frame

    frame = join_batches(frame, repo, profile)
    horizon = 30
    if BREAK_LABEL in profile.targets:
        horizon = int(profile.targets[BREAK_LABEL].horizon_min or 30)
    frame = join_events(frame, repo, profile, horizon_min=horizon)
    frame = join_quality(frame, repo, profile)
    frame = add_calendar(frame)
    frame = add_domain_ratios(frame, profile)

    if with_dynamics:
        dynamic_cols = (
            [v.name for v in profile.controllables]
            + [v.name for v in profile.contexts]
            + ["press_dryness_pct"]
            + list(derived_for(profile.name))
        )
        dynamic_cols = [c for c in dynamic_cols if c in frame.columns]
        frame = add_dynamics(
            frame,
            columns=sorted(set(dynamic_cols)),
            lags_min=list(settings.get_path("features.lags_min", [0, 5, 10, 15, 30])),
            rolling_min=list(settings.get_path("features.rolling_min", [15, 60])),
            delta_min=list(settings.get_path("features.delta_min", [15])),
            step_min=step_min,
        )

    log.info("Feature tablosu: %d satir x %d kolon", len(frame), frame.shape[1])
    return frame


# --------------------------------------------------------------------------- #
# 5. Denetimli ogrenme matrisi
# --------------------------------------------------------------------------- #
NON_FEATURE = {"ts", "batch_id", "line_id", "product_code", "crew", DOWNTIME_FLAG,
               "lab_prev_all_passed", "dayofweek"}


def select_features(frame: pd.DataFrame, profile: Profile, target: str) -> List[str]:
    """Sizinti korumasi: hedefin kendisi ve turevleri dislanir."""
    spec = profile.targets[target]
    excluded = set(spec.exclude_features) | {target} | NON_FEATURE
    # Diger hedefler de ozellik olamaz (ayni anda olculmezler)
    excluded |= {t for t in profile.targets if t != target}

    # Turetilmis ozellikler girdileri uzerinden sizabilir:
    # water_load = f(uretim_hizi, ...) olsaydi, uretim hizini gizlice geri sokardi.
    for name, definition in derived_for(profile.name).items():
        if set(definition["inputs"]) & excluded:
            excluded.add(name)
            log.debug("Turetilmis ozellik dislandi (girdisi sizdiriyor): %s", name)

    features: List[str] = []
    for col in frame.columns:
        if col in excluded:
            continue
        base = col.split("__")[0]
        if base in excluded:
            # ...ancak GECMIS degerleri serbesttir: nem_lag15 mesru bir ozelliktir
            if "__lag" in col or "__rmean" in col or "__rstd" in col:
                features.append(col)
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        features.append(col)
    return sorted(set(features))


def make_supervised(
    frame: pd.DataFrame,
    profile: Profile,
    target: str,
    drop_downtime: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """(X, y, feature_names) doner. Zaman sirasi korunur."""
    if target not in frame.columns:
        raise KeyError(f"Hedef kolon tabloda yok: {target}")

    work = frame.copy()
    is_classification = profile.targets[target].kind == "classification"

    if drop_downtime and DOWNTIME_FLAG in work.columns:
        # Durus sirasinda proses calismiyor; bu satirlar hem regresyon hedefini
        # bozar hem de kopus modelinde trivial kural uretir.
        work = work[work[DOWNTIME_FLAG] == 0]

    features = select_features(work, profile, target)
    work = work.dropna(subset=[target])

    # Kategorik kodlar feature tablosunda hazir (join_batches); burada UYDURULMAZ.
    # Egitim ve servis ayni kolonlari gormek zorunda.
    X = work[features]

    # Lag'lerden gelen ilk NaN'lar
    valid = X.notna().all(axis=1)
    X, work = X[valid], work[valid]
    y = work[target].astype(int if is_classification else float)

    X = X.assign(ts=work["ts"].values).set_index("ts")
    y.index = X.index
    return X, y, features
