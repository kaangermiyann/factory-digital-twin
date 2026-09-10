# 03 — ML Yaklaşımı

## Problem çerçevesi

Dijital ikizin kalbi bir **vekil (surrogate) model**tir:

```
ŷ = f( kontrol_edilebilir_setpointler , bağlam )
```

- **Kontrol edilebilir**: makine hızı, buhar basıncı, nip yükü, vakum, refiner enerjisi, O₂ enjeksiyonu…
- **Bağlam**: ürün grade'i, hammadde özellikleri, ortam sıcaklık/nem, vardiya, ekipman yaşı

Model bir kez öğrenildikten sonra **milisaniyede** çalışır. Bu, optimizasyonu mümkün kılar:
gerçek fabrikada 10.000 deney yapamayız ama model üzerinde 10.000 senaryo saniyeler sürer.

---

## Hedef değişkenler (4 model)

| # | Hedef | Tip | Neden bu? | Metrik |
|---|---|---|---|---|
| M1 | `sec_total_kwh_t` — özgül enerji tüketimi | Regresyon | **Paranın olduğu yer.** %3 iyileşme = yılda 7 haneli tasarruf | MAE, MAPE, R² |
| M2 | `reel_moisture_pct` (kağıt) / `tap_temp_c` (çelik) | Regresyon | Kalite kısıtı. Optimizasyon bunu spec içinde tutmak zorunda | MAE, spec-ihlal oranı |
| M3 | `production_rate_tph` | Regresyon | Verimlilik kolu; enerjiyle ters yönlü tradeoff | MAE, R² |
| M4 | `break_next_30m` — 30 dk içinde kopuş/duruş | Sınıflandırma | Dengesiz sınıf (~%2–5). Erken uyarı | PR-AUC, recall@precision=0.5 |

> M1 ve M2 birlikte **asıl hikâyeyi** anlatır: enerjiyi düşür ama kaliteyi bozma.
> Tek başına M1 gösterilirse "buharı kapat, enerji sıfır" saçmalığına düşülür.

---

## Neden RandomForest?

| Gerekçe | Açıklama |
|---|---|
| Tabular veride güçlü | 20–200 özellikli proses verisinde deep learning'i **yenmez ama yenilmez** |
| Ön işleme toleransı | Ölçekleme gerektirmez, aykırı değerlere dayanıklı, doğrusal olmayan ilişkileri yakalar |
| Etkileşimleri öğrenir | "Yüksek hız + düşük buhar" birleşimini otomatik yakalar (doğrusal model yakalayamaz) |
| Açıklanabilir | Feature importance + permutation importance + PDP → **proses mühendisi ikna edilir** |
| Hızlı inference | Optimizasyon döngüsünde 10k çağrı < 1 sn |
| Az hiperparametre | Sunum odaklı projede zaman kazandırır |

**Zayıflığı ve karşı önlemi:**
- ❌ Eğitim aralığının dışına **ekstrapole edemez** → optimizasyon aramasını
  `config`'teki emniyet aralığı **∩** eğitim verisinin gördüğü aralık ile sınırlıyoruz
  (`optimize/search.py` içinde `in_domain` kontrolü). Bu, "model 300 m/dk önerdi ama
  makine hiç 300'de çalışmadı" tuzağını engeller.
- ❌ Zaman serisi trendini doğal olarak modellemez → lag/rolling özellikleriyle telafi.

**Karşılaştırma seti (her zaman birlikte eğitilir):**
1. `Naive` — ortalamayı/önceliği tekrarla → **alt sınır**
2. `Ridge` / `Logistic` — doğrusal referans → "problem doğrusal mı?" sorusunu cevaplar
3. `RandomForest` — ana ağaç modeli
4. `HistGradientBoosting` — genelde en iyisi; RF'in ne kadar geride kaldığını gösterir

> Sunumda "RF şu kadar iyi" demek zayıftır. **"Naive 85.7 → Ridge 18.9 → RF 19.5 → GBM 18.8"**
> demek güçlüdür — hem modelin bir şey öğrendiğini kanıtlar hem de dürüsttür.

### Gözlemlenen sonuç ve nasıl anlatılmalı

Sentetik veride ölçülen gerçek davranış:

| Hedef | Doğrusal (ridge/logistic) | Ağaç (RF/GBM) | Yorum |
|---|---|---|---|
| Özgül enerji | R² ≈ 0.95 | R² ≈ 0.95 | **Ağaçlar fark yaratmıyor** — bu rejimde ilişki neredeyse toplamsal |
| Sarıcı nemi | R² ≈ 0.76 | R² ≈ 0.90 | Ağaçlar açık ara önde — eşikli/etkileşimli fizik |
| Duruş riski | zayıf | PR-AUC ≈ 0.73 | Ağaçlar olmadan olmuyor |

**Bu tabloyu gizlemeyin, sunun.** İki nedenle:

1. Enerji modelinin doğrusala yakın çıkması bir başarısızlık değil, bir **bulgu**dur:
   "işletme aralığınızda enerji, kolların toplamı gibi davranıyor; değer model
   karmaşıklığında değil, **kısıtlı optimizasyonda**." Zaten para orada.
2. Ağaçların nem ve duruş riskinde kazanması, onları savunmak için gereken kanıttır.
   Her hedefte ağaç dayatmak yerine, hedef başına en iyi modeli seçmek
   (`train.py` bunu otomatik yapar) daha savunulabilir bir mühendislik duruşudur.

### Model seçimi ≠ sadece doğruluk — optimizasyon-güvenli seçim

Seçilen model yalnızca bir tahminci değil, **optimizasyonun üzerinde arama yaptığı
vekil yüzeydir**. Fark burada kritik:

| | Ağaç modeli | Doğrusal model |
|---|---|---|
| Eğitim desteği dışında | **Doyuma gider** — sabit değer basar | **Sınırsız ekstrapole eder** |
| Çözücü ne bulur? | Uydurulmuş kazanç bulamaz | Katsayıları toplar, sahte kazanç üretir |

Gerçek bir örnek — bu repoda yaşandı: özgül enerji hedefinde ridge MAE 19.25,
GBM 19.36 çıktı (**%0.5 fark, istatistiksel olarak anlamsız**). Ridge seçilince
optimizasyon `SEC 1356 → 1225 kWh/t, 234 ₺/ton, yılda 50 M₺` önerdi. Her değişken
tek tek emniyet kutusunun içindeydi — ama **birleşimleri hiç gözlenmemişti** ve
doğrusal model bunu bilemez. Güven rozeti 🔴 yandı, yani koruma çalıştı; fakat
tavsiyenin kendisi baştan üretilmemeliydi.

Bu yüzden `train.py` bir **eşitlik bozma kuralı** uygular:

> Ağaç modeli en iyi skora `optimizer_safe_tolerance` (varsayılan %3) kadar
> yakınsa, doğrusal model yerine **o** seçilir. Fark eşiği aşıyorsa doğruluk
> feda edilmez.

Ödenen bedel binde birkaç doğruluk; kazanılan, güvenle optimize edilebilen bir
yüzey. Seçim gerekçesi model künyesine not olarak yazılır ve dashboard'da görünür.

> Ek not: doğrusal model yine de seçilirse yaprak-yoğunluğu güven kontrolü o model
> için devre dışı kalır (ridge'in yaprağı yoktur). Sistem güveni diğer ağaç tabanlı
> modellerin yaprak desteğinden okur — bkz. `optimize/search.py::assess_confidence`.

---

## Feature engineering

### Zorunlu: proses ölü zamanı (dead time)
Kurutma grubundaki buhar basıncı değişimi, sarıcıdaki neme **8–15 dakika sonra** yansır.
Aynı andaki değerlerle model kurmak fiziksel olarak yanlıştır.

```python
# features/build.py
LAGS_MIN = [0, 5, 10, 15, 30]      # her kontrol değişkeni için
ROLLING  = [15, 60]                # ortalama + std (proses kararlılığı)
DELTAS   = [15]                    # değişim hızı → operatör müdahalesinin izi
```
`rolling_std` özellikle değerlidir: **kararsız proses = yüksek enerji + düşük kalite.**

### Takvim
saat (sin/cos), hafta günü, vardiya no, ekip kodu.

### Batch/kalite join
Telemetri (dakikalık) ← `batch_id` ile ← üretim (batch) ← ← lab sonucu (seyrek).
`merge_asof` ile en yakın geçmiş kayda bağlanır — **geleceğe bakmadan**.

### Leakage (sızıntı) koruması — en sık yapılan hata
`config/profile.<ad>.yaml` içinde her hedef için `exclude_features` listesi vardır:
- `sec_total_kwh_t` modelinde **enerji sayacı okumaları kullanılamaz** (hedefin kendisi)
- `reel_moisture_pct` modelinde **nem scanner'ının lag'siz değeri kullanılamaz**
- `break_next_30m` modelinde **duruşun kendisinden türeyen tag'ler kullanılamaz**
  (duruşta hız 0'a düşer → model "hız 0 ise kopuş var" der, işe yaramaz bir model)

> Bir model şüphe uyandıracak kadar iyiyse (R² > 0.98), neredeyse kesinlikle sızıntı vardır.
> `models/evaluate.py` bunu otomatik uyarı olarak basar.

#### En sinsi tür: parti seviyesinde toplanan değerler

`exclude_features` listesi *bilinen* sızıntıları kapatır. Asıl tehlike, **kolon adı
masum görünen** sızıntılardır. Bu projede aynı sınıftan **üç tane** yakalandı:

| Kolon | Neden sızıntı |
|---|---|
| `lab_moisture` | Numune partinin **sonunda** alınır; parti ortasındaki satır geleceği görür |
| `batch_scrap_ratio` | Iskarta parti kapanınca kesinleşir — ve **kopuş, ıskartayı üreten şeydir**. Model "ıskarta yüksekse kopuş gelecek" öğrenir; holdout parlar, saha çuvallar |
| `water_load_tph` | Türetilmiş görünür ama içinde `production_rate_tph` (bir hedef) vardır |

Üçü de holdout skorunu **yükseltiyordu** — yani "model iyileşti" diye sevinilecek
türden. Kural şudur:

> **Parti seviyesinde toplanan her şey (ıskarta, üretim miktarı, lab sonucu, spec
> uyumu) ancak parti kapandığında bilinir.** Özellik olarak yalnızca **bir önceki**
> partininki kullanılabilir — operatörün tahmin anında elinde olan da odur.

Tek tek avlamak yerine `tests/test_pipeline.py::test_no_batch_level_aggregate_leaks`
bir **yapısal değişmez** uygular: parti-bazlı bir kolon `prev_` öneki olmadan
özellik tablosuna girerse test düşer. Yeni eklenen özellikler otomatik denetlenir.

### Eğitim ↔ servis pariteti — sessiz kayıpların olduğu yer

Model, eğitimde gördüğü kolonların **birebir aynısını** serviste de görmek zorunda.
Bu iki noktada sessizce bozulur ve hiçbir hata mesajı vermez:

| Tuzak | Ne olur | Çözüm |
|---|---|---|
| Kategorik kodlama eğitim tarafında `cat.codes` ile üretilir | Kolon canlıda hiç doğmaz, medyanla dolar → **özellik ölü kalır** | Kodlama `features/build.py`'de, **sabit** eşlemeyle üretilir (`crew_id`, `product_code_id`) |
| Kodlama pencereye göre değişir | 6 saatlik pencerede "A ekibi" 0, 90 günlükte 2 olur → **yanlış tahmin** | Eşleme tüm `batches` tablosundan kurulur, pencereden bağımsız |
| Canlı satırda NaN | Ağaç modelleri sindirir, **doğrusal modeller patlar** — yani hata ancak ridge seçildiğinde ve tam demo sırasında çıkar | `ModelBundle.align()` eğitim medyanıyla doldurur; `imputed_features()` neyin dolduğunu raporlar |

Son satır önemli: eğitimde NaN'lı satırlar atılır, canlıda atma lüksü yoktur
(lag'i dolmamış kolon, gelmemiş lab sonucu, anlık sensör boşluğu). Doldurma tek
yerde yapılır ve **şeffaftır** — `/kpis` kaç özelliğin doldurulduğunu döner.
Sayı büyürse bu bir model sorunu değil, bir **veri sorunu** işaretidir.

---

## Validasyon — zaman serisinde `train_test_split` YASAK

Rastgele bölme, geleceği geçmişe sızdırır ve modeli **olduğundan iyi** gösterir.

```
|--- train ---|- val -|                          fold 1
|------- train -------|- val -|                  fold 2
|----------- train -----------|- val -|          fold 3
                                       ↑ gap: proses ölü zamanı kadar boşluk
|-------------- TRAIN --------------|--- HOLDOUT (son %20, hiç dokunulmaz) ---|
```

- `TimeSeriesSplit` + hedefin en uzun lag'i kadar **gap**
- Son %20 zaman dilimi **holdout** — sadece final raporda bir kez kullanılır
- Grade değişimi olan periyotlar rapora ayrı kırılım olarak konur (en zor senaryo)

---

## Değerlendirme çıktıları (sunum malzemesi)

`models/evaluate.py` şunları üretir:

1. **Model karşılaştırma tablosu** (naive/ridge/RF/GBM × MAE/RMSE/MAPE/R²)
2. **Gerçek vs Tahmin zaman serisi** — dashboard'daki ana grafik
3. **Artık (residual) analizi** — hata rejime göre değişiyor mu? (grade bazlı kırılım)
4. **Permutation importance** — sentetik veride model şunları en üste koydu:
   `water_load_index` (kurutmaya giren su yükü), `ambient_humidity_pct` (ortam nemi),
   `press_dryness_pct`, `steam_total_bar`. Yani model **kurutma fiziğini yeniden keşfetti**.
   Proses mühendisi bu listeyi onaylıyorsa güven kazanılır; onaylamıyorsa veri veya
   etiketleme hatalıdır. Her iki durumda da bilgi kazanılır
5. **Partial dependence (PDP)** — "hız 900'ü geçince enerji tüketimi tırmanıyor" → **aksiyon alınabilir bilgi**
6. **Kalibrasyon eğrisi** (M4 için) — "risk %80 dediğimizde gerçekten %80 mi?"

---

## Modelden aksiyona: nasıl "optimizasyon" oluyor?

Sadece tahmin, para kazandırmaz. Zincir şudur:

```
1. Model öğrenir:  enerji = f(hız, buhar, nip, vakum, nem_hedefi, ortam)
2. Anlık bağlam sabitlenir (ürün grade'i, hammadde, hava — bunlar değişmez)
3. Kontrol edilebilir setpointler emniyet aralığında taranır
4. Kısıtları sağlayan en düşük maliyetli kombinasyon seçilir
5. Operatöre gösterilir: "Buhar G3: 3.8 → 3.5 bar, Nip: 320 → 345 kN/m
                          Beklenen: -4.2% enerji, nem 5.4% (spec içi)"
6. Uygulanır → sonuç ölçülür → veri setine geri döner (kapalı öğrenme döngüsü)
```

Detay: `docs/04-optimizasyon.md`

---

## Model yönetimi

- Her eğitim, `data/models/<profil>/<hedef>/<sürüm>/` altına yazılır:
  `model.joblib` (model + yaprak destek tabloları) ve `meta.json`
- `meta.json`: özellik listesi, holdout metrikleri, model karşılaştırma tablosu,
  permutation importance, eğitim veri aralığı, satır sayısı ve her özelliğin
  min/p01/medyan/p99/max değerleri → optimizasyonun ekstrapolasyon kilidi bunu kullanır
- `latest.txt` en son sürümü işaret eder; eski sürümler diskte kalır (geri dönülebilir)
- **Drift takibi**: canlı `y_pred` vs sonradan gelen `y_true` üzerinden haftalık MAE.
  Eğitim MAE'sinin 1.5 katını aşarsa → yeniden eğitim tetiklenir.
- Yeniden eğitim ritmi: aylık (proses değişikliği/bakım sonrası elle de tetiklenebilir)
