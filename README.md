# Fabrika Dijital İkizi — Enerji ve Kalite Optimizasyonu

> **Durum:** Elimizde müşteri verisi yok. Bu repo, veri gelene kadar beklemek yerine
> **tüm hattı bugün çalışır halde** kurar: fizik-temelli bir simülatör canonical şemayı
> üretir, pipeline gerçek veriymiş gibi çalışır. Müşteri verisi geldiğinde
> **sadece `src/twin/ingest/` adaptörü** değişir.

Kağıt makinesi ve demir-çelik (EAF) için iki proses profili içerir. Yeni bir fabrika
eklemek = yeni bir `config/profile.<ad>.yaml` yazmak.

---

## 60 saniyede çalıştır

```bash
make setup      # bağımlılıklar
make demo       # 90 günlük veri üret + 4 model eğit + son 48 saati skorla
make dash       # http://localhost:8501
```

Süreler (8 çekirdekli bir dizüstünde ölçüldü): veri üretimi **~15 sn**,
feature tablosu **~4 sn**, eğitim **~20 dk** (4 hedef × 4 model × 4 katlı CV).
Sadece ekranı görmek için `make train-fast` (~4 dk) yeter — ama o modun
metrikleri rapora konmaz, eğitim çıktısı bunu ayrıca uyarır.

Docker/Elasticsearch gerekmez — varsayılan depo lokal Parquet.
Elasticsearch ile çalıştırmak için:

```bash
make elastic-up
export TWIN_STORAGE__BACKEND=elastic
make demo && make dash
```

---

## Ne yapıyor?

```
Fabrika verisi ──▶ Canonical şema ──▶ Feature ──▶ Model (RF/GBM) ──▶ Optimizasyon ──▶ Dashboard
   (yok, şimdilik           ES / MySQL /      lag, rolling,     "SEC = f(setpoint,   kısıtlı arama    canlı vs tahmin
    simülatör)              Parquet           batch join        bağlam)              → ₺/ton          + ₺/yıl kazanç
```

**Zincir tek cümlede:** Model, "özgül enerji tüketimi = f(operatörün çevirebildiği kollar,
değiştiremediği bağlam)" ilişkisini veriden öğrenir. Sonra bağlamı sabitleyip kolları
emniyet sınırları içinde tarar ve **kaliteyi bozmadan en ucuz** kombinasyonu bulur.

### Dört model

| Hedef | Tip | Neden |
|---|---|---|
| `sec_total_kwh_t` | Regresyon | **Paranın olduğu yer.** Ana optimizasyon hedefi |
| `reel_moisture_pct` / `tap_temp_c` | Regresyon | Kalite kısıtı — optimizasyon bunu spec içinde tutmak zorunda |
| `production_rate_tph` | Regresyon | Verimlilik kolu; enerjiyle ters yönlü takas |
| `break_next_30m` | Sınıflandırma | 30 dk içinde duruş/kopuş riski — erken uyarı |

İlk ikisi **birlikte** asıl hikâyeyi anlatır: *enerjiyi düşür ama kaliteyi bozma.*
Tek başına birincisi gösterilirse "buharı kapat, enerji sıfır" saçmalığına düşülür.

---

## Örnek sonuçlar (sentetik veri, 90 gün, kağıt profili)

`make train` çıktısı — 25.920 satır, 171 özellik, 3 katlı zaman serisi CV, son %20 holdout:

```
sec_total_kwh_t  (özgül enerji, kWh/t)          reel_moisture_pct  (sarıcı nemi, %)
        model     mae    rmse  mape     r2              model    mae   rmse   mape     r2
 persistence*   15.11   33.49  1.12  0.905   gradient_boosting  0.413  0.598   8.15  0.875  ←
      ▸ GBM     19.11   24.58  1.40  0.949 ←             ridge  0.486  0.910   9.64  0.709
        ridge   19.24   24.75  1.41  0.948       random_forest  0.578  0.794  11.57  0.779
random_forest   19.50   25.67  1.42  0.944               naive  0.953  1.688  18.82  0.000
        naive   85.70  108.55  6.13 -0.001        persistence*  1.279  2.080  24.64 -0.519
  naive'e göre %78 iyileşme                        naive'e göre %57 iyileşme

break_next_30m  (30 dk içinde duruş)             production_rate_tph  (t/h)
            model  pr_auc  roc_auc  recall@p50            model     mae     r2
     persistence*   0.776    0.924       0.871            ridge  0.0410  0.9998  ←
         logistic   0.515    0.849       0.328    random_forest  0.0464  0.9993
▸ gradient_boosting 0.509    0.845       0.437  ←  gradient_b..  0.0510  0.9994
    random_forest   0.428    0.791       0.215            naive  3.2725 -0.0067
            naive   0.147    0.500       0.000     (fiziksel olarak deterministik:
  naive'e göre %246 iyileşme                        üretim = hız × en × gramaj)
```

**Nasıl okunmalı — üç nokta:**

**1. Taban çizgisi olmadan sayı anlamsızdır.** "RF iyi" demek zayıf; "naive 85.7 →
19.1" demek modelin gerçekten bir şey öğrendiğini kanıtlar.

**2. `persistence*` = "son ölçülen değeri tekrarla".** Enerjide MAE 15.1 ile
*herkesi geçiyor* — ama **karşı-olgusal soru soramaz**: "buharı düşürsem ne olur?"
sorusuna cevabı yoktur, optimize edilemez. Sınıflandırmada etiket
otokorelasyonunu ölçer: devam eden alarmı sürdürür, **ilk alarmı asla veremez**.
Bu yüzden raporlanır ama seçilemez.

**3. Kopuş modelinin PR-AUC'si 0.73 değil 0.51 — ve bu düzeltilmiş sayı.**
İlk eğitimde 0.727 çıkmıştı; en önemli özelliği `batch_scrap_ratio` idi
(önem 0.52, sonrakinin 20 katı). Ama ıskarta parti kapanınca kesinleşir ve
**kopuş, ıskartayı üreten şeydir** — model olacak duruşun izini görüyordu.
Sızıntı kapatılınca skor %30 düştü. Aşağı giden sayı, doğru olan.

Şimdi modelin baktığı yer fiziksel: `fiber_flux` (hız × gramaj), ürün grade'i,
`water_load_index` (kurutmaya giren su), refiner enerjisi.

---

## Neden ağaç modeli tercih ediliyor? — optimizasyon-güvenli seçim

Bu repoda gerçekten yaşandı: enerji hedefinde ridge seçilince optimizasyon
`SEC 1356 → 1225 kWh/t · 234 ₺/ton · yılda 50 M₺` önerdi. Her setpoint tek tek
emniyet aralığının içindeydi — ama **birleşimleri hiç gözlenmemişti**.

| | Ağaç modeli | Doğrusal model |
|---|---|---|
| Eğitim desteği dışında | **Doyuma gider** | **Sınırsız ekstrapole eder** |
| Çözücü ne bulur? | Uydurulmuş kazanç bulamaz | Katsayıları toplar, sahte kazanç üretir |

Güven rozeti 🔴 yandı, yani koruma çalıştı — fakat tavsiyenin kendisi baştan
üretilmemeliydi. Bu yüzden `train.py` bir eşitlik bozma kuralı uygular:

> Ağaç modeli en iyi skora **%3** kadar yakınsa, doğrusal model yerine **o** seçilir.
> Fark eşiği aşıyorsa doğruluk feda edilmez.

Enerjide fark %0.5 → GBM seçildi. Üretim hızında fark %4.4 → ridge korundu.
Karar, model künyesine not olarak yazılır ve dashboard'da görünür.

---

## Dashboard

**Ana ekran tek bir soruya cevap verir: model, sahadan ölçüleni tutturabiliyor mu?**

```bash
make dash   # http://localhost:8501
```

| | |
|---|---|
| **📈 Gerçek vs Tahmin** (ana ekran) | Zaman serisi karşılaştırması, hata metrikleri, saçılım, drift takibi |
| 🎯 Optimizasyon | What-if slider'ları, kısıtlı optimizasyon, ₺/yıl kazanç |
| 🧠 Model Detayı | naive/doğrusal/ağaç karşılaştırması, permutation importance, PDP |
| 🔍 Veri Kalitesi | Sensör doluluk/donukluk/aralık taraması — müşteri verisi geldiğinde ilk bakılacak yer |

Model karşılaştırma tabloları ve özellik önemleri **ana ekranda değil** — alt
sayfalarda. Günlük kullanımda bakılacak tek şey ölçülen ile tahminin örtüşüp
örtüşmediğidir.

### En kritik tasarım kararı: hangi veri gösteriliyor?

Grafik **varsayılan olarak holdout döneminden** çizilir — modelin hiç görmediği veri.

> Modelin eğitildiği dönemde iyi uyum göstermesi **beklenen** bir şeydir ve hiçbir
> şey kanıtlamaz. Oradaki güzel grafiği müşteriye göstermek yanıltıcıdır.

Ekranın üstünde hangi dönemin çizildiği yazar; eğitim dönemini de görmek isteyen
kenar çubuğundan açar ve ⚠️ uyarısını görür. Model paketleri `holdout_start`
alanını taşır, sınır tahmin edilmez.

**Ana ekranda ne var:**

- **Hata kartları** — MAE, MAPE ve **sapma (bias)**. Sapma sıfıra yakınsa model
  sistematik olarak yüksek/düşük tahmin etmiyor demektir; MAE kadar önemlidir ve
  genelde atlanır.
- **Zaman serisi** — ölçülen (düz) vs tahmin (kesikli), spec bandı ve duruş
  dönemleri gri şeritle. Grafikteki kopukluk model hatası değil, makine duruşudur.
- **Hata zaman içinde** — 6 saatlik ortalama, eğitimdeki MAE ve **1,5× yeniden
  eğitim eşiği** çizgileriyle. Drift buradan görülür.
- **Saçılım** — ölçülen vs tahmin, y=x çizgisine yakınlık.
- **Spec dışı dönemler** — ölçülen %X, model %Y, **yakalanan %Z**. Ortalamayı
  tutturmak kolaydır; uç durumları yakalamak zordur, bir kalite modelinin sınavı budur.
- **Duruş riski** ayrı okunur: olasılık tahminidir, düz çizgiyle kıyaslanamaz.
  Doğru soru *"gerçek duruşlardan önce risk yükseliyor mu?"* — gerçekleşen duruşlar
  ✕ ile işaretlenir.

Görülmemiş 18 günde ölçülen hata, eğitim holdout tahminiyle örtüşüyor:

```
hedef                     n      MAE   MAPE%    sapma    eğitimdeki MAE
production_rate_tph   7.853    0.042   0.17%   -0.001         0.041
reel_moisture_pct     7.853    0.395   7.88%   +0.010         0.413
sec_total_kwh_t       7.853   20.094   1.45%   -2.249        19.110
```

---

## Mimarideki dört karar

**1. Depolama soyutlaması.**
`Repository` arayüzünün arkasında Elasticsearch / MySQL / Parquet. Zaman serisi ES'e,
master data MySQL'e gider; ama üst katmanlar bunu bilmez. Müşteri "sadece MySQL"
derse tek satır değişir. → [`docs/02-mimari.md`](docs/02-mimari.md)

**2. `role` alanı: setpoint / measurement / context.**
Projenin en kritik metadata'sı. Sıcaklık *ölçümünü* "optimize etmek" anlamsızdır;
buhar basıncı *set noktasını* optimize etmek anlamlıdır. Bu ayrım yoksa optimizasyon
yapılamaz — sadece tahmin ve erken uyarı teslim edilir.

**3. Sızıntı koruması, opsiyonel değil — üç katman.**
(a) `profile.yaml` içindeki `exclude_features` listeleri hedefin bileşenlerini dışarıda
tutar. (b) Eğitim, R² > 0.98 gördüğünde **kendiliğinden uyarı basar**. (c) Bir
**yapısal değişmez**: parti seviyesinde toplanan hiçbir değer (ıskarta, lab sonucu,
üretim miktarı) `prev_` öneki olmadan özellik tablosuna giremez.

(c) katmanı sonradan eklendi — çünkü bu repoda aynı sınıftan **üç sızıntı** yakalandı
ve üçünün de kolon adı masum görünüyordu. En sinsisi: kopuş modelinin en önemli
özelliği `batch_scrap_ratio` çıkmıştı (önem 0.52, sonrakinin 20 katı) — oysa ıskartayı
üreten şey kopuşun kendisiydi.

**4. Ekstrapolasyon kilidi — iki katmanlı.**
Arama uzayı üç kaynağın **kesişimi**: emniyet aralığı ∩ değişim hızı limiti ∩
eğitim zarfı (değişkenin ham değeri *ve* lag/rolling kopyaları üzerinden).
Ama kutu kontrolü yetmez — her değişken tek tek aralığında olup **birleşimleri
hiç gözlenmemiş** olabilir. Bu yüzden ikinci katman: model paketinde saklanan
eğitim alt örneğine **normalize maks-norm komşuluk sayımı**. *"Bu çalışma
noktasına benzer bir noktayı eğitimde kaç kez gördüm?"* Ölçüm yapılamıyorsa
güven en fazla 🟡 raporlanır — ölçülmeyeni "yüksek" demek, hiç raporlamamaktan
kötüdür.

---

## Simülatör neden fizik-temelli?

Rastgele sayı üretmek pipeline'ı test eder ama **optimizasyonu test etmez**.
Bu simülatörde bilinçli olarak gerçek bir tasarruf boşluğu bırakılmıştır:

- Operatör **spec'in ortasını değil kuru tarafını** hedefler ("spec dışına çıkmayayım")
  → fazladan buhar
- **Ucuz mekanik su alma** (pres nip yükü, vakum) alışkanlıktan sabit bırakılır,
  telafi **pahalı termal enerjiyle** yapılır → asıl tasarruf kalemi
- Üç vardiya ekibinin farklı bias'ı var → aynı üründe farklı enerji tüketimi
- Gözlenmeyen (latent) proses durumu var → modelin açıklayamayacağı artık varyans
  **olması gerektiği için** var. Yoksa model gerçekte tutturulamayacak kadar iyi çıkar.

Denklemler modele verilmez; model bunları veriden öğrenmek zorundadır.

---

## Dokümanlar

| Doküman | İçerik |
|---|---|
| [`docs/01-veri-sozlesmesi.md`](docs/01-veri-sozlesmesi.md) | **Müşteriye birebir gönderilebilir.** Hangi veri, hangi formatta, neden |
| [`docs/02-mimari.md`](docs/02-mimari.md) | Katmanlar, ES vs MySQL kararı, canonical şema |
| [`docs/03-ml-yaklasimi.md`](docs/03-ml-yaklasimi.md) | Neden RandomForest, ölü zaman, validasyon, sızıntı |
| [`docs/04-optimizasyon.md`](docs/04-optimizasyon.md) | Problem formülasyonu, çözücü seçimi, kazanç doğrulama protokolü |
| [`docs/05-yol-haritasi.md`](docs/05-yol-haritasi.md) | Fazlar, riskler, başarı kriterleri, **45 dk sunum akışı** |

---

## Komutlar

```bash
make sim         # sentetik geçmiş üret (90 gün)
make live        # canlı akış — dashboard hareket etsin diye (ayrı terminal)
make train       # model eğit (~20 dk)
make train-fast  # hızlı eğitim — sadece geliştirme için
make score       # son 48 saati skorla → gerçek vs tahmin grafiği dolsun
make recipe      # altın reçete tablosu (ürün × ortam × tarife) → CSV + Markdown
make api         # FastAPI  → http://localhost:8000/docs
make dash        # Streamlit → http://localhost:8501
make test        # testler
```

Müşteri verisi yüklemek:

```bash
python -m twin.ingest.loader telemetry "data/raw/historian_2026*.csv" --wide --tz Europe/Istanbul
python -m twin.ingest.loader batches   data/raw/production.csv
```

Yükleyici, veri kalite raporu basar (aralık dışı, donuk sensör, zaman boşluğu,
hiç değişmemiş setpoint). Kapıdan geçemeyen kayıtlar **sessizce düşürülmez**,
`quarantine` deposuna yazılır ve sayılır.

---

## Proje yapısı

```
config/
  settings.yaml            depo, feature, eğitim, optimizasyon, fiyat ayarları
  profile.paper.yaml       kağıt makinesi proses sözlüğü ← YENİ FABRİKA = YENİ DOSYA
  profile.steel.yaml       EAF çelikhane proses sözlüğü
src/twin/
  config.py schema.py      canonical şema ve profil yükleme
  storage/                 Repository arayüzü + ES / MySQL / Parquet + ES mapping'leri
  simulator/               fizik-temelli kağıt makinesi ve EAF
  ingest/                  ← MÜŞTERİ VERİSİ GELDİĞİNDE DEĞİŞEN TEK YER
      loader.py            geniş/uzun CSV → canonical, birim + saat dilimi
      validate.py          veri kalite kapısı → rapor + karantina
  features/build.py        lag/rolling/join, türetilmiş özellikler, sızıntı koruması
  models/                  eğitim, metrikler, açıklanabilirlik, model kayıt defteri
  optimize/                amaç fonksiyonu, kısıtlı arama, altın reçete üreticisi
  serving/                 FastAPI + canlı skorlama döngüsü
  dashboard/               app.py (Gerçek vs Tahmin) + pages/ + theme.py + data.py
scripts/                   müşteriye gönderilecek sensor_registry şablonu
docs/                      1–5 numaralı dokümanlar
tests/                     52 test — sızıntı, zaman sırası, ekstrapolasyon kilidi,
                           eğitim↔servis paritesi, dashboard duman testi
```

---

## Neyin doğrulandığı

| Katman | Durum |
|---|---|
| Simülatör → Parquet → feature → 4 model → optimizasyon → skorlama | **Uçtan uca çalıştırıldı** (90 gün, 2,76M telemetri satırı) |
| FastAPI — 9 uç nokta | Hepsi `TestClient` ile çağrıldı, hata yolları dahil |
| Streamlit dashboard | 4 sayfanın hepsi `AppTest` ile koşturuldu: 0 istisna, deprecation yok |
| Çelik (EAF) profili | Simülasyon → feature → denetimli matris adımlarına kadar çalıştırıldı |
| Müşteri CSV ingest'i | Bozuk veri enjekte edilerek test edildi (birim hatası, `-9999`, bilinmeyen tag, saat dilimi) |
| Testler | **52 test**, hepsi geçiyor (`make test`) — atlanan test yok |
| Elasticsearch adaptörü | Index şablonları, aylık bölme ve sorgu üretimi doğrulandı; **canlı bir kümeye karşı çalıştırılmadı** |
| MySQL adaptörü | Şema ve sorgu üretimi yazıldı; **canlı bir sunucuya karşı çalıştırılmadı** |
| Docker Compose | Yazıldı, ayağa kaldırılmadı |

Depolama adaptörleri aynı `Repository` arayüzünü uygular ve Parquet üzerinden
test edilmiş kod yolunu paylaşır; yine de ES/MySQL'e geçmeden önce
`make elastic-up && make demo` ile bir tur atılmalıdır.

---

## Altın reçete tablosu

`make recipe` → ürün × ortam × tarife segmentleri için optimum setpoint tablosu
(CSV + basılabilir Markdown). İkiz kapalıyken bile operatörün elinde kalan çıktı;
çoğu müşteri için en somut teslimat budur.

```
 urun ortam tarife    n  guven  kazanc_TL_ton
TL120 sicak  puant 1956 medium         169.02
TL120 sicak gunduz 4634 medium         110.32
 FL45 sicak gunduz 1984 medium          71.91
...
UYARI: 15/15 segment 'high' guven seviyesinde DEGIL.
```

Tablo **kendi uyarısını taşır**: optimize edilen nokta tanımı gereği geçmişte sık
gözlenmemiş bir noktadır, model orada iyimserdir. Sayılar **üst sınır** olarak
okunur, taahhüt olarak değil. Sözleşmeye yazılacak hedef `docs/05`'te: doğrulanmış
enerji iyileşmesi **%2** — bilerek mütevazı.

---

## Sınırlar (baştan söylenmeli)

- İkiz **salt okunurdur**. PLC/DCS'e yazmaz. Çıktı operatöre tavsiye olarak gösterilir.
  Kapalı döngü kontrol ayrı bir proje ve ayrı bir emniyet onayıdır.
- Buradaki metrikler **sentetik veriden** gelir ve gerçek veriden daha iyimserdir.
  Gerçek fabrikada beklenen: SEC modeli MAPE < %5, doğrulanmış enerji tasarrufu ≥ %2.
- Model, eğitimde görülmemiş rejimde tavsiye vermez — vermeye kalkarsa
  🔴 rozetiyle işaretlenir.
- Hiç değişmemiş bir setpoint'in etkisi öğrenilemez. Veri kalite raporu bunu
  ayrıca uyarır; çözümü kontrollü deney tasarımıdır (DOE).
