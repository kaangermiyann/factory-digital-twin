# 01 — Veri Sözleşmesi (Data Contract)

> Bu doküman müşteriye **birebir gönderilebilir**. "Bize ne verebilirsiniz?" sorusunun
> cevabı değil; "bize şunları verin, karşılığında şunu üretiriz" taahhüdüdür.

Elimizde bugün **sıfır veri** var. Projenin en büyük riski model değil, veri riskidir.
Bu yüzden veriyi 3 kovaya ayırıyoruz:

| Kova | Anlamı | Yoksa ne olur? |
|------|--------|----------------|
| **P0 — Zorunlu** | Bu olmadan proje çalışmaz | Proje durur |
| **P1 — Yüksek Değer** | Optimizasyon kalitesini belirler | Model çalışır ama tavsiye üretemez |
| **P2 — Bonus** | Doğruluğu artırır | Kayıpsız devam edilir |

---

## 0. Her veri seti için istenen minimum metadata

Müşteriden gelen **her** dosya/tablo/endpoint için şunlar sorulmalı:

- **Zaman damgası**: UTC mi lokal mi? Yaz saati uygulanıyor mu? Kolon tipi? (`2026-01-04T09:15:00Z` bekliyoruz)
- **Örnekleme sıklığı**: 1 sn / 10 sn / 1 dk / olay-bazlı?
- **Deadband / kompresyon**: Historian (PI, Aspen IP.21, Wonderware) veriyi sıkıştırıyor mu? Sıkıştırıyorsa değerler "değişince mi" yazılıyor? (→ forward-fill gerekir)
- **Eksik değer kodlaması**: `NULL`, `-9999`, `0`, `Bad`, `NaN`?
- **Birim**: bar mı kPa mı? °C mi °F mi? kg/t mi kg/h mi? **Birim tablosu olmadan veri değersizdir.**
- **Tag adı → anlam eşlemesi**: `10FIC0234.PV` ne demek? Bu eşleme `config/profile.<ad>.yaml`
  içindeki her değişkenin `tag:` alanında tutulur — proses sözlüğü tek kaynaktan gelir.
- **Geçmiş derinliği**: **En az 6 ay, ideal 12–18 ay.** Sebep: mevsimsel etki (ortam nemi/sıcaklığı kurutma enerjisini doğrudan değiştirir) ve ürün/grade çeşitliliği ancak bu sürede görülür.
- **Kim sahibi**: veriyi kim çıkaracak, kime soracağız, hangi sistemden?

---

## 1. Master Data / Statik Veri  — **P0**

İlişkisel, yavaş değişen. **MySQL**'de tutulur.

### 1.1 Varlık hiyerarşisi (`asset`)
```
plant → area → line → machine → sensor
Örn: Kağıt Fab. → Kurutma → PM2 → Dryer Group 3 → 10TIC0341
```
| Alan | Örnek |
|---|---|
| asset_id | `PM2.DRY.G3` |
| parent_id | `PM2.DRY` |
| type | line / machine / group |
| nominal_capacity | 320 t/gün |
| rated_power_kw | 850 |
| commissioning_date | 2014-06-01 |

### 1.2 Sensör kataloğu (`sensor_registry`) — **projenin en kritik tablosu**
| Alan | Neden gerekli |
|---|---|
| `tag` | `10FIC0234.PV` |
| `description` | "Hamur kasası konsantrasyonu" |
| `asset_id` | Hangi makineye ait |
| `unit` | `%`, `bar`, `m/min`, `kWh` |
| `role` | **`setpoint` / `measurement` / `context`** |
| `low_limit`, `high_limit` | Fiziksel geçerli aralık (veri temizliği için) |
| `op_low`, `op_high` | **Operasyonel emniyet aralığı** — optimizasyon bu sınırlar içinde arar |
| `max_rate_of_change` | "Buhar basıncı 5 dk'da en fazla 0.3 bar değişebilir" |
| `sampling_period_s` | 1 / 10 / 60 |

> ⚠️ **`role` kolonu olmadan optimizasyon yapılamaz.**
> Model, "hangi değişkeni operatör gerçekten değiştirebilir?" bilgisini buradan alır.
> Sıcaklık ölçümünü "optimize etmek" anlamsızdır; buhar basıncı **set noktasını** optimize etmek anlamlıdır.

### 1.3 Ürün / reçete kataloğu (`product`)
Kağıt: `grade_code`, hedef gramaj (g/m²), hedef nem (%), en (m), selüloz reçetesi, spec min/max.
Demir-Çelik: `grade_code`, hedef kimyasal analiz (C, Mn, Si, S, P aralıkları), hedef mekanik özellik (akma/çekme dayanımı), kalınlık/genişlik.

### 1.4 Vardiya takvimi (`shift_calendar`)
Vardiya no, başlangıç/bitiş, ekip kodu.
> Operatör ekibi, enerji tüketiminde çoğu fabrikada **%3–8 fark** yaratır. Bu tek başına sunum malzemesidir.

### 1.5 Enerji tarifesi (`energy_tariff`) — **P1**
Saat bazlı elektrik fiyatı (gündüz/puant/gece), doğalgaz birim fiyatı, buhar maliyeti (₺/ton).
> Optimizasyonun çıktısı "kWh" değil **"₺"** olmalı. Yönetim kWh'a değil paraya bakar.

---

## 2. Proses Telemetrisi (Zaman Serisi) — **P0**

Historian/SCADA/OPC-UA'dan. **Elasticsearch**'te tutulur.

**Format (canonical):**
```json
{"ts":"2026-01-04T09:15:00Z","asset_id":"PM2.DRY.G3","tag":"10TIC0341.PV",
 "value":128.4,"unit":"C","role":"measurement","quality":100}
```
Müşteri genelde **geniş (wide) CSV** verir — bunu biz `ingest` katmanında canonical'a çeviririz:
```
timestamp, 10FIC0234.PV, 10TIC0341.PV, 10SIC0102.SP, ...
```
Her ikisi de kabul edilir.

**İstenen sıklık:** 1 dakika yeterli. 1 saniye lüks (ve 1 yıl × 300 tag × 1sn ≈ 9.5 milyar satır — ilk fazda gereksiz).
İlk teslimatta: **1 dk ortalama + 1 dk min/max** iste. Min/max, kompresyonda kaybolan salınımı geri getirir.

### 2.1 Kağıt Fabrikası — istenecek tag listesi

| Bölüm | Tag | Rol | Neden |
|---|---|---|---|
| Hamur hazırlama | Refiner özgül enerji (kWh/t) | setpoint | Dayanım ↔ enerji tradeoff'unun kalbi |
| | Freeness / CSF (mL) | context | Hammadde değişkenliği |
| | Selüloz oranları (%) | context | Reçete |
| Hamur kasası | Konsantrasyon (%) | setpoint | Formasyon |
| | Basınç (kPa), jet/wire oranı | setpoint | |
| Elek (wire) | Makine hızı (m/min) | **setpoint** | Ana verim kolu |
| | Vakum seviyeleri (kPa) | setpoint | Mekanik su alma (ucuz) |
| | Kuru madde çıkışı (%) | measurement | |
| Pres | Nip yükü (kN/m) | setpoint | Mekanik su alma (ucuz) |
| | Pres sonrası kuruluk (%) | measurement | **Kurutmaya giren su miktarı** |
| Kurutma | Grup bazlı buhar basıncı (bar) × 4–6 | **setpoint** | Termal enerjinin %70–80'i |
| | Silindir yüzey sıcaklığı (°C) | measurement | |
| | Hood besleme sıcaklık/nem | setpoint | |
| | Kondens debisi | measurement | Gerçek buhar tüketimi |
| Sarıcı (reel) | **Nem (%)** — online scanner | measurement | **Ana kalite hedefi** |
| | **Gramaj (g/m²)** — online scanner | measurement | Ana kalite hedefi |
| | Kalınlık/caliper | measurement | |
| Ortam | Salon sıcaklığı / nemi | **context** | Mevsimsellik. Çoğu fabrika ölçmez → dış hava istasyonu verisi kullanılır |
| Kimyasal | Retansiyon yardımcısı (kg/t), nişasta, tutkal | setpoint | |

### 2.2 Demir-Çelik Fabrikası — istenecek tag listesi

| Bölüm | Tag | Rol | Neden |
|---|---|---|---|
| EAF (Ark Ocağı) | Hurda şarj reçetesi (sepet ağırlıkları, kalite) | context | Girdi değişkenliği |
| | Aktif güç (MW), power-on süresi (dk) | setpoint | Enerjinin %60'ı |
| | Elektrot akım/gerilim, tap kademesi | setpoint | |
| | Oksijen (Nm³/heat), karbon enjeksiyonu (kg) | setpoint | Kimyasal enerji ↔ elektrik ikamesi |
| | Kireç / dolomit (kg) | setpoint | Cüruf kimyası |
| | Döküm (tap) sıcaklığı (°C) | measurement | **Aşırı ısıtma = boşa enerji** |
| | Heat başı elektrik (kWh/t) | measurement | Ana hedef |
| | Tap-to-tap süresi (dk) | measurement | Ana verimlilik hedefi |
| Pota ocağı (LF) | Isıtma süresi, alaşım ilaveleri (kg) | setpoint | |
| | Pota sıcaklığı, süperheat (°C) | measurement | |
| Sürekli döküm | Döküm hızı (m/dk) | setpoint | |
| | Kalıp seviyesi salınımı, soğutma suyu debisi | setpoint | Yüzey kalitesi |
| | Süperheat | measurement | Breakout riski |
| Tav fırını | Bölge sıcaklıkları (°C) × 3–5 | **setpoint** | Doğalgazın tamamı |
| | Baca O₂ / hava-yakıt oranı | setpoint | Yanma verimi |
| | Slab bekleme süresi | measurement | |
| Haddehane | Stand kuvvetleri (kN), ezme programı | setpoint | |
| | Sarma sıcaklığı, bitirme sıcaklığı (°C) | setpoint | Mekanik özellikleri belirler |
| Ortam | Dış hava sıcaklığı | context | |

---

## 3. Kalite / Laboratuvar Verisi (LIMS) — **P0**

Seyrektir (heat/reel/coil başına 1 satır) ve **batch_id ile birleştirilir**.

```json
{"sample_id":"L-2026-00981","batch_id":"REEL-2026-0412","ts":"2026-01-04T10:02:00Z",
 "property":"tensile_index","value":48.2,"unit":"Nm/g","spec_min":45,"spec_max":null,"passed":true}
```

- Kağıt: nem, gramaj, kül, patlama/çekme dayanımı, parlaklık, gözeneklilik, kopma sayısı
- Çelik: dökümhane kimyasal analizi (spektrometre), akma/çekme, uzama, sertlik, yüzey kusur raporu

> **Spec min/max mutlaka istenmelidir.** Kalite "iyi/kötü" değil, "spec içi/dışı"dır.
> Optimizasyon problemi = *spec'in alt sınırına en yakın kalitede üret* (over-quality = boşa para).

---

## 4. Üretim / MES Olayları — **P0**

| Alan | Örnek |
|---|---|
| batch_id | `REEL-2026-0412` / `HEAT-88231` |
| line_id | `PM2` / `EAF1` |
| product_code | `TL80` / `S235JR` |
| start_ts / end_ts | |
| produced_qty / uom | 24.6 ton |
| scrap_qty | 1.2 ton (broke / ıskarta) |
| order_id | |

Bu tablo, saniyelik telemetriyi "ton başına" metriklere çevirmemizi sağlar. **Olmadan kWh/ton hesaplanamaz.**

---

## 5. Duruş & OEE — **P0**

| Alan | Örnek |
|---|---|
| ts_start / ts_end | |
| asset_id | `PM2` |
| category | planned / unplanned |
| reason_code | `BRK-01` (kağıt kopuşu), `MEC-14` (rulman) |
| reason_text | |
| shift | |

> Duruş verisi iki iş yapar: (1) OEE hesaplama, (2) **kestirimci bakım / kopuş tahmini modelinin etiketi**.
> Reason code sözlüğü (kod → açıklama) mutlaka istenmeli.

---

## 6. Enerji Sayaçları — **P0**

| Alan | Örnek |
|---|---|
| ts | |
| meter_id / asset_id | `EM-PM2-DRIVES` |
| medium | electricity / steam / natural_gas / compressed_air / water |
| value + unit | 412.5 kWh |

**Kritik soru: sayaçlar ne kadar alt-kırılımlı?**
- Fabrika geneli tek sayaç → sadece kabaca modelleme (**kabul edilebilir minimum**)
- Hat bazlı → iyi
- Makine/bölüm bazlı (refiner, vakum pompaları, kurutma buharı ayrı) → **ideal**, tavsiye kalitesi buradan gelir

Alt sayaç yoksa: motor listesi + nominal güç + çalışma saatleri ile **tahmini dağıtım (disaggregation)** yaparız; bunu müşteriye baştan söyleriz.

---

## 7. Bakım Verisi — **P1**

İş emri no, ekipman, arıza kodu, başlangıç/bitiş, planlı/plansız, maliyet, değişen parça.
Kestirimci bakım demosu için gerekir.

---

## 8. Bağlam / Dış Veri — **P1 / P2**

- **Hava durumu** (P1): dış sıcaklık/nem/basınç. Fabrikada yoksa açık meteoroloji API'sinden çekilir — **bedava ve etkili**.
- Hammadde giriş analizi (P1): selüloz partisi özellikleri / hurda kalite sertifikası.
- Enerji piyasa fiyatı (P2): saatlik PTF.

---

## 9. İlk Teslimat Paketi — müşteriden istenecek somut liste

> Bu listeyi olduğu gibi maile koy.

1. `sensor_registry.xlsx` — tag, açıklama, birim, **rol (setpoint/ölçüm/bağlam)**, min/max, emniyet aralığı
2. `telemetry_YYYYMM.csv` — son **12 ay**, **1 dk** ortalama, geniş format, UTC timestamp
3. `production_batches.csv` — son 12 ay, batch/heat/reel bazında üretim ve ıskarta
4. `quality_lab.csv` — son 12 ay, batch_id ile eşleşen laboratuvar sonuçları + spec sınırları
5. `downtime.csv` — son 12 ay, duruş kayıtları + `reason_code_dictionary.csv`
6. `energy_meters.csv` — son 12 ay, mümkün olan en alt kırılım
7. `products.csv` — ürün/grade kataloğu, hedefler ve spec'ler
8. `shifts.csv` — vardiya takvimi
9. **1 saatlik teknik görüşme** — proses mühendisiyle: "Bugün operatör hangi kolları çeviriyor ve neye göre karar veriyor?"

**Boyut beklentisi:** 300 tag × 1 dk × 12 ay ≈ 158M satır ≈ ham CSV olarak ~8–12 GB, Elasticsearch'te sıkıştırılmış ~2–3 GB. Tamamen yönetilebilir.

---

## 10. Veri gelmeden ne yapıyoruz? — **Sentetik İkiz**

Bu repo, yukarıdaki şemanın **birebir aynısını** üreten fizik-temelli bir simülatör içerir
(`src/twin/simulator/`). Böylece:

- Tüm pipeline (ingest → feature → model → optimizasyon → dashboard) **bugün** ayakta,
- Müşteri verisi geldiğinde **sadece `ingest` adaptörü** değişir, gerisi aynen çalışır,
- Demo/sunum sentetik veriyle yapılabilir, "veri bekliyoruz" ölü zamanı olmaz.

Simülatörde bilinçli olarak **gerçek optimizasyon boşluğu** bırakılmıştır:
fazla buhar = boşa enerji, az buhar = spec dışı nem, fazla hız = kopuş.
Model bu ilişkiyi veriden öğrenir — biz denklemi modele söylemeyiz.
