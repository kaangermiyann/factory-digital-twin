# 02 — Mimari

## Tasarım ilkesi

> **Depolama, modelden bağımsız olmalı. Model, arayüzden bağımsız olmalı.
> Müşteri verisi geldiğinde sadece bir adaptör değişmeli.**

Bu yüzden tek bir **canonical şema** (`src/twin/schema.py`) var ve her şey ona konuşuyor.
Elasticsearch, MySQL veya lokal Parquet — üçü de aynı `Repository` arayüzünü uygular.

---

## Katmanlar

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. KAYNAK                                                            │
│    Historian (PI/IP.21) · SCADA/OPC-UA · MES · LIMS · ERP · Sayaç    │
│    [bugün: fizik-temelli simülatör]                                  │
└───────────────┬──────────────────────────────────────────────────────┘
                │  CSV / Parquet / REST / OPC-UA
┌───────────────▼──────────────────────────────────────────────────────┐
│ 2. INGEST + VALIDATION          src/twin/ingest/                     │
│    · wide → canonical dönüşümü (profile.yaml içindeki `tag:` alanı)  │
│    · birim normalizasyonu, saat dilimi → UTC                         │
│    · veri kalite kapısı: aralık dışı, donuk sensör, boşluk, çift kayıt│
│    · REDDEDİLEN kayıtlar quarantine index'ine → raporlanır            │
└───────────────┬──────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│ 3. DEPOLAMA                     src/twin/storage/                    │
│                                                                      │
│   Elasticsearch (birincil, zaman serisi)   MySQL (master data)       │
│   ├ twin-telemetry-{YYYY.MM}   (ILM)       ├ asset                   │
│   ├ twin-energy-{YYYY.MM}                  ├ sensor_registry         │
│   ├ twin-quality                           ├ product                 │
│   ├ twin-batches                           ├ shift_calendar          │
│   ├ twin-events                            └ energy_tariff           │
│   └ twin-predictions-{YYYY.MM}                                       │
│                                                                      │
│   Parquet backend: docker'sız lokal geliştirme/demo için             │
└───────────────┬──────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│ 4. FEATURE ENGINEERING          src/twin/features/                   │
│    · long → wide pivot, 5 dk yeniden örnekleme                       │
│    · lag / rolling mean / rolling std / delta (proses ölü zamanı!)   │
│    · batch & kalite & ortam join                                     │
│    · takvim özellikleri (vardiya, saat, hafta günü)                  │
│    · leakage koruması: hedefin downstream'i olan tag'ler dışlanır    │
└───────────────┬──────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│ 5. MODEL                        src/twin/models/                     │
│    RandomForest (ana) + HistGradientBoosting + Ridge/naive baseline  │
│    TimeSeriesSplit CV · permutation importance · model registry      │
│    Hedefler: SEC(kWh/t) · nem/kalite · üretim hızı · kopuş riski     │
└───────────────┬──────────────────────────────────────────────────────┘
                │  eğitilmiş model = "vekil (surrogate) proses modeli"
┌───────────────▼──────────────────────────────────────────────────────┐
│ 6. DİJİTAL İKİZ + OPTİMİZASYON  src/twin/optimize/                   │
│    · what-if: "hızı 20 m/dk artırırsam ne olur?"                     │
│    · constrained optimization (differential evolution):              │
│         min  enerji_maliyeti(₺/ton)                                  │
│         s.t. nem ∈ spec, üretim ≥ min, kopuş_riski ≤ eşik,           │
│              setpoint ∈ emniyet aralığı, |Δsetpoint| ≤ rate limit    │
└───────────────┬──────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│ 7. SERVİS                       src/twin/serving/                    │
│    FastAPI:  /predict  /whatif  /optimize  /compare  /kpis           │
│    Scorer loop: her 60 sn canlı veriyi skorlar → twin-predictions    │
└───────────────┬──────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────┐
│ 8. DASHBOARD                    src/twin/dashboard/                  │
│    ANA EKRAN: Gerçek vs Tahmin — sadece holdout (görülmemiş) dönem   │
│    pages/: Optimizasyon ▸ Model Detayı ▸ Veri Kalitesi               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Neden Elasticsearch?

| Kriter | Elasticsearch | MySQL |
|---|---|---|
| 150M+ satır zaman serisi | ✅ Doğal | ⚠️ Partition + index bakımı gerekir |
| "Son 24 saatin 5 dk ortalaması" | ✅ `date_histogram` agg, saniyeler | ⚠️ `GROUP BY` + full scan |
| Şema esnekliği (yeni tag eklenmesi) | ✅ dynamic mapping | ❌ ALTER TABLE |
| ILM / otomatik arşivleme | ✅ Yerleşik | ❌ Manuel |
| Kibana ile hızlı keşif | ✅ Bedava geliyor | ❌ |
| Transaction / referans bütünlüğü | ❌ | ✅ |
| Master data, JOIN | ❌ | ✅ |

**Karar: Hibrit.** Zaman serisi + olay → Elasticsearch. Master data → MySQL.
Ama kod bunu bilmiyor; `Repository` arayüzünün arkasında. Müşteri "sadece MySQL"
derse `MySQLRepository` kullanılır, geri kalan hiçbir satır değişmez.

**Kritik ES ayarları** (`storage/mappings/`):
- `telemetry` index'i: `tag` ve `asset_id` → `keyword`, `value` → `double`, `ts` → `date`
- Aylık index + ILM: 90 gün hot → 12 ay warm → sonra frozen/delete
- `index.codec: best_compression`
- `refresh_interval: 30s` (canlı yazımda gereksiz yere refresh yapma)

---

## Canonical şema özeti

| Index / Tablo | Anahtar alanlar |
|---|---|
| `twin-telemetry-*` | ts, asset_id, tag, value, unit, role, quality |
| `twin-energy-*` | ts, meter_id, asset_id, medium, value, unit |
| `twin-batches` | batch_id, line_id, product_code, start_ts, end_ts, produced_qty, scrap_qty |
| `twin-quality` | sample_id, batch_id, ts, property, value, spec_min, spec_max, passed |
| `twin-events` | event_id, ts_start, ts_end, asset_id, type, category, reason_code, duration_s |
| `twin-predictions-*` | ts, line_id, model_name, model_version, target, y_pred, y_true, horizon_min |

`twin-predictions` ayrı bir index olması önemli: **canlı vs tahmin kıyaslaması** ve
**model drift takibi** bu index üzerinden yapılır. `y_true` başta boştur, gerçek değer
geldiğinde geriye dönük doldurulur.

---

## Deployment

```
docker compose up            →  elasticsearch, kibana, mysql, api, dashboard, simulator
```
On-prem çalışır (fabrikalar veriyi dışarı çıkarmaz). Tek makinede 16 GB RAM yeter.
İnternet gerekmez.

**Güvenlik notu:** İkiz **read-only**'dir. PLC/DCS'e hiçbir zaman yazmaz.
Çıktı, operatöre "tavsiye" olarak gösterilir (advisory / open-loop).
Closed-loop kontrol ayrı bir proje ve ayrı bir güvenlik onayıdır — kapsam dışı.
