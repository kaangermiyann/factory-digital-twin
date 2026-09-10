# 05 — Yol Haritası, Riskler ve Sunum Planı

## Fazlar

### Faz 0 — Sentetik İkiz (veri beklerken) · **1–2 hafta** · ✅ bu repo
- Canonical şema + depolama katmanı (ES/MySQL/Parquet)
- Fizik-temelli simülatör (kağıt makinesi + EAF çelikhane)
- Feature pipeline + RF/GBM eğitimi + optimizasyon + dashboard
- **Çıktı:** uçtan uca çalışan demo. Müşteri toplantısına bununla gidilir.
- **Kritik fayda:** "Veri gelince ne yapacağız?" değil, "veriyi buraya koyunca bu çıkacak" denir.

### Faz 1 — Veri Keşfi · **2–3 hafta**
- `docs/01-veri-sozlesmesi.md` müşteriye gönderilir
- Proses mühendisiyle 1 saatlik atölye: **"operatör hangi kolları çeviriyor?"**
- Örnek veri (1 ay) alınır → `ingest/validate.py` ile kalite raporu üretilir
- **Çıktı:** Veri Kalite Raporu + doldurulmuş `config/profile.<musteri>.yaml` + fizibilite kararı (git/gitme)
- **Bu fazda proje iptal edilebilir olmalı.** Kötü veriyle devam etmek en pahalı hatadır.

### Faz 2 — Baseline Modeller · **3–4 hafta**
- 12 aylık tam veri yüklenir
- M1–M4 modelleri eğitilir, naive/ridge/RF/GBM karşılaştırması
- **Çıktı:** Model karşılaştırma raporu + feature importance + PDP grafikleri
- **Başarı kriteri:** SEC modeli MAPE < %5 ve naive baseline'ı ≥ %40 geçmeli

### Faz 3 — Optimizasyon + Advisory · **3–4 hafta**
- Emniyet aralıkları ve kısıtlar proses mühendisiyle birlikte doğrulanır
- Altın reçete tablosu üretilir
- Dashboard operatör ekranına konur (**salt tavsiye, PLC'ye yazma yok**)
- **Çıktı:** Canlı advisory sistemi + kabul/ret kaydı

### Faz 4 — Doğrulama ve Ölçekleme · **8+ hafta**
- A/B vardiya denemesi, kazanç doğrulaması
- Drift takibi + otomatik yeniden eğitim
- İkinci hatta/fabrikaya replikasyon (yeni bir `config/profile.<hat>.yaml` yazarak)

---

## Riskler ve önlemler

| Risk | Olasılık | Etki | Önlem |
|---|---|---|---|
| **Veri hiç gelmez / aylarca gecikir** | Yüksek | Kritik | Sentetik ikiz hazır — demo veriye bağlı değil |
| **Tag açıklamaları yok / anlamsız** | Yüksek | Kritik | Faz 1 atölyesi zorunlu; `sensor_registry` teslimat şartı |
| **Setpoint/ölçüm ayrımı yok** | Orta | Kritik | Ayrım yapılamazsa optimizasyon yerine sadece tahmin+erken uyarı teslim edilir |
| **Enerji sayacı sadece fabrika geneli** | Yüksek | Orta | Motor listesi + çalışma saatiyle disaggregation; beklenti baştan düşürülür |
| **Proses hep aynı noktada çalışmış (varyans yok)** | Orta | Yüksek | Model öğrenecek bir şey bulamaz → kontrollü DOE (deney tasarımı) önerilir |
| **Duruş kayıtları elle ve güvenilmez** | Yüksek | Orta | Duruşu telemetriden türet (hız < eşik & süre > 2 dk) |
| **Operatör güvenmez, tavsiyeyi uygulamaz** | Yüksek | Kritik | Açıklanabilirlik + güven rozeti + what-if aracıyla önce **eğitim aracı** olarak konumlandır |
| **Model drift (bakım sonrası proses değişir)** | Kesin | Orta | Haftalık MAE takibi + aylık yeniden eğitim |
| **Kapalı döngü kontrol beklentisi** | Orta | Yüksek | Kapsam net: advisory. Closed-loop ayrı proje, ayrı güvenlik onayı |

---

## Başarı kriterleri (sözleşmeye yazılabilir)

| Seviye | Kriter |
|---|---|
| Teknik | SEC modeli holdout MAPE < %5; naive baseline'a göre ≥ %40 iyileşme |
| Teknik | Kopuş modeli: precision %50'de recall ≥ %30 (30 dk ufuk) |
| Operasyonel | Tavsiye kabul oranı ≥ %40 (8 hafta sonunda) |
| İş | Doğrulanmış özgül enerji iyileşmesi ≥ %2 (normalize edilmiş fark yöntemiyle) |
| İş | Iskarta oranında ≥ %5 göreli azalma |

> **%2 enerji tasarrufu iddiası kasten mütevazıdır.** Literatürde %3–8 raporlanır ama
> az vaat edip fazla teslim etmek, tersinden çok daha iyidir.

---

## Sunum akışı (45 dk) — demo senaryosu

1. **(5 dk) Problem** — "Fabrikanız ton başına X kWh harcıyor. Bu sayı vardiyaya, havaya
   ve operatöre göre %15 oynuyor. Bu oynamanın büyük kısmı **gereksiz**."
2. **(5 dk) Veri** — `docs/01`: "Sizden şunları istiyoruz, bunlar zaten sisteminizde var."
3. **(5 dk) Mimari** — `docs/02` şeması. On-prem, veri dışarı çıkmaz.
4. **(10 dk) Canlı demo — Dashboard ana ekranı (Gerçek vs Tahmin)**
   Ölçülen ve tahmin çizgileri üst üste biniyor. Ekranın üstündeki yeşil bant
   önemli: *"model bu veriyi hiç görmedi."* Eğitim döneminde güzel grafik
   göstermek kolaydır; buradaki, holdout dönemidir.
5. **(5 dk) Model Detayı sayfası** — naive/doğrusal/ağaç karşılaştırma tablosu + feature importance.
   *"Model en çok kurutmaya giren su yüküne ve ortam nemine bakıyor."* → proses mühendisi
   onaylar, güven kazanılır. Ardından PDP: *"hız şu değeri geçince enerji tırmanıyor."*
   Burada dürüst olun: enerji modelinde ağaçlar doğrusalı geçemedi, nem ve duruş riskinde
   açık ara geçti. Bunu söylemek güven artırır (bkz. `docs/03`).
6. **(10 dk) Optimizasyon sayfası** — what-if slider'ları, sonra "Optimize Et" butonu.
   ₺/yıl kazanç ekranı. **Sunumun doruk noktası.**
7. **(5 dk) Yol haritası + riskler** — `docs/05`. Dürüstlük satar.

**Demo altın kuralı:** Simülatör olduğunu **gizleme**. "Sizin verinizle bu ekranın
aynısı, sizin sayılarınızla dolar" de. Sahtelik yakalanırsa tüm güven gider.

---

## Sonraki adımlar (bu repoda yapılabilecekler)

- [ ] Çelikhane (EAF) simülatörünü kağıt kadar detaylandır
- [ ] Kestirimci bakım modeline titreşim/akım imzası ekle
- [ ] Optimizasyon tavsiyelerini `twin-recommendations` index'ine yaz, kabul/ret takibi
- [ ] Altın reçete tablosu üreticisi (`optimize/golden_recipe.py`)
- [ ] MLflow / basit model registry UI
- [ ] Grade-değişimi optimizasyonu (transition time minimizasyonu) — ayrı ve değerli bir problem
