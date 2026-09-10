# 04 — Optimizasyon Katmanı

> Tahmin bir rapordur. **Optimizasyon bir karardır.** Müşteri ikincisi için para öder.

---

## Problem formülasyonu

Anlık bağlam **c** (grade, hammadde, ortam — değiştiremeyiz) verildiğinde,
kontrol edilebilir setpoint vektörü **x**'i ara:

```
min_x    J(x, c) = w_e · Enerji_Maliyeti(x,c)
                 + w_q · Kalite_Cezası(x,c)
                 − w_p · Üretim_Değeri(x,c)

s.t.     x_i ∈ [op_low_i, op_high_i]                (emniyet aralığı — sensor_registry)
         |x_i − x_i^current| ≤ Δmax_i               (değişim hızı limiti)
         x ∈ eğitim_veri_zarfı                      (ekstrapolasyon yasağı)
         spec_min ≤ Kalite(x,c) ≤ spec_max          (nem / sıcaklık / kimyasal)
         Üretim(x,c) ≥ min_rate                     (sipariş taahhüdü)
         P(kopuş | x,c) ≤ τ                          (risk tavanı)
```

Her `Enerji`, `Kalite`, `Üretim`, `P(kopuş)` fonksiyonu **eğitilmiş RandomForest modelidir.**
Yani fiziği biz yazmıyoruz — veriden öğrenilmiş vekil modeli kullanıyoruz.

### Amaç fonksiyonu — neden ₺?

```python
J = elektrik_fiyatı(saat) · kWh_elektrik/ton
  + gaz_fiyatı          · kWh_termal/ton
  + ıskarta_maliyeti    · P(spec_dışı)
  − ürün_marjı          · (ton/saat)
```
Çıktı **₺/ton**. Yönetim sunumunda "kWh/ton %4 düştü" değil,
**"ton başına 60 ₺, yılda 12,2 milyon ₺"** denir.

Saatlik elektrik tarifesi varsa optimizasyon **kendiliğinden** puant saatlerde
enerji-yoğun rejimden kaçmayı öğrenir — ekstra kod yazmadan.

---

## Çözücü seçimi

| Yöntem | Ne zaman | Neden |
|---|---|---|
| **Grid search** | ≤ 3 değişken, demo/görselleştirme | Isı haritası çizilebilir, %100 açıklanabilir |
| **Random + local refine** | 4–8 değişken, hızlı cevap | Bağımlılıksız, 1 sn'de sonuç |
| **Differential Evolution** (varsayılan) | 4–15 değişken | scipy'de hazır, türev istemez, RF'in basamaklı yüzeyinde çalışır, kısıtları ceza terimiyle alır |
| Bayesian opt. (Optuna) | > 15 değişken, pahalı simülasyon | Bu ölçekte gereksiz |

> ⚠️ **Gradyan tabanlı çözücüler (SLSQP, L-BFGS) RandomForest ile çalışmaz.**
> RF çıktısı parçalı-sabittir; gradyan her yerde sıfırdır. Türevsiz çözücü şart.

Kısıtlar **ceza (penalty)** olarak eklenir; böylece çözücü fizibıl olmayan bölgeden
yumuşakça uzaklaşır ve her zaman bir cevap döner:
```python
J_total = J + 1e4 * max(0, moisture - spec_max)**2
            + 1e4 * max(0, spec_min - moisture)**2
            + 1e4 * max(0, min_rate - rate)**2
            + 1e4 * max(0, p_break - tau)**2
```

---

## Ekstrapolasyon koruması — projenin en kritik güvenlik önlemi

RandomForest gördüğü aralığın dışını **bilmez, sabit değer basar**. Bu, optimizasyonu
kandırmak için bire birdir: çözücü "buharı 0.5 bar yap, enerji sıfıra iner" der ve saçmalar.

İki katmanlı koruma (`optimize/search.py`):

**1. Kutu kısıtı.** Arama uzayı üç kaynağın kesişimi: emniyet aralığı ∩ değişim
hızı limiti ∩ eğitim zarfı. Zarf, değişkenin **tüm kolon ailesi** üzerinden
kesilir — ham değer, lag'leri ve rolling ortalamaları. Sadece ham değişkeni
kesmek tutarsızlık üretir: arama uzayı bir nokta önerir, güven kontrolü aynı
noktaya "ekstrapolasyon" der. Rolling ortalamaların dağılımı anlık değerden
**daha dardır** (*"bu hızı 60 dakika boyunca hiç sürdürmediniz"*), asıl
kısıtlayıcı odur.

**2. Yoğunluk (joint support) kontrolü.** Kutu kontrolü tek başına yetmez:

> Her değişken tek tek aralığının içinde olup **birleşimleri hiç gözlenmemiş**
> olabilir. Optimizasyonu kandıran nokta tam olarak orasıdır.

Model paketinde eğitim tasarımının temsili bir alt örneği (≤2000 satır, ham
değişkenler) saklanır. Önerilen nokta ile aralarındaki **normalize edilmiş
maks-norm mesafesi** ölçülür; `radius=0.15` "her değişkende aralığın %15'i kadar
yakınlık" demektir. Sonuç: *"bu çalışma noktasına benzer bir noktayı eğitimde
kaç kez gördüm?"*

> ⚠️ Önceki sürümde bu kontrol ağacın yaprak sayımlarıyla yapılıyordu. Sorun:
> ne `HistGradientBoosting` ne de `Ridge` `.apply()` sunar — kontrol **sessizce
> devre dışı kalıyor** ve sistem "bol gözlenmiş" diye raporluyordu. Ölçülmeyen
> bir şeyi "yüksek" diye raporlamak, hiç raporlamamaktan kötüdür. Yöntem bu
> yüzden model-bağımsız hale getirildi; ölçüm yapılamıyorsa güven **en fazla
> ORTA** raporlanır ve sebebi yazılır.

> Dashboard'da her tavsiyenin yanında **güven rozeti** gösterilir:
> 🟢 Yüksek (benzer rejim bol gözlendi) · 🟡 Orta (az gözlendi ya da ölçülemedi) ·
> 🔴 Ekstrapolasyon — uygulanmaz

---

## Üç kullanım modu

### 1. What-if (senaryo analizi)
Operatör/mühendis slider'ları çevirir, ikiz anında cevap verir.
*"Hızı 900'e çıkarırsam ne olur?"* → `+1.7 t/h, nem 6.9% (SPEC DIŞI!), enerji +3%`
**En kolay satılan özellik.** Eğitim aracı olarak da kullanılır.

### 2. Advisory (tavsiye) — ana mod
Her 5 dakikada bir mevcut bağlamla optimizasyon koşar, en iyi setpoint setini
operatör ekranına yazar. **Operatör uygular veya reddeder.**
Kabul/ret kaydı tutulur → hem güven ölçümü hem de modelin zayıf noktalarının haritası.

### 3. Setpoint keşfi (offline)
Gece/hafta sonu, tüm grade × mevsim kombinasyonları için optimum reçeteler taranır
→ **"altın reçete" (golden recipe) tablosu** üretilir. Bu tablo, ikiz kapalıyken bile
operatörün elinde kalan somut çıktıdır. Çoğu müşteri için en somut teslimat budur.

---

## Kazanç nasıl kanıtlanır? — Doğrulama (attribution) protokolü

En sık düşülen tuzak: "enerji düştü çünkü ikiz" demek. Hayır — hava ısındı, ondan düştü.

**Kullandığımız yöntem — normalize edilmiş fark:**

```
Beklenen_taban = Model(mevcut_setpointler, bağlam)      ← ikiz olmasaydı ne olurdu
Gerçekleşen    = ölçülen kWh/ton
Kazanç         = Beklenen_taban − Gerçekleşen
```
Bağlam (grade, hava, hammadde) her iki tarafta da aynı olduğu için etkisi düşer.

Ek olarak, ilk 8 haftada **A/B vardiya denemesi** önerilir:
tek gün tavsiye açık, ertesi gün kapalı; grade karması dengelenir.
İstatistiksel anlamlılık testi ile rapor edilir.

> Bu bölümü sunuma koymak, projeyi "güzel ekran" olmaktan çıkarıp
> **ölçülebilir iş sonucu** haline getirir.

---

## Örnek çıktı (dashboard "Optimizasyon" sekmesi)

```
Bağlam:  Grade TL80 · Ortam 8°C / %72 RH · Selüloz CSF 385 mL · Saat 03:00 (gece tarifesi)

Değişken              Mevcut    Tavsiye    Δ        Emniyet aralığı
─────────────────────────────────────────────────────────────────────
Makine hızı           845       862 m/dk   +17      [500, 1150]
Buhar G1              2.71      2.52 bar   −0.19    [0.5, 3.5]
Buhar G2              3.38      3.11 bar   −0.27    [0.8, 4.5]
Buhar G3              3.76      3.42 bar   −0.34    [0.8, 5.0]
Pres nip yükü         312       348 kN/m   +36      [250, 420]
Elek vakumu           −47       −53 kPa    −6       [−70, −35]
Refiner SEC           95        91 kWh/t   −4       [60, 140]

Tahmini sonuç:
  Özgül enerji  1.371 → 1.309 kWh/t   (−4.5 %)
    elektrik      231 →   239 kWh/t   (+3.5 %  ← nip ve vakum arttı)
    termal      1.140 → 1.070 kWh/t   (−6.1 %  ← buhar azaldı)
  Nem            5.05 →  5.44 %       (spec 4.8–6.2 ✅)
  Üretim         26.1 →  26.6 t/h     (+1.9 %)
  Duruş riski     3.1 →   3.4 %       (tavan 6 % ✅)
  ──────────────────────────────────────────────────
  Kazanç         60 ₺/ton  ≈  1.560 ₺/saat  ≈  12,2 M₺/yıl
                 (26,1 t/h × 7.800 saat varsayımıyla)
  Güven          🟢 Yüksek (benzer rejim 1.240 kez gözlendi)
```

**Tablodaki asıl mesaj:** elektrik tüketimi ARTIYOR, toplam maliyet DÜŞÜYOR.
Pres ve vakumla alınan her kg su, kurutmada alınmayan kg sudur; mekanik su alma
termalin ~1/10'u maliyetindedir. Sadece "kWh düşürmeye" bakan bir optimizasyon
bu çözümü **bulamaz** — amaç fonksiyonu ₺ olduğu için bulunur.

Bu ekran, projenin tek cümlelik özetidir.
