# Süper Lig Skor Tahmini (İstatistik + Makine Öğrenmesi + Yapay Zekâ)

## NOSTRADAMUS: çevrimdışı matematiksel araştırma modeli

`src/nostradamus.py`, mevcut `data/matches.csv` üzerinden çalışan ayrı ve yeniden
üretilebilir bir araştırma akışıdır. API anahtarı veya internet gerektirmez; haber,
elle gol çarpanı ve yapay zekâ yorumu kullanmaz. Henüz doğrulanmış kârlılık yoktur.

PowerShell'de proje klasöründen:

```powershell
.venv/Scripts/python.exe src/nostradamus.py test --baslangic 2024-07-01
.venv/Scripts/python.exe src/nostradamus.py tahmin Alanyaspor Trabzonspor --tarih 2026-11-08 --oran 4.8 4.1 1.645 --ornek 100
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

İkinci komuttaki oranlar kullanım örneğidir; güncel fiyat olarak kabul edilmemelidir.
`--oran` sırası ev sahibi / beraberlik / deplasmandır. `--kasa 1000` yalnızca sanal
kasa hesabını ekler; verilmezse tutarlar sıfırdır. Takımların adları ve tarih kullanıcı
girdisidir; araç maçın gerçekten fikstürde olduğunu veya henüz başlamadığını doğrulamaz.

### Matematik ve karar

- Ev gol beklentisi: `exp(mu + ev_avantajı + hücum_ev + savunma_dep)`;
  deplasman: `exp(mu + hücum_dep + savunma_ev)`. Burada savunma parametresinin
  yüksek olması daha fazla gol yeme eğilimidir. Takım parametreleri merkezlenir.
- Son dört yılın yalnızca kesim tarihinden önce oynanmış maçları kullanılır.
  Ağırlık `2^(-geçen_gün / yarı_ömür)`; varsayılan yarı ömür 365 gündür.
  L2 katsayısı 0,5'tir. Bu ayarlar bu testte optimize edilmemiştir.
- Uygulama iki aşamalıdır: önce düzenlileştirilmiş Poisson hücum/savunma
  parametreleri, sonra düşük skor Dixon–Coles `rho` düzeltmesi öğrenilir.
  Bütün parametrelerin birlikte optimize edildiği tam olabilirlik modeli değildir.
- `rho` tüm dört düşük skor hücresinin pozitif kalacağı aralıkta tutulur.
  Skor matrisinin boyutu gol beklentisine göre büyür; 9 golün üzerindeki kuyruk
  sabit bir kesmeyle kaybedilmez.
- Tahminde maç haftaları 100 kez yeniden örneklenip model yeniden eğitilir.
  Olasılıkların %5 yüzdeliği muhafazakâr alt tahmindir; bu yöntem kalibrasyon
  hatasını ortadan kaldırmaz ve kapsama garantili bir güven aralığı değildir.
- `alt_EV = min(nokta_olasılık, alt_tahmin) * oran - 1`. Varsayılan %5 eşiğini
  aşan en yüksek alt EV için, maç başına en fazla bir araştırma adayı üretilir.
  Sanal tutar çeyrek Kelly ile hesaplanır ve sanal kasanın %0,5'iyle sınırlandırılır.
- En az 200 eğitim maçı, iki takımın her biri için en az 5 yakın dönem maçı
  ve son 60 gün içinde sonuç bulunması gerekir. Veri yetersizse seçim üretilmez.
- Bu karar ve tutar yalnızca 1X2 içindir. Alt/üst ve KG olasılıkları bilgi amaçlıdır;
  bunlara ayrı kalibrasyon veya bahis kuralı uygulanmamıştır.

### Gerçekleştirilen kontrol

7 Kasım 2026'da mevcut yerel veriyle, 1 Temmuz 2024–6 Kasım 2026 tarih aralığında
739 maç üzerinde yeniden test yapıldı. Model en az 30 günde bir yeniden eğitildi;
her tahminin eğitim bitişi maç tarihinden öncedir. Bugünkü sonuçlar dışarıda tutulur.

| Ölçüm | Sonuç |
|---|---:|
| 739 maçta 1X2 isabeti | %51,42 |
| 739 maçta 1X2 log loss | 0,9863 |
| Oranı olan aynı 703 maçta model isabeti | %52,20 |
| Aynı 703 maçta marjı temizlenmiş piyasa isabeti | %58,61 |
| Aynı 703 maçta model / piyasa log loss | 0,9802 / 0,9373 |
| Nokta EV > %5, maç başına bir seçim | 463 bahis |
| Birer birim sabit tutarla keşifsel net sonuç | −161,736 birim |
| Keşifsel ROI | −%34,93 |
| Haftalık küme bootstrap ile %95 ROI aralığı | −%45,79 / −%22,92 |

Bu strateji testi **tahmin belirsizliği filtresini ve Kelly tutarlarını kullanmaz**;
basit nokta EV kuralının geçmiş sonucudur. Canlı tahmin komutunun muhafazakâr
kuralının kârlılık testi olarak yorumlanmamalıdır. Yerel oranların hangi anda
alındığı doğrulanmadığından sonuçlar keşifseldir. Daha önce incelenmiş dönemlerin
tekrar kullanılması yeni, dokunulmamış bir test oluşturmaz. Piyasa karşılaştırması
aynı maçlarla yapılır; oran yokken model tahmini piyasa sonucu gibi gösterilmez.

Çıktılar `reports/nostradamus/` altındadır:

- `summary.json`: metrikler, veri SHA-256 özeti, parametreler ve zaman bilgisi.
- `predictions.csv`: maç tahminleri ve her tahminin eğitim kesimi.
- `bets.csv`: yalnızca keşifsel sabit tutarlı test seçimleri.
- `calibration.csv`: 1/X/2 için olasılık dilimleri ve gerçekleşme oranları.
- `forecast.json`: son örnek maç tahmini, belirsizlik alt tahminleri ve sanal kararlar.

`--veri`, `--cikti`, `--yari-omur`, `--l2`, `--esik` ile girdiler değiştirilebilir.
Testte `--yenile` yeniden eğitim aralığıdır; tahminde `--ornek` (en az 30) ve
`--tohum` belirsizlik hesabını kontrol eder. Sonuçları gördükten sonra ayar seçmek
aynı dönemi doğrulama verisine dönüştürür; sonraki değerlendirme ayrı dönemde yapılmalıdır.

Paylaşılan `src/models.py` içinde tarih sınırı, geçersiz gol beklentileri,
negatif skor olasılıkları ve kuyruk kesimi için kontroller eklendi. Önceden üretilmiş
diğer raporlar ve kalibrasyon dosyaları bu değişiklikten önceki modele aittir;
bu araştırma komutu onları kullanmaz veya yeniden yazmaz. Eski ML akışının yeni
çekirdekle ölçülmesi için `src/backtest.py` ayrıca yeniden çalıştırılmalıdır.

Süper Lig maçları için skor olasılıkları üreten uygulama. Üç katmandan oluşur:

1. **İstatistik ve ML modeli:** 90'dan fazla değişkenden (API verisiyle 150'den fazla) beklenen gol, 1/X/2, 2.5 alt/üst, karşılıklı gol ve skor tablosu üretir.
2. **Claude yapay zekâ analisti:** Güncel haberleri web'de arar, modelin göremediği durumları (sakatlık, hoca değişikliği, motivasyon) değerlendirir, tahmini gerekçeleriyle sınırlı ölçüde düzeltir.
3. **Monte Carlo sezon simülasyonu:** Kalan fikstürü 10.000 kez oynatıp şampiyonluk, Avrupa ve küme düşme olasılıklarını hesaplar.

## Masaüstü uygulaması

Masaüstündeki veya Başlat menüsündeki **Nostradamus** kısayoluna çift tıklayın (ya da `Nostradamus.pyw` dosyasına). Konsol penceresi açılmaz.

| Sekme | İçerik |
|---|---|
| Bu Hafta | Yaklaşan maçlar. Bir maç seçince: olasılık çubukları, skor ısı haritası, "Neden?" gerekçeleri, value tablosu (kendi oranlarınızı girip Hesapla), yapay zekâ analizi düğmesi |
| Maç Tahmini | İki takım ve tarih seçip herhangi bir maçı tahmin etme |
| Canlı Takip | Kayıtlı tahminler, canlı skorlar, isabet ve value kâr/zarar özeti; canlı maç varken dakikada bir yenilenir |
| Sezon Simülasyonu | Şampiyonluk / Avrupa / küme düşme olasılıkları; yeniden çalıştırma düğmesi |
| Model Başarısı | Geçmiş test, sürpriz alarmı ve value testi sonuçları |
| Günlük | Otomatik güncellemelerin kaydı |

Üst çubuktaki **Hızlı güncelle / Tam güncelle** düğmeleri otomatik görevlerle aynı işi elle yapar. İlk maç detayı açılışta modeli eğittiği için 10–20 saniye sürebilir; sonrakiler hızlıdır. Simge: `assets/nostradamus.ico` (portre: `assets/nostradamus_portre.webp`). Uygulama açılamazsa hata `reports/app_hata.txt` dosyasına yazılır.

## Kurulum

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Linux/Mac: .venv/bin/pip
```

`.env.example` dosyasını `.env` adıyla kopyalayıp anahtarlarınızı yazın:
- `API_FOOTBALL_KEY`: [api-football.com](https://www.api-football.com) adresinden alınır, ücretsiz plan yeterlidir. Güncel sonuçlar ile xG, şut, kadro ve hakem verisi için gerekir.
- `ANTHROPIC_API_KEY`: [platform.claude.com](https://platform.claude.com) adresinden alınır. Yapay zekâ analizi için gerekir.

İki anahtar da isteğe bağlıdır. Anahtarsız çalışırsa açık kaynak skor verisi kullanılır ve yapay zekâ analizi yapılmaz.

## Kullanım

```bash
python src/espn_source.py                      # ANA KAYNAK: 2017'den bugüne sonuçlar, oranlar, istatistikler (anahtarsız)
python src/livescore_source.py                 # güncel sezon (yedek kaynak, anahtarsız)
python src/api_football.py --sezon 2026        # xG, kadro, hakem ve geçmiş sezonlar (anahtar varsa; ~22 istek)
python src/parse_openfootball.py               # tüm kaynakları birleştir
python src/backtest.py                         # değişkenler + model eğitimi + geriye dönük test (~1 dk)

python src/predict.py Galatasaray Fenerbahçe --tarih 2026-11-08          # tek maç
python src/predict.py Galatasaray Fenerbahçe --tarih 2026-11-08 --yz     # + Claude analizi
python src/predict.py --fikstur 2026-11-06 2026-11-10                    # haftanın maçları
python src/simulate.py                                                   # sezon simülasyonu
python src/value.py tahmin Beşiktaş Gençlerbirliği --oran 1.25 6.00 11.00 --ust 1.55 2.45   # value bet
python src/value.py hafta                                                # 7 günün value analizi
python src/value.py test                                                 # value stratejisinin geçmiş kâr/zarar testi
```

`--ev-carpan 0.9` / `--dep-carpan 1.1`: beklenen golü elle ayarlamak için.

### Otomatik günlük güncelleme

Windows Görev Zamanlayıcı'da iki görev kayıtlı ("Nostradamus Tam Guncelleme" ve "Nostradamus Hizli Guncelleme"):

| Saat | Ne yapar |
|---|---|
| 11:00 | Yeni sonuç ve oranları indirir, modeli ve sürpriz modelini yeniden eğitir, biten maçları değerlendirir, 7 gün içindeki maçların tahminlerini kaydeder |
| 12:30–22:30, saatte bir | Güncel oranlar ve (maçtan ~1 saat önce açıklanan) ilk 11'lerle başlamamış maçların tahminlerini yeniler; son kayıt maç başlamadan hemen önceki tahmindir |

- Bilgisayar o saatte kapalıysa görev, açıldığında çalışır.
- Elle çalıştırmak için: `guncelle.bat` (tam) veya `guncelle.bat hizli`.
- Çalışma günlüğü `reports/daily_log.txt`, son değerlendirme `reports/son_durum.txt` dosyasında.
- Görevleri kaldırmak için Görev Zamanlayıcı'dan iki görevi silin.

### Canlı takip: tahminleri gerçek sonuçlarla karşılaştırma

```bash
python src/tracker.py kaydet      # önümüzdeki 7 günde başlamamış maçların tahminlerini kaydet
python src/tracker.py durum       # canlı skorlar + biten maçlarda isabet özeti
python src/tracker.py             # ikisi birden
```

- Tahmin yalnızca maç **başlamadan önce** kaydedilir (`data/predictions_log.csv`, zaman damgalı). Maç başladıktan sonra kayıt kilitlenir; böylece ölçüm dürüst kalır.
- `durum` komutu ESPN'den canlı skorları çeker. Oynanan maçlarda tahminin "şu an tutup tutmadığını" gösterir. Biten maçlarda maç sonucu, tam skor, 2.5 alt/üst isabetini, log loss değerini, sürpriz alarmı seviyelerine göre favori kayıplarını ve (oran varsa) bahis piyasasıyla karşılaştırmayı birikimli olarak raporlar (`reports/live_evaluation.csv`).
- En doğru ölçüm için her maç gününden önce `kaydet`, maçlardan sonra `durum` çalıştırın.
- Her tahminde bir **sürpriz alarmı** (düşük / orta / yüksek / çok yüksek) ve o seviyede geçmişte favorinin ne sıklıkla kaybettiği gösterilir (`src/upset_model.py`).

## Değişkenler

| Grup | Değişkenler | Kaynak |
|---|---|---|
| **Gelişmiş metrikler** | xG, xGA, **xPTS** (xG'ye göre hak edilen puan), puan − xPTS (şans), gol − xG (bitiricilik) | API |
| **Temel istatistikler** | Gol ortalamaları (genel, iç saha, deplasman), toplam şut, isabetli şut, ceza sahası içi şut, korner, kart, topla oynama, pas isabeti | openfootball + API |
| **İkili mücadele ve bireysel** | İkili mücadele kazanma %, başarılı çalım %, kilit pas, top kapma ve araya girme, oyuncu reyting ortalaması | API (oyuncu verisi) |
| **Güç ve derece sistemleri** | Elo (gol farkı çarpanlı), **Pi-rating** (iç saha ve deplasman ayrı), Dixon-Coles hücum/savunma gücü | Hesaplanan |
| **Form** | Son 5 ve 10 maç puan ortalaması, seriler, Elo beklentisine göre sapma, sezon puan ortalaması, lig sırası | Hesaplanan |
| **Kadro** | Eksik oyuncu sayısı, **kilit oyuncu eksikliği** (eksiklerin sezon dakika payı; 1.0 = bir tam zamanlı ilk 11 oyuncusu), eksiklerin gol ve asist payı | API |
| **İlk 11 ve rotasyon** | Önceki maçın ilk 11'iyle süreklilik, sahadaki 11'in son 10 maçta ilk 11'de başlama oranı, son 10 maçın %70'inden fazlasında başlayıp bu maçta olmayan asıl oyuncu sayısı, yeni yüzler | ESPN (maçtan ~1 saat önce) |
| **Yorgunluk** | Dinlenme günü (Avrupa kupası maçları dahil; Türkiye Kupası yalnızca API-Football ile), son 21 günde oynanan maç, son 14 günde kupa/Avrupa maçı, **deplasman yolculuk mesafesi** | Hesaplanan + API |
| **İç saha avantajı** | Takıma özel iç saha ve deplasman puanı ile aradaki fark | Hesaplanan |
| **Çevresel** | Maç günü sıcaklık, yağış, rüzgar, şiddetli yağmur | Open-Meteo (ücretsiz) |
| **Hakem** | Hakemin maç başına kart ve penaltı ortalaması, ev sahibinin kazanma oranı | API |
| **Bağlam** | Derbi, büyük kulüp, yeni yükselen takım, ikili rekabet geçmişi, sezonun ilerleme oranı | Hesaplanan |
| **Güncel haberler** | Sakatlık ve ceza haberleri, hoca değişikliği, motivasyon | Claude web araması |

Zemin tipi kullanılmıyor: Süper Lig'de tüm stadyumlar doğal çim olduğu için ayırt edici bilgi taşımıyor.

## Model seçimi ve doğruluk

Her maç yalnızca o tarihten önceki verilerle tahmin edildi (walk-forward). Ayarlar yalnızca önceki sezonlarda öğrenildi; test sezonları 2023-24'ten bugüne (1.121 maç), "görülmemiş" sütunu 2024-25 ve sonrası.

**Nihai tahmin = bahis piyasası + model.** Oran varsa piyasa olasılıkları ile modelin olasılıkları geçmiş sezonlarda öğrenilen ağırlıkla birleştirilip kalibre edilir. Oran yoksa model (Poisson GLM + LightGBM, kalibre) tek başına kullanılır. Oranlar gol modellerine girdi olarak verilmez; eski sezonlarda yarısı eksik olduğu için modeli bozuyordu.

| Model | 1X2 doğruluk | Görülmemiş sezonlar | Log loss ↓ | 2.5 Ü/A |
|---|---|---|---|---|
| Taban (lig ortalaması) | %45,7 | %45,7 | 1,065 | %53,5 |
| Dixon-Coles | %51,8 | %51,6 | 0,990 | %55,1 |
| Model (GLM + LightGBM) | %53,9 | %54,1 | 0,975 | %55,5 |
| Bahis piyasası tek başına | %57,0 | %57,9 | 0,944 | – |
| **NİHAİ: piyasa + model** | **%57,1** | **%58,0** | **0,934** | %56,0 |

- Model tek başına piyasanın yaklaşık 3 puan gerisinde. Piyasa sakatlık ve kadro bilgisini fiyata yansıtıyor.
- Birleşim piyasadan biraz daha iyi; asıl kazanç olasılıkların kalibrasyonunda (log loss 0,944 → 0,934).
- Model, oranı olmayan maçlarda ve sürpriz tespitinde (aşağıda) değer katıyor.
- GLM yalnızca eğitim verisinde en fazla %20'si boş olan sütunları kullanır; LightGBM boşlukları kendisi yönetir.

### Sürpriz alarmı

Sürpriz = **piyasanın favorisinin** (oran yoksa modelin favorisinin) kaybetmesi. Test sezonlarında genel oran %17,8.

| Alarm | Maçların payı | Favori gerçekte kaybetti |
|---|---|---|
| Düşük | %60 | %13 |
| Orta | %25 | %21 |
| Yüksek | %10 | %31 |
| Çok yüksek | %5 | %30 |

İlk 11 ve Avrupa yorgunluğu değişkenleri eklendiğinde geçmiş testte sürpriz yakalama ölçülebilir şekilde artmadı (farklar ±0,01 AUC). Olası nedenler: ESPN'deki oranlar maç saatine yakın oranlar olduğu için piyasa bu bilgiyi zaten fiyatlıyor; ayrıca sürpriz modeli yaklaşık 1.000 maçla eğitiliyor. Değişkenler korunuyor: ana modele zararları yok ve 2.5 alt/üst isabetini %56,0'dan %57,2'ye çıkardılar.

Sürpriz modeli (LightGBM) en riskli %10'luk dilimde sürprizlerin %31'ini yakalıyor; piyasanın kendi olasılıkları aynı dilimde %26 yakalıyor. Yine de en yüksek alarmda bile favori 10 maçın 7'sinde kaybetmiyor: alarm bir risk işaretidir, kesin bir tahmin değildir.

## Value bet (değerli oran)

Bir oran, olayın gerçek olasılığına göre olması gerekenden yüksekse "değerlidir":

- **Beklenen değer (EV)** = olasılık × oran − 1. EV > 0 ise uzun vadede kâr beklenir.
- **Adil oran** = 1 / olasılık.

Araç iki olasılık kaynağı gösterir:

| Kaynak | Ne | Ne için |
|---|---|---|
| **Adil (piyasa + model)** | Bet365/ESPN oranlarının marjı temizlenmiş olasılığı, modelle birleştirilip kalibre edilir | Kendi bahis şirketinizin oranını karşılaştırmak için. Value işareti: oranınız adil oranı en az %5 aşıyorsa |
| **Model (bağımsız)** | Yalnızca istatistik modeli | Bilgi amaçlı; geçmişte zarar ettirdi (aşağıda) |

**Geçmiş test (Bet365/ESPN oranlarıyla, 1 birim sabit bahis, 1.083 maç):**

| Strateji | Görülmemiş sezonlarda ROI | %95 güven aralığı |
|---|---|---|
| 1X2'de her sonuca oyna (referans) | %−11,6 | −17,8 / −5,4 |
| Hep piyasa favorisine oyna | %+6,5 | −0,3 / +13,5 |
| Model (bağımsız), EV > %5 | **%−19,8** | −31,6 / −7,6 |
| Adil (piyasa + model), EV > %5 | %+0,5 | −8,1 / +9,2 |

Ne anlama geliyor:
- **Modelin kendi olasılığıyla value aramak zarar ettiriyor.** Model piyasayla ayrıştığında genelde yanılan modeldir; EV yükseldikçe zarar artıyor.
- **Bet365'e karşı kanıtlanmış bir avantaj yok.** Adil olasılıkla bulunan value'lar başa baş çıktı; güven aralığı sıfırı kapsıyor.
- Süper Lig'de piyasa favorileri biraz hafife alıyor (favori-sürpriz yanlılığı). Kalibrasyon bunu düzeltiyor; value işaretlerinin çoğu bu yüzden favori tarafında çıkar. Bu etki istatistiksel olarak kanıtlanmış değil.
- Gerçek value, **oran karşılaştırmasıyla** bulunur: kullandığınız bahis şirketinin oranı, keskin piyasanın adil oranının belirgin üstündeyse. Oranlarınızı `--oran` ile girin.
- Canlı takip (`tracker.py`) her maçın value seçimini kaydeder ve maçlar bittikçe gerçek kâr/zararı biriktirir.

**İddaa oranları otomatik gelir (`src/nesine.py`).** Maç Nesine'nin maç öncesi bülteninde varsa maç detayındaki oran kutuları ve "Bu Hafta" listesindeki Value sütunu oynanabilir iddaa oranlarıyla doldurulur. Yasal sitelerin (Nesine, iddaa.com, Bilyoner, Misli, Oley) standart oranları aynı olduğu için tek kaynak yeterli. Maç henüz bültende değilse, başladıysa ya da internet yoksa ESPN/Bet365 oranları kullanılır; hangisinin kullanıldığı tablonun altında yazar. "Adil %" her zaman Bet365'ten hesaplanır. İddaa'nın kâr payı yüksek olduğu için (1X2'de ~%15) iddaa oranlarıyla value nadiren çıkar. Günün bütün bültenini CSV olarak almak için: `python oran_topla.py`.

Bu bir kişisel analiz aracıdır; kâr garantisi yoktur. Türkiye'de yalnızca lisanslı (İddaa) bahis yasaldır. Sorumlu oynayın.

## Yapay zekâ katmanı nasıl çalışır

İki motor var; `.env` içindeki `AI_MOTORU` ile seçilir (`otomatik` varsayılan: Claude anahtarı varsa Claude, yoksa yerel model).

| | Yerel model (Ollama) | Claude |
|---|---|---|
| Gereken | Ollama + `qwen3:8b` (kurulu) | `ANTHROPIC_API_KEY` |
| Ücret | Ücretsiz, internet yalnızca haberler için | API kullanım ücreti |
| Haberler | Google Haberler (Türkçe RSS) + ESPN maç haberleri, son 10 gün | Claude web araması + aynı başlıklar |
| Süre | ~20–60 sn (ekran kartında) | 1–2 dk |
| Kalite | Daha zayıf; bazen yorum uydurabilir | Daha güçlü |

Akış:
1. **Haberler:** Maçla ilgili son 10 günün başlıkları toplanır (muhtemel 11, sakat/cezalı, teknik direktör).
2. **Doğrulanmış bilgiler:** Uygulamanın kendi hesapladığı gerekçeler (form, yorgunluk, eksik asıllar, piyasa…) modele "kesin bilgi" olarak verilir; yerel modelin uydurmasını azaltır.
3. **Çıkarım:** Model önemli etkenleri, gol çarpanlarını, gerekçeyi, güven düzeyini ve riskleri JSON olarak döndürür.
4. **Düzeltme:** Çarpanlar **0,85–1,15** ile sınırlanır ve **nihai tahmine** (piyasa + model) oransal uygulanır. Ekranda "önce / sonra" olasılıkları gösterilir.

- Ollama çalışmıyorsa uygulama onu arka planda kendisi başlatır.
- Yapay zekâ düzeltmesi geriye dönük test edilemez (geçmiş maçların haberleri o tarihteki haliyle aranamaz). Bir yorum aracı olarak görün; çarpanlar bu yüzden sınırlı.
- Kullanım: arayüzde maç detayındaki **Yapay zekâ analizi** düğmesi ya da `python src/predict.py Galatasaray Fenerbahçe --yz`.

## Dosyalar

| Dosya | Görev |
|---|---|
| `src/espn_source.py` | ESPN istemcisi (ana kaynak): tüm sezonların sonuçları, maç öncesi oranlar, ilk yarı skoru, kart, penaltı, istatistik, ilk 11 |
| `src/livescore_source.py` | livescoreFootball (worldcup26.ir) istemcisi: güncel sezon skorları, şut, korner, topla oynama, faul, olaylardan ilk yarı skoru, kart ve penaltı |
| `src/api_football.py` | API-Football istemcisi (önbellekli, istek bütçeli): maçlar, istatistikler, oyuncular, olaylar, sakatlıklar, kupa ve Avrupa maçları |
| `src/parse_openfootball.py` | openfootball + API + ek CSV verilerini `data/matches.csv` içinde birleştirir |
| `src/context_data.py` | Takım konumları, seyahat mesafesi, Open-Meteo hava durumu |
| `src/features.py` | Maç öncesi değişkenler (yalnızca geçmiş veri ile) |
| `src/models.py` | Dixon-Coles, skor matrisi, metrikler |
| `src/stacking.py` | Model birleştirme ve kalibrasyon |
| `src/backtest.py` | Walk-forward test, model karşılaştırması, eğitim tablosu |
| `src/predict.py` | Tek maç ve fikstür tahmini |
| `src/ai_analyst.py` | Claude haber araştırması ve yapay zekâ çıkarımı |
| `src/simulate.py` | Monte Carlo sezon simülasyonu |
| `src/upset_model.py` | Sürpriz modeli: favorinin kaybetme olasılığı ve alarm seviyeleri |
| `src/value.py` | Value bet: EV, adil oran, geçmiş kâr/zarar testi |
| `src/nesine.py` | Nesine bülteninden güncel iddaa oranları ve takım adı eşleştirme |
| `oran_topla.py` | Günün bütün futbol bültenini `oranlar.csv` / `tum_oranlar.csv` olarak kaydeder |
| `src/app.py`, `Nostradamus.pyw` | Masaüstü arayüzü ve başlatıcı |
| `src/news.py` | Maç öncesi haber başlıkları (Google Haberler RSS + ESPN) |
| `src/explain.py` | Tahmin gerekçelerini sade cümlelerle üretir |
| `src/tracker.py` | Canlı takip: tahmin kaydı, canlı skor, isabet değerlendirmesi |
| `reports/` | Test sonuçları, tahminler, değişken etkileri, simülasyon |

## Veri kaynakları ve sınırlar

Bu uygulama yalnızca kişisel kullanım içindir; aşağıdaki ücretsiz kaynaklar da bu amaçla kullanılır.

Kaynaklar şu öncelikle birleştirilir: API-Football > ESPN > livescore > openfootball.

- **ESPN Avrupa kupaları:** Şampiyonlar Ligi, Avrupa Ligi, Konferans Ligi ve elemeleri (2017'den bugüne yaklaşık 420 Türk takımı maç günü). Türkiye Kupası ESPN'de yok.
- **ESPN (ana kaynak):** herkese açık, anahtarsız API (`site.api.espn.com`). 2017-18'den bugüne tüm sezonlar (yaklaşık 3.200 maç). Yıl başına tek istekle sonuçlar; maç başına detay isteğiyle **maç öncesi Bet365 / ESPN BET oranları** (1X2, 2.5 alt/üst), ilk yarı skoru, kartlar, penaltılar, ilk 11'ler ve takım istatistikleri gelir. İstatistikler 2017–2021 ve 2024 sonrası için var, 2021–2024 arası boş. İlk tam indirme yaklaşık 40 dakika sürer; sonraki güncellemeler yalnızca yeni maçları indirir. Bir sezonda her ev-deplasman eşleşmesi tek kez oynandığı için eşleştirme takım çiftiyle yapılır; böylece ertelenen maçların tarihleri de güncellenir.

- **openfootball/europe:** skor ve ilk yarı verisi (kamu malı). 2021-22 ve 2022-23 sezonları yok. 2025-26 verisi Kasım 2025'te, 2026-27 verisi Eylül 2026'da kesiliyor.
- **livescoreFootball ([worldcup26.ir](https://worldcup26.ir/football-api), [GitHub](https://github.com/rezarahiminia/livescoreFootball)):** ücretsiz ve anahtarsız; dakikada 120 istek sınırı var. 2026-27 sezonunu 14 Eylül'den itibaren kapsıyor. xG ve geçmiş sezonlar yok. Bireysel bir hobi projesi: hizmet garantisi yok ve lisansı belirtilmemiş, bu yüzden yalnızca kişisel kullanım için uygun. 11–13 Eylül 2026'daki 8 maç hiçbir ücretsiz kaynakta yok; API-Football bu boşluğu kapatır.
- **API-Football:** Ücretsiz plan günde 100 istek verir; bazı sezonlara erişim vermeyebilir (araç uyarır).
- **Open-Meteo:** Ücretsiz. Çok sık istekte geçici sınır uygular; araç bekleyip yeniden dener.
- football-data.co.uk Türkiye'den erişime kapalı.
