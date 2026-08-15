# NeRF Stüdyo

[![CI](https://github.com/tansuozcelebi/NeRF/actions/workflows/ci.yml/badge.svg)](https://github.com/tansuozcelebi/NeRF/actions/workflows/ci.yml)

Fotoğraflardan **NeRF (Sinirsel Işıma Alanı / Neural Radiance Fields)** eğiten ve hiç
fotoğraflanmamış açılardan görüntü sentezleyen bir React uygulaması. Eğitim de, görüntü üretimi de
tamamen tarayıcıda, bir web worker içinde çalışır — sunucu yok, GPU zorunluluğu yok, harici bir
makine öğrenmesi kütüphanesi yok.

NeRF, farklı açılardan çekilmiş 2 boyutlu fotoğrafları kullanarak gerçeğe yakın 3 boyutlu sahneler
oluşturan bir yapay zekâ tekniğidir. Klasik 3B modellemeden farklı olarak yüzey değil **hacim**
öğrenilir: uzaydaki her noktanın yoğunluğu ve o noktadan hangi yöne ne renk ışık yayıldığı
modellenir. Bu sayede saydamlık, parlama ve bakış açısına göre değişen yansımalar doğal görünür.

## Hızlı başlangıç

```bash
npm install
npm run dev      # geliştirme sunucusu
npm run build    # üretim derlemesi
npm test         # birim ve uçtan uca testler
```

Uygulamayı açtığınızda dört adım sizi karşılar:

1. **Kaynak** — sentetik demo sahnesi ya da kendi fotoğraflarınız.
2. **Kameralar** — her fotoğrafın hangi konumdan çekildiği.
3. **Eğitim** — kayıp/PSNR grafiği ve canlı önizleme ile ağın eğitimi.
4. **Keşfet** — sürükleyip döndürerek yeni açılardan görüntü sentezi, derinlik haritası, PNG ve
   model ağırlığı dışa aktarımı.

İlk denemede **sentetik demo sahnesini** kullanın: orada kamera konumları tanım gereği kusursuz
olduğu için yöntemin ne yapabildiğini en net şekilde gösterir.

## Dürüst uyarı: kamera konumları

NeRF eğitimi için her fotoğrafın **kamera konumunun bilinmesi** gerekir. Bunu rastgele
fotoğraflardan çıkarmak (structure-from-motion) tarayıcıda yapılabilecek bir iş değildir. Uygulama
bu yüzden üç seçenek sunar ve hangisini kullandığını saklamaz:

| Seçenek | Ne yapar |
| --- | --- |
| `halka` | Fotoğrafların özne etrafında sabit yükseklikte, eşit aralıklı çekildiğini **varsayar**. |
| `kubbe` | Aynı varsayım, ancak yükseklik dizi boyunca değişir. |
| `dosya` | COLMAP / Instant-NGP / NeRF Studio çıktısı olan `transforms.json` dosyasını okur. |

Çekiminiz varsayıma uymuyorsa sonuç bulanık çıkar. Bu bir hata değil, yöntemin sınırıdır — gerçek
pozlar için masaüstünde bir SfM aracı çalıştırıp `transforms.json` dosyasını içe aktarın.

İyi bir çekim için: özneyi ortada tutun, etrafında düzenli adımlarla dönün, 20–60 kare çekin,
odak ve pozlamayı sabitleyin, sahneyi hareket ettirmeyin.

## Nasıl çalışıyor?

Uygulamanın çekirdeği hazır bir kütüphane çağrısı değil; NeRF hattının tamamı bu depoda yazılıdır.

```
konum ─▶ çok çözünürlüklü hash kodlaması ─▶ yoğunluk MLP ─▶ yoğunluk σ + geometri özniteliği
                                                              │
              bakış yönü ─▶ küresel harmonikler ──────────────┴──▶ renk MLP ─▶ RGB
```

Işın boyunca alınan örnekler önden arkaya harmanlanır:

```
alfa_i = 1 − exp(−σ_i · δ_i)
w_i    = T_i · alfa_i,   T_i = Π_{j<i} (1 − alfa_j)
C      = Σ_i w_i · c_i + T_son · arka plan
```

Bu ifadenin türevi elle çıkarılmıştır; otomatik türev kütüphanesi kullanılmaz.

### Dosya haritası

| Dosya | Sorumluluk |
| --- | --- |
| `src/nerf/hashGrid.ts` | Instant-NGP tarzı çok çözünürlüklü hash kodlaması, seyrek gradyan biriktirme |
| `src/nerf/mlp.ts` | Toplu tam bağlantılı katman, elle yazılmış ileri/geri geçiş |
| `src/nerf/sphericalHarmonics.ts` | Bakış yönünün küresel harmoniklerle kodlanması (görüş bağımlılığı) |
| `src/nerf/field.ts` | Yoğunluk + renk ağlarının birleşimi, ağırlık dışa/içe aktarımı |
| `src/nerf/volumeRender.ts` | Işın örnekleme, hacimsel harmanlama ve analitik geri yayılımı |
| `src/nerf/occupancy.ts` | Boş alan atlama ızgarası |
| `src/nerf/trainer.ts` | Işın seçimi, kayıp, optimizasyon adımı, yeni açı üretimi |
| `src/nerf/camera.ts` | Poz matrisleri, ışın geometrisi, yörünge üreteçleri |
| `src/nerf/syntheticScene.ts` | Demo sahnesini üreten klasik ışın izleyici |
| `src/worker/` | Eğitim worker'ı ve mesaj sözleşmesi |
| `src/components/`, `src/hooks/` | React arayüzü |

### Neden bu tasarım tercihleri?

- **Hash kodlaması** sayesinde ağ küçük kalabiliyor; eğitim dakikalar yerine saniyeler mertebesinde
  ilerliyor.
- **Boş alan atlama ızgarası** hacmin çoğunun hava olduğunu öğrenip oradaki örnekleri atlıyor;
  tipik olarak 3–6 kat hızlanma sağlıyor.
- **Cauchy seyreklik cezası** yarı saydam "hayalet" birikintilerini bastırıyor; bunlar eğitim
  görüntülerinde iyi görünüp yeni açılarda dağılan tipik NeRF hatasıdır.
- **Kademeli görüntüleme**: kamerayı sürüklerken kaba kare hemen basılıyor, kamera durunca daha
  keskin geçişler onu değiştiriyor.

## Başarım

Tek çekirdekli CPU üzerinde ölçülen kabaca değerler (dengeli ön ayar, 512 ışın × 32 örnek):

| Ölçüm | Değer |
| --- | --- |
| Eğitim hızı | ~5–7 adım/sn |
| Tanınabilir sonuç | ~200–400 adım |
| Belirgin şekilde keskin sonuç | ~1000–3000 adım |
| Yeni açı karesi (112²) | ~0,3–1 sn |
| Parametre sayısı | ~790 bin |

`Hızlı` ön ayarı yaklaşık iki kat hızlıdır, `Kaliteli` ön ayarı daha yavaş ama daha detaylıdır.

## Testler

```bash
npm test
```

- **Gradyan denetimleri** — hash kodlaması, MLP katmanları, hacimsel harmanlama ve tüm ağ için
  sonlu farklarla sayısal karşılaştırma. Elle türetilmiş her türev burada doğrulanır.
- **Kamera testleri** — poz matrisleri, ışın yönleri, kutu kesişimi, poz normalizasyonu.
- **Uçtan uca eğitim** — sentetik sahnede eğitim yapılır ve **eğitimde görülmemiş** bir kamera
  açısındaki hatanın gerçekten düştüğü doğrulanır (yalnızca eğitim kaybının düşmesi yeterli
  değildir).
- **Gerileme testi** — doluluk ızgarasının hacmin tamamını budayıp eğitimi kalıcı olarak
  öldürmediği kontrol edilir.

Testler, tip denetimi ve üretim derlemesi her itmede GitHub Actions üzerinde Node 20 ve 22 ile
çalışır (`.github/workflows/ci.yml`).

## Sınırlar

- Eğitim CPU üzerinde yürür; bu yüzden çözünürlükler küçük (48–160 piksel) tutulmuştur.
- Kamera pozları ya varsayılır ya da dışarıdan içe aktarılır (yukarıdaki uyarıya bakın).
- Arka plan tek bir sabit renkle modellenir; karmaşık arka planlı çekimlerde özneyi izole etmek
  daha iyi sonuç verir.
