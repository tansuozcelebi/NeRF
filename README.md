# KREA NeRF Stüdyo

[![CI](https://github.com/tansuozcelebi/NeRF/actions/workflows/ci.yml/badge.svg)](https://github.com/tansuozcelebi/NeRF/actions/workflows/ci.yml)
[![Deploy](https://github.com/tansuozcelebi/NeRF/actions/workflows/deploy.yml/badge.svg)](https://github.com/tansuozcelebi/NeRF/actions/workflows/deploy.yml)

**Canlı sürüm: <https://tansuozcelebi.github.io/NeRF/>**

Fotoğraflardan **NeRF (Sinirsel Işıma Alanı / Neural Radiance Fields)** eğiten ve hiç
fotoğraflanmamış açılardan görüntü sentezleyen bir React uygulaması. Her şey tarayıcıda çalışır —
sunucu yok, harici bir makine öğrenmesi kütüphanesi yok:

- **eğitim** bir web worker içinde, işlemci üzerinde,
- **yeni açı sentezi** ekran kartında, gerçek zamanlı.

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
4. **Keşfet** — mouse ile tam orbit (döndür, yaklaş, kaydır), derinlik haritası, kırpma kutusu,
   3B model (.ply), PNG ve model ağırlığı dışa aktarımı. Eğitim arka planda sürerken ağırlıklar
   saniyede bir tazelenir ve sahne gözünüzün önünde keskinleşir.

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

## Doğrulama: ezberledi mi, öğrendi mi?

Görüntülerin sekizde biri eğitimden tamamen çıkarılır ve yalnızca ölçüm için kullanılır. Panoda iki
sayı yan yana durur:

- **Eğitim PSNR** — modelin gördüğü karelerdeki keskinlik. Tek başına ezberlemeyi de gösterebilir.
- **Doğrulama PSNR** — modelin *hiç görmediği* karelerdeki keskinlik. Asıl bakılması gereken sayı
  budur.

Aradaki fark aşırı uyumun (overfitting) ölçüsüdür ve grafikte kesikli çizgi olarak izlenebilir. Bir
NeRF eğitim kaybını düşürürken yeni açılardan berbat görüntü üretebilir; bu ayrım olmadan bunu fark
etmenin yolu yoktur. Sekiz kareden az veri varsa ayırma atlanır ve sayı boş kalır.

## 3B model olarak dışa aktarma

NeRF geometriyi yüzey olarak değil sürekli bir yoğunluk fonksiyonu olarak saklar — sisi, saçı ve
camı iyi becermesinin sebebi de, sonucu Blender'da açamamanızın sebebi de budur. Keşfet sekmesindeki
dışa aktarma bu köprüyü kurar:

1. Yoğunluk alanı düzenli bir ızgarada örneklenir (64³–160³),
2. yoğunluğun eşiği aştığı yüzey üçgenlere çevrilir,
3. her köşe, renk ağına "bu noktaya dik baksam ne görürdüm" diye sorularak boyanır,
4. sonuç renkli **PLY** olarak inilir — Blender, MeshLab, CloudCompare ve Houdini doğrudan okur.

Eşik otomatik seçilir (örneklerin yüksek bir yüzdeliği), çünkü yoğunluk sınırsız bir büyüklüktür ve
ölçeği sahneden sahneye değişir; kaydırıcı bu öneriyi ölçekler.

Yüzey çıkarımı **marching cubes değil marching tetrahedra** kullanır. Marching cubes 256×16'lık bir
üçgen tablosu ister ve yanlış yazılmış tek bir satır mesh'te deliktir. Her küpü altı dörtyüzlüye
bölmek tabloyu tamamen ortadan kaldırır: dörtyüzlünün yalnızca dört köşesi vardır, üç durumu da kod
içinde türetilebilir. Her küp aynı köşegen etrafında aynı biçimde bölündüğü için komşu küpler ortak
yüzlerini aynı yerden keser ve sonuç su geçirmezdir. Örneklenen kutunun duvarına dayanan yüzeyler de
kapatılır, böylece dışa aktarılan model kapalı bir katıdır.

## Kırpma kutusu

NeRF'ler neredeyse her zaman artık bırakır: yeterince fotoğrafın anlaşamadığı boşluklarda yüzen yarı
saydam lekeler. Kırpma bunun standart temizliğidir — bedava, yeniden eğitim istemez ve çoğu zaman
"nesne" ile "bulut içindeki nesne" arasındaki farktır. Kutu hem görüntüleyiciye hem de dışa
aktarılan modele uygulanır; zeminden kurtulmak için genellikle Y alt sınırını yükseltmek yeterlidir.

## Ekran kartında gerçek zamanlı görüntüleme

Eğitilmiş model — hash tablosu, iki MLP ve doluluk ızgarası — dokulara yüklenir ve bir **fragment
shader** her piksel için hacmi yeniden tarar. Sahne tek bir tam ekran dörtgeninden ibarettir; işin
tamamı shader'da yapılır. Three.js burada gerçekten iyi olduğu iş için kullanılır: WebGL bağlamını
yönetmek, dokuları yüklemek ve düzgün bir yörünge kamerası (`OrbitControls`) vermek.

Sonuç, işlemci yolunda kare başına saniyeler süren bir işlemin serbestçe sürüklenebilir hâle
gelmesidir. Görüntüleyici kare aralığını ölçüp çözünürlüğü kendisi ayarlar; yavaş bir makinede
piksel sayısını düşürür, hızlı bir makinede tam çözünürlüğe çıkar.

WebGL2 bulunmayan tarayıcılarda uygulama sessizce işlemci görüntüleyicisine düşer ve bunu söyler.
WebGL'in yazılımla (SwiftShader, llvmpipe) emüle edildiği durumlar da tanınır; orada shader doğru
çalışır ama çok yavaştır, bu yüzden çözünürlük ve örnek sayısı düşük başlatılır.

**Eğitim hâlâ işlemcide yürür.** Geri yayılım, hash tablosuna dağınık gradyan biriktirmeyi
gerektirir; bu WebGL2'de güvenilir biçimde yapılamaz (compute shader ve atomik float toplama
yoktur). GPU'ya taşınan kısım, gösterimi asıl yavaşlatan kısımdır: görüntü üretimi.

### GPU ile CPU aynı sonucu vermek zorunda

Shader, hash kodlamasının ve iki MLP'nin ikinci bir bağımsız uygulamasıdır. İkisinin zamanla
birbirinden ayrılmasını engelleyen tek şey, aynı modeli aynı kameradan iki yolla da üretip
pikselleri karşılaştıran bir denetimdir:

```bash
npm run dev        # sonra /dev/parity.html adresini açın
```

Bu sayfa, rastgele ağırlıklı ve eğitilmiş iki durumda karşılaştırma yapar. Ölçülen fark: rastgele
ağırlıklarda **tam olarak 0**, eğitilmiş modelde en fazla **1/255**.

### Dosya haritası

| Dosya | Sorumluluk |
| --- | --- |
| `src/nerf/hashGrid.ts` | Instant-NGP tarzı çok çözünürlüklü hash kodlaması, seyrek gradyan biriktirme |
| `src/nerf/mlp.ts` | Toplu tam bağlantılı katman, elle yazılmış ileri/geri geçiş |
| `src/nerf/sphericalHarmonics.ts` | Bakış yönünün küresel harmoniklerle kodlanması (görüş bağımlılığı) |
| `src/nerf/field.ts` | Yoğunluk + renk ağlarının birleşimi, ağırlık dışa/içe aktarımı |
| `src/nerf/volumeRender.ts` | Işın örnekleme, hacimsel harmanlama ve analitik geri yayılımı |
| `src/nerf/occupancy.ts` | Boş alan atlama ızgarası |
| `src/nerf/meshExtract.ts` | Marching tetrahedra ile yüzey çıkarımı, sınır kapatma |
| `src/nerf/trainer.ts` | Işın seçimi, kayıp, optimizasyon adımı, yeni açı üretimi |
| `src/nerf/camera.ts` | Poz matrisleri, ışın geometrisi, yörünge üreteçleri |
| `src/nerf/syntheticScene.ts` | Demo sahnesini üreten klasik ışın izleyici |
| `src/gpu/nerfShader.ts` | Modelin şekli gömülerek üretilen GLSL: hash araması, iki MLP, hacim harmanlama |
| `src/gpu/GpuNerfRenderer.ts` | Three.js tabanlı gerçek zamanlı görüntüleyici, doku yükleme, yörünge kamerası |
| `src/worker/` | Eğitim worker'ı ve mesaj sözleşmesi |
| `src/utils/ply.ts` | Renkli PLY yazıcı (ikili) |
| `src/components/`, `src/hooks/` | React arayüzü |
| `dev/parity.html` | GPU ile CPU çıktısını karşılaştıran geliştirme denetimi (üretim derlemesinde yer almaz) |

### Neden bu tasarım tercihleri?

- **Hash kodlaması** sayesinde ağ küçük kalabiliyor; eğitim dakikalar yerine saniyeler mertebesinde
  ilerliyor.
- **Boş alan atlama ızgarası** hacmin çoğunun hava olduğunu öğrenip oradaki örnekleri atlıyor;
  tipik olarak 3–6 kat hızlanma sağlıyor.
- **Cauchy seyreklik cezası** yarı saydam "hayalet" birikintilerini bastırıyor; bunlar eğitim
  görüntülerinde iyi görünüp yeni açılarda dağılan tipik NeRF hatasıdır.
- **Uyarlanabilir çözünürlük**: kare aralığı ölçülüp piksel sayısı buna göre ayarlanıyor. Ölçüt
  bilerek kare aralığıdır, çizim çağrısının süresi değil — WebGL komutları kuyruğa atıp hemen
  döndüğü için çizim süresi shader ne kadar ağır olursa olsun birkaç milisaniye görünür.

## Başarım

Eğitim, tek çekirdekli işlemci üzerinde ölçülen kabaca değerler (dengeli ön ayar, 512 ışın × 32
örnek):

| Ölçüm | Değer |
| --- | --- |
| Eğitim hızı | ~5–7 adım/sn |
| Tanınabilir sonuç | ~200–400 adım |
| Belirgin şekilde keskin sonuç | ~1000–3000 adım |
| Parametre sayısı | ~790 bin |

`Hızlı` ön ayarı yaklaşık iki kat hızlıdır, `Kaliteli` ön ayarı daha yavaş ama daha detaylıdır.

Görüntüleme tarafında işlemci yolu 112² bir kareyi ~0,3–1 saniyede üretir; GPU yolu aynı işi
paralel yaptığı için kıyaslanamayacak kadar hızlıdır ve kamera serbestçe sürüklenebilir. Kesin
kare hızı ekran kartına bağlıdır, bu yüzden bir rakam vermiyoruz: görüntüleyici kare aralığını
kendisi ölçüp çözünürlüğü ayarlar ve o anki değeri sol üstteki rozette gösterir.

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
- **Doğrulama ayrımı** — ayrılan karelerin eğitime hiç karışmadığı, az veride ayırmanın atlandığı ve
  bu karelerdeki hatanın gerçekten düştüğü kontrol edilir.
- **Yüzey çıkarımı** — cevabı tam olarak bilinen bir şekle, küreye karşı sınanır: her köşe küre
  üzerinde mi, her kenar tam iki üçgen tarafından mı paylaşılıyor (su geçirmezlik), yüzler dışarı mı
  bakıyor ve kapanan hacim analitik değere uyuyor mu.
- **Gerileme testi** — doluluk ızgarasının hacmin tamamını budayıp eğitimi kalıcı olarak
  öldürmediği kontrol edilir.

Testler, tip denetimi ve üretim derlemesi her itmede GitHub Actions üzerinde Node 20 ve 22 ile
çalışır (`.github/workflows/ci.yml`).

## Yayın

`main` dalına giren her değişiklik, **CI yeşil bittikten sonra** GitHub Pages'e yayınlanır
(`.github/workflows/deploy.yml`). Testleri yayın akışında tekrar koşmak yerine CI'ın sonucunu
beklemek, aynı işi iki kez yapmadan bozuk bir derlemenin yayına çıkmasını engeller.

Uygulama tamamen statiktir — sunucu tarafı yoktur, eğitim ve görüntüleme ziyaretçinin kendi
tarayıcısında çalışır. Varlıklar göreli yolla istendiği için (`vite.config.ts` içinde
`base: './'`) site `/NeRF/` alt dizininde sorunsuz çalışır; bu, alt dizine yayınlanan tek
sayfalık uygulamalarda en sık kırılan yerdir ve derleme bu koşulda tarayıcıda sınanmıştır.

Elle yayın gerekirse Actions sekmesinden **Deploy** akışı `workflow_dispatch` ile çalıştırılabilir.

### Bir kereye mahsus kurulum

Pages'in depoda açık olması gerekir. Akıştaki `configure-pages` adımı bunu kendisi açmayı
dener, ancak `GITHUB_TOKEN` site oluşturmaya yetkili değilse adım şu hatayla düşer:

```
Create Pages site failed. Error: Resource not accessible by integration
```

Bu durumda depo ayarlarından **Settings → Pages → Build and deployment → Source: GitHub
Actions** seçilir; ardından Actions sekmesinden **Deploy** akışı bir kez elle çalıştırılır.
Bu adım yalnızca ilk yayında gerekir — site açıldıktan sonra `main`'e giren her değişiklik
kendiliğinden yayınlanır.

## Sınırlar

- Eğitim işlemci üzerinde yürür; bu yüzden eğitim çözünürlükleri küçük (48–160 piksel) tutulmuştur.
- Gerçek zamanlı görüntüleme WebGL2 ister. Yoksa uygulama işlemci görüntüleyicisine düşer.
- Kamera pozları ya varsayılır ya da dışarıdan içe aktarılır (yukarıdaki uyarıya bakın).
- Arka plan tek bir sabit renkle modellenir; karmaşık arka planlı çekimlerde özneyi izole etmek
  daha iyi sonuç verir.
