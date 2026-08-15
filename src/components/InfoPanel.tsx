const STEPS = [
  {
    title: 'Fotoğraf toplama',
    body:
      'Nesnenin veya odanın etrafında çekilmiş çok sayıda 2B fotoğraf sisteme yüklenir. ' +
      'Her fotoğrafın hangi konumdan çekildiği de bilinmelidir.',
  },
  {
    title: 'Sinir ağı eğitimi',
    body:
      'Ağ, her fotoğrafın piksellerinden geçen ışınları izleyerek uzaydaki noktaların renk ve ' +
      'yoğunluk değerlerini öğrenir. Tahmin ile gerçek piksel arasındaki fark küçüldükçe sahne netleşir.',
  },
  {
    title: 'Hacimsel modelleme',
    body:
      'Uzaydaki her noktanın yoğunluğu ve yaydığı ışık hesaplanır. Klasik 3B modellemeden farklı ' +
      'olarak yüzey değil, hacmin tamamı temsil edilir; bu yüzden saydamlık ve parlama doğal görünür.',
  },
  {
    title: 'Yeni açı sentezi',
    body:
      'Daha önce hiç fotoğraflanmamış bir açıdan bile, o an kamerayla çekilmiş gibi görüntü üretilir. ' +
      'NeRF’in asıl marifeti budur.',
  },
]

const USES = [
  { title: 'Oyun ve sinema', body: 'Gerçekçi dijital ortamlar, karakterler ve görsel efektler (VFX).' },
  { title: 'VR / AR', body: 'Gerçek dünyadaki mekânların dijital dünyaya birebir taşınması.' },
  { title: 'Robotik ve haritalama', body: 'Robotların çevrelerini 3 boyutlu algılaması.' },
  { title: 'E-ticaret', body: 'Ürünlerin tek bir telefon çekimiyle her yönden incelenebilen 3B modelleri.' },
]

export function InfoPanel() {
  return (
    <div className="info">
      <p className="info-lede">
        <strong>NeRF (Sinirsel Işıma Alanı / Neural Radiance Fields)</strong>, farklı açılardan
        çekilmiş 2 boyutlu fotoğrafları kullanarak gerçeğe yakın 3 boyutlu sahneler ve nesneler
        oluşturan bir yapay zekâ tekniğidir. Klasik 3B modellemeden farklı olarak nesnenin içini ve
        ışık alımını öğrenir, böylece yepyeni açılardan tutarlı görüntüler üretir.
      </p>

      <h3>Nasıl çalışır?</h3>
      <ol className="info-steps">
        {STEPS.map((step, i) => (
          <li key={step.title}>
            <span className="info-step-number">{i + 1}</span>
            <div>
              <strong>{step.title}</strong>
              <p>{step.body}</p>
            </div>
          </li>
        ))}
      </ol>

      <h3>Nerelerde kullanılır?</h3>
      <ul className="info-uses">
        {USES.map((use) => (
          <li key={use.title}>
            <strong>{use.title}</strong>
            <span>{use.body}</span>
          </li>
        ))}
      </ul>

      <h3>Bu uygulamada ne var?</h3>
      <p>
        Buradaki NeRF gerçek bir uygulamadır, hazır bir kütüphane çağrısı değil: çok çözünürlüklü
        hash kodlaması, küçük bir sinir ağı, hacimsel ışın izleme ve elle türetilmiş geri yayılım
        tamamen tarayıcıda, bir web worker içinde çalışır. Eğitim CPU üzerinde yürür; bu yüzden
        çözünürlükler küçük tutulmuştur.
      </p>
      <p className="info-caveat">
        <strong>Dürüst uyarı:</strong> NeRF eğitimi için her fotoğrafın kamera konumu bilinmelidir.
        Bunu rastgele fotoğraflardan çıkarmak (structure-from-motion) tarayıcıda yapılabilecek bir
        iş değildir. Bu yüzden uygulama ya konumları <em>varsayar</em> (özne etrafında düzenli bir
        halka/kubbe çekimi), ya da COLMAP gibi bir araçtan gelen <code>transforms.json</code>{' '}
        dosyasını okur. Çekiminiz varsayıma uymuyorsa sonuç bulanık çıkar — bu bir hata değil,
        yöntemin sınırıdır. Yöntemin kendisini net bir şekilde görmek için sentetik demo sahnesini
        kullanın: orada kamera konumları tanım gereği kusursuzdur.
      </p>
    </div>
  )
}
