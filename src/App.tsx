import { useCallback, useMemo, useState } from 'react'
import { CameraSetup } from './components/CameraSetup'
import { InfoPanel } from './components/InfoPanel'
import { PhotoUploader } from './components/PhotoUploader'
import { TrainingPanel } from './components/TrainingPanel'
import { Viewer } from './components/Viewer'
import { useNerfWorker } from './hooks/useNerfWorker'
import { focalFromFov } from './nerf/camera'
import type { QualityPreset } from './nerf/types'
import { buildPoses, DEFAULT_POSE_CONFIG, type PoseConfig } from './state/poseConfig'
import { averageColor, cssToRgb, paintCanvas, rgbToCss, type DecodedPhoto } from './utils/image'
import type { ImportedPoses } from './utils/transforms'
import type { SerializedView } from './worker/protocol'

type Source = 'sentetik' | 'foto'
type Step = 'kaynak' | 'kamera' | 'egitim' | 'kesfet' | 'bilgi'

const STEPS: Array<{ id: Step; label: string }> = [
  { id: 'kaynak', label: '1 · Kaynak' },
  { id: 'kamera', label: '2 · Kameralar' },
  { id: 'egitim', label: '3 · Eğitim' },
  { id: 'kesfet', label: '4 · Keşfet' },
  { id: 'bilgi', label: 'NeRF nedir?' },
]

export default function App() {
  const nerf = useNerfWorker()
  const [step, setStep] = useState<Step>('kaynak')
  const [source, setSource] = useState<Source>('sentetik')
  const [preset, setPreset] = useState<QualityPreset>('dengeli')

  const [photos, setPhotos] = useState<DecodedPhoto[]>([])
  const [trainingResolution, setTrainingResolution] = useState(96)
  const [poseConfig, setPoseConfig] = useState<PoseConfig>(DEFAULT_POSE_CONFIG)
  const [imported, setImported] = useState<ImportedPoses | null>(null)
  const [backgroundCss, setBackgroundCss] = useState('#101018')

  const [syntheticViews, setSyntheticViews] = useState(36)
  const [syntheticResolution, setSyntheticResolution] = useState(96)
  const syntheticRadius = 3.6
  const syntheticFov = 42

  const photoNames = useMemo(() => photos.map((p) => p.name), [photos])
  const built = useMemo(
    () => buildPoses(poseConfig, photoNames, imported),
    [poseConfig, photoNames, imported],
  )

  const setPhotosAndBackground = useCallback((next: DecodedPhoto[]) => {
    setPhotos(next)
    if (next.length > 0) setBackgroundCss(rgbToCss(averageColor(next)))
  }, [])

  const prepare = useCallback(() => {
    if (source === 'sentetik') {
      nerf.initSynthetic(preset, {
        viewCount: syntheticViews,
        resolution: syntheticResolution,
        radius: syntheticRadius,
        fovDegrees: syntheticFov,
      })
      setStep('egitim')
      return
    }

    if (built.poses.length !== photos.length || photos.length === 0) return
    const views: SerializedView[] = photos.map((photo, i) => ({
      id: photo.name,
      width: photo.width,
      height: photo.height,
      // Copies: these buffers are transferred to the worker and detached here.
      pixels: new Float32Array(photo.pixels),
      pose: new Float32Array(built.poses[i]),
      focal: focalFromFov((poseConfig.fovDegrees * Math.PI) / 180, photo.width),
      cx: photo.width / 2,
      cy: photo.height / 2,
    }))
    nerf.initPhotos(preset, views, poseConfig.aabbSize, cssToRgb(backgroundCss))
    setStep('egitim')
  }, [
    source, nerf, preset, syntheticViews, syntheticResolution, built.poses, photos,
    poseConfig.fovDegrees, poseConfig.aabbSize, backgroundCss,
  ])

  const canPrepare =
    source === 'sentetik' || (photos.length > 0 && built.poses.length === photos.length)
  const modelReady = nerf.status === 'hazir' || nerf.status === 'egitiliyor'
  const viewerFov = source === 'sentetik' ? syntheticFov : poseConfig.fovDegrees
  const viewerRadius = source === 'sentetik' ? syntheticRadius : poseConfig.radius

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>NeRF Stüdyo</h1>
          <p>Fotoğraflardan sinirsel ışıma alanı eğitin, hiç çekilmemiş açılardan görüntü üretin.</p>
        </div>
        <span className={`status status--${nerf.status}`}>{statusLabel(nerf.status)}</span>
      </header>

      <nav className="stepper">
        {STEPS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`stepper-item ${step === item.id ? 'stepper-item--active' : ''}`}
            onClick={() => setStep(item.id)}
            disabled={(item.id === 'egitim' || item.id === 'kesfet') && !modelReady}
          >
            {item.label}
          </button>
        ))}
      </nav>

      {nerf.error && <p className="error-banner">{nerf.error}</p>}

      <main className="panel">
        {step === 'kaynak' && (
          <section>
            <h2>Kaynak seçin</h2>
            <div className="source-choice">
              <SourceCard
                active={source === 'sentetik'}
                onClick={() => setSource('sentetik')}
                title="Sentetik demo sahne"
                body="Uygulama sahneyi kendisi üretir, dolayısıyla kamera konumları tam olarak bilinir. NeRF'in ne yapabildiğini görmenin en net yolu."
              />
              <SourceCard
                active={source === 'foto'}
                onClick={() => setSource('foto')}
                title="Kendi fotoğraflarım"
                body="Bir nesnenin etrafını dolaşarak çektiğiniz fotoğrafları yükleyin. Kamera konumları varsayılır veya transforms.json ile içe aktarılır."
              />
            </div>

            {source === 'sentetik' ? (
              <div className="synthetic-options">
                <NumberField
                  label="Görüntü sayısı"
                  value={syntheticViews}
                  min={8}
                  max={80}
                  step={4}
                  onChange={setSyntheticViews}
                  hint="Daha fazla açı, daha tutarlı bir 3B yeniden yapılandırma demek."
                />
                <NumberField
                  label="Eğitim çözünürlüğü"
                  value={syntheticResolution}
                  min={48}
                  max={160}
                  step={16}
                  onChange={setSyntheticResolution}
                  hint="Piksel cinsinden kenar uzunluğu. Yükseltmek detayı artırır, eğitimi yavaşlatır."
                />
              </div>
            ) : (
              <>
                <div className="synthetic-options">
                  <NumberField
                    label="Eğitim çözünürlüğü"
                    value={trainingResolution}
                    min={48}
                    max={160}
                    step={16}
                    onChange={setTrainingResolution}
                    hint="Fotoğraflar bu uzun kenara küçültülür. Sonraki yüklemeler için geçerlidir."
                  />
                  <div className="field">
                    <label htmlFor="bg">Arka plan rengi</label>
                    <input
                      id="bg"
                      type="color"
                      value={backgroundCss}
                      onChange={(e) => setBackgroundCss(e.target.value)}
                    />
                    <p className="hint">
                      Hacmin arkasına konan sabit renk. Fotoğraf kenarlarından tahmin edildi.
                    </p>
                  </div>
                </div>
                <PhotoUploader
                  photos={photos}
                  onChange={setPhotosAndBackground}
                  trainingResolution={trainingResolution}
                />
              </>
            )}

            <div className="button-row">
              <button
                type="button"
                className="btn btn--primary"
                disabled={!canPrepare || nerf.status === 'hazirlaniyor'}
                onClick={() => (source === 'sentetik' ? prepare() : setStep('kamera'))}
              >
                {source === 'sentetik' ? 'Sahneyi üret ve modeli kur' : 'Kamera ayarına geç'}
              </button>
            </div>
          </section>
        )}

        {step === 'kamera' && (
          <section>
            <h2>Kameralar nerede duruyordu?</h2>
            {source === 'sentetik' ? (
              <p className="hint">
                Sentetik sahnede kamera konumları uygulamanın kendisi tarafından belirlendiği için
                kesindir; ayarlanacak bir şey yok.
              </p>
            ) : (
              <>
                <CameraSetup
                  config={poseConfig}
                  onChange={setPoseConfig}
                  poses={built.poses}
                  note={built.note}
                  problem={built.problem}
                  imported={imported}
                  onImport={setImported}
                />
                <div className="button-row">
                  <button type="button" className="btn" onClick={() => setStep('kaynak')}>
                    Geri
                  </button>
                  <button
                    type="button"
                    className="btn btn--primary"
                    disabled={!canPrepare || nerf.status === 'hazirlaniyor'}
                    onClick={prepare}
                  >
                    Modeli kur
                  </button>
                </div>
              </>
            )}
          </section>
        )}

        {step === 'egitim' && (
          <section>
            <h2>Eğitim</h2>
            <TrainingPanel
              nerf={nerf}
              preset={preset}
              onPresetChange={setPreset}
              onRebuild={prepare}
            />
            <div className="thumb-strip">
              {nerf.thumbnails.slice(0, 24).map((thumb) => (
                <ThumbCanvas key={thumb.id} thumb={thumb} />
              ))}
            </div>
            <div className="button-row">
              <button
                type="button"
                className="btn btn--primary"
                disabled={!nerf.stats}
                onClick={() => setStep('kesfet')}
              >
                Yeni açıları keşfet
              </button>
            </div>
          </section>
        )}

        {step === 'kesfet' && (
          <section>
            <h2>Yeni açı sentezi</h2>
            <Viewer nerf={nerf} fovDegrees={viewerFov} radius={viewerRadius} />
            <ExportRow nerf={nerf} />
          </section>
        )}

        {step === 'bilgi' && (
          <section>
            <h2>NeRF nedir?</h2>
            <InfoPanel />
          </section>
        )}
      </main>

      <footer className="app-footer">
        Eğitim ve görüntü sentezi tamamen tarayıcıda, bir web worker içinde CPU üzerinde çalışır.
      </footer>
    </div>
  )
}

function statusLabel(status: string): string {
  switch (status) {
    case 'hazirlaniyor':
      return 'hazırlanıyor'
    case 'hazir':
      return 'hazır'
    case 'egitiliyor':
      return 'eğitiliyor'
    case 'hata':
      return 'hata'
    default:
      return 'boşta'
  }
}

function SourceCard({
  active, onClick, title, body,
}: {
  active: boolean
  onClick: () => void
  title: string
  body: string
}) {
  return (
    <button
      type="button"
      className={`source-card ${active ? 'source-card--active' : ''}`}
      onClick={onClick}
      aria-pressed={active}
    >
      <strong>{title}</strong>
      <span>{body}</span>
    </button>
  )
}

function NumberField({
  label, value, min, max, step, onChange, hint,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
  hint?: string
}) {
  return (
    <div className="field field--slider">
      <label>
        {label}
        <output>{value}</output>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <p className="hint">{hint}</p>}
    </div>
  )
}

function ThumbCanvas({ thumb }: { thumb: { width: number; height: number; rgba: Uint8ClampedArray } }) {
  return (
    <canvas
      className="strip-canvas"
      width={thumb.width}
      height={thumb.height}
      ref={(canvas) => {
        if (canvas) paintCanvas(canvas, thumb.rgba, thumb.width, thumb.height)
      }}
    />
  )
}

function ExportRow({ nerf }: { nerf: ReturnType<typeof useNerfWorker> }) {
  const [busy, setBusy] = useState(false)

  const download = async () => {
    setBusy(true)
    try {
      const { data, step, paramCount } = await nerf.exportWeights()
      const blob = new Blob([data.buffer as ArrayBuffer], { type: 'application/octet-stream' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `nerf-agirliklar-adim${step}-${paramCount}param.bin`
      link.click()
      URL.revokeObjectURL(url)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="button-row">
      <button type="button" className="btn" disabled={busy} onClick={() => void download()}>
        {busy ? 'Hazırlanıyor…' : 'Model ağırlıklarını indir (.bin)'}
      </button>
      <span className="hint">
        Float32 ham ağırlıklar: hash tablosu ve ardından dört katmanın parametreleri.
      </span>
    </div>
  )
}
