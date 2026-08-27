/**
 * Real-time novel-view explorer running on the GPU.
 *
 * The trained weights are pushed into textures and the fragment shader marches
 * the volume per pixel, so the camera can be dragged around freely instead of
 * waiting seconds per frame. While training continues in the background the
 * weights are re-uploaded periodically, and the scene visibly sharpens as you
 * watch it.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { describeWebglSupport, GpuNerfRenderer, type RenderMode } from '../gpu/GpuNerfRenderer'
import type { NerfWorkerApi } from '../hooks/useNerfWorker'
import { downloadCanvasPng } from '../utils/download'
import { paintCanvas } from '../utils/image'

interface Props {
  nerf: NerfWorkerApi
  fovDegrees: number
  radius: number
  /** World-space box the renderer traverses; trims floaters out of the view. */
  crop?: { min: [number, number, number]; max: [number, number, number] }
}

/** How often the viewer pulls fresh weights while training is running. */
const SNAPSHOT_INTERVAL_MS = 1500
/**
 * Frame-interval band the adaptive resolution aims for.
 *
 * The measurement has to be the gap between animation frames, not how long the
 * draw call takes: WebGL queues commands and returns immediately, so timing
 * `render()` reports a millisecond or two no matter how heavy the shader is.
 * Adapting on that number would ratchet the resolution up until the pipeline is
 * hopelessly backed up. The rAF interval, in contrast, reflects real throughput
 * once the GPU is the bottleneck.
 */
const FAST_FRAME_MS = 20
const SLOW_FRAME_MS = 40
const MIN_SCALE = 0.12
const MAX_SCALE = 1
/**
 * Deliberately timid opening settings. The very first frame is drawn before we
 * know anything about the machine, and on a weak GPU — or a software GL
 * implementation — a full-resolution frame of this shader can take long enough
 * to lock the tab up. Start small, measure, climb.
 */
const INITIAL_SCALE = 0.4
const INITIAL_SAMPLES = 32
const SOFTWARE_SCALE = 0.15
const SOFTWARE_SAMPLES = 16
/** Supersampling factor for the downloadable still. */
const EXPORT_SCALE = 2
/** Frames between UI updates of the fps readout — one per frame would thrash React. */
const FPS_UPDATE_EVERY = 15

export function GpuViewer({ nerf, fovDegrees, radius, crop }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rendererRef = useRef<GpuNerfRenderer | null>(null)

  // A software GL stack cannot carry this shader at normal settings; detect it
  // once and open with something it can actually finish.
  const software = useMemo(() => describeWebglSupport().software === true, [])
  const [mode, setMode] = useState<RenderMode>('color')
  const [samples, setSamples] = useState(software ? SOFTWARE_SAMPLES : INITIAL_SAMPLES)
  const [autoRotate, setAutoRotate] = useState(false)
  const [stats, setStats] = useState({ fps: 0, scale: INITIAL_SCALE })
  const [exporting, setExporting] = useState(false)
  const [failure, setFailure] = useState<string | null>(null)

  // The animation loop is created once, so it reads live settings through refs.
  const settings = useRef({ samples, mode, autoRotate })
  settings.current = { samples, mode, autoRotate }
  const scaleRef = useRef(software ? SOFTWARE_SCALE : INITIAL_SCALE)

  const { requestSnapshot, snapshot, status } = nerf

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let renderer: GpuNerfRenderer
    try {
      renderer = new GpuNerfRenderer(canvas, { fovDegrees, radius })
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error))
      return
    }
    rendererRef.current = renderer
    requestSnapshot()

    let frame = 0
    let lastFrameAt = performance.now()
    let smoothedFrameMs = 1000 / 60
    let smoothedFps = 0
    let sinceUiUpdate = 0
    let sizedWidth = 0
    let sizedHeight = 0
    let sizedScale = 0

    const loop = () => {
      frame = requestAnimationFrame(loop)
      if (!renderer.ready) return

      const rect = canvas.getBoundingClientRect()
      if (rect.width === 0 || rect.height === 0) return
      if (rect.width !== sizedWidth || rect.height !== sizedHeight || scaleRef.current !== sizedScale) {
        renderer.setSize(rect.width, rect.height, scaleRef.current)
        sizedWidth = rect.width
        sizedHeight = rect.height
        sizedScale = scaleRef.current
      }

      renderer.setSamples(settings.current.samples)
      renderer.setMode(settings.current.mode)
      renderer.controls.autoRotate = settings.current.autoRotate
      renderer.controls.autoRotateSpeed = 1.6

      renderer.render()

      const now = performance.now()
      const frameMs = now - lastFrameAt
      lastFrameAt = now
      if (frameMs > 0) {
        smoothedFrameMs = smoothedFrameMs * 0.85 + frameMs * 0.15
        smoothedFps = smoothedFps * 0.9 + (1000 / frameMs) * 0.1
      }

      // Trade resolution for smoothness: dropping pixels is far less noticeable
      // than a camera that lurches behind the mouse. Shrink fast, grow slowly —
      // the gap between the two thresholds keeps it from oscillating.
      if (smoothedFrameMs > SLOW_FRAME_MS && scaleRef.current > MIN_SCALE) {
        scaleRef.current = Math.max(MIN_SCALE, scaleRef.current - 0.06)
        smoothedFrameMs = FAST_FRAME_MS
      } else if (smoothedFrameMs < FAST_FRAME_MS && scaleRef.current < MAX_SCALE) {
        scaleRef.current = Math.min(MAX_SCALE, scaleRef.current + 0.02)
        smoothedFrameMs = SLOW_FRAME_MS * 0.6
      }

      if (++sinceUiUpdate >= FPS_UPDATE_EVERY) {
        sinceUiUpdate = 0
        setStats({ fps: smoothedFps, scale: scaleRef.current })
      }
    }
    frame = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(frame)
      renderer.dispose()
      rendererRef.current = null
    }
  }, [fovDegrees, radius, requestSnapshot])

  // Push new weights into the textures whenever the worker sends them.
  useEffect(() => {
    if (snapshot && rendererRef.current) rendererRef.current.setSnapshot(snapshot)
  }, [snapshot])

  useEffect(() => {
    if (crop && rendererRef.current) rendererRef.current.setBounds(crop.min, crop.max)
  }, [crop, snapshot])

  // Keep pulling fresh weights while the model is still learning.
  useEffect(() => {
    if (status !== 'egitiliyor') return
    const timer = window.setInterval(requestSnapshot, SNAPSHOT_INTERVAL_MS)
    return () => window.clearInterval(timer)
  }, [status, requestSnapshot])

  const downloadFrame = useCallback(async () => {
    const renderer = rendererRef.current
    const canvas = canvasRef.current
    if (!renderer || !canvas) return
    setExporting(true)
    try {
      const rect = canvas.getBoundingClientRect()
      // One supersampled frame, read back straight away — the animation loop
      // restores its own size on the next tick. `readPixels` blocks until the
      // GPU is done, so on a software stack the supersampling is dropped:
      // otherwise this single call would freeze the tab for minutes.
      renderer.setSize(rect.width, rect.height, software ? 1 : EXPORT_SCALE)
      renderer.setSamples(software ? samples : Math.max(samples, 64))
      renderer.render()
      const image = renderer.readPixels()
      const target = document.createElement('canvas')
      paintCanvas(target, image.rgba, image.width, image.height)
      await downloadCanvasPng(target, `krea-nerf-${Date.now()}.png`)
    } finally {
      setExporting(false)
    }
  }, [samples, software])

  if (failure) {
    return <p className="warning">GPU görüntüleyici başlatılamadı: {failure}</p>
  }

  return (
    <div className="viewer">
      <div className="viewer-stage">
        <canvas ref={canvasRef} className="viewer-canvas viewer-canvas--gpu" />
        <span className="viewer-badge">
          GPU · {stats.fps.toFixed(0)} fps · ölçek %{(stats.scale * 100).toFixed(0)}
        </span>
        {!snapshot && <span className="viewer-busy">ağırlıklar yükleniyor…</span>}
      </div>

      <div className="viewer-controls">
        <button
          type="button"
          className={`btn ${autoRotate ? 'btn--primary' : ''}`}
          onClick={() => setAutoRotate((v) => !v)}
        >
          {autoRotate ? 'Döndürmeyi durdur' : 'Etrafında döndür'}
        </button>
        <button
          type="button"
          className="btn"
          onClick={() => setMode((m) => (m === 'color' ? 'depth' : 'color'))}
        >
          {mode === 'color' ? 'Derinlik haritası' : 'Renkli görüntü'}
        </button>
        <button
          type="button"
          className="btn"
          disabled={exporting}
          onClick={() => void downloadFrame()}
        >
          {exporting ? 'Kare üretiliyor…' : 'Bu kareyi PNG indir'}
        </button>
        <button type="button" className="btn btn--ghost" onClick={requestSnapshot}>
          Ağırlıkları tazele
        </button>
      </div>

      <div className="field field--slider viewer-quality">
        <label>
          Işın başına örnek
          <output>{samples}</output>
        </label>
        <input
          type="range"
          min={16}
          max={128}
          step={8}
          value={samples}
          onChange={(e) => setSamples(Number(e.target.value))}
        />
      </div>

      {software && (
        <p className="warning">
          Bu tarayıcıda WebGL yazılımla (CPU üzerinde) çalışıyor, gerçek bir ekran kartıyla değil.
          Görüntü doğru üretilir ama çok yavaştır; bu yüzden çözünürlük ve örnek sayısı düşük
          başlatıldı.
        </p>
      )}

      <p className="hint">
        Sol tuşla döndürün, tekerlekle yaklaşın, sağ tuşla kaydırın — tam orbit serbest. Bu
        açılardan hiç fotoğraf yok; her kare, ağın öğrendiği hacimden ekran kartında yeniden
        üretiliyor. Eğitim sürerken ağırlıklar saniyede bir tazelenir ve sahne gözünüzün önünde
        keskinleşir.
      </p>
    </div>
  )
}
