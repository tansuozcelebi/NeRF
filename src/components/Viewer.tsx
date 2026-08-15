/**
 * Novel-view explorer: drag to orbit, wheel to zoom.
 *
 * Rendering a NeRF frame on a CPU costs hundreds of milliseconds, so the viewer
 * renders progressively — a coarse frame lands immediately while you are
 * dragging, and sharper passes replace it once the camera settles.
 *
 * Requests are strictly serialised: at most one render is in flight, and while
 * it runs only the *latest* camera is kept. Firing a request per pointer move or
 * per animation frame would queue work in the worker far faster than it can be
 * consumed, and the viewer would fall minutes behind the mouse.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { orbitPose } from '../nerf/camera'
import type { NerfWorkerApi } from '../hooks/useNerfWorker'
import { downloadCanvasPng } from '../utils/download'
import { paintCanvas } from '../utils/image'

interface Props {
  nerf: NerfWorkerApi
  fovDegrees: number
  /** Starting distance, normally the same radius the cameras were placed at. */
  radius: number
}

/**
 * Refinement ladder, coarse first. Early passes also take fewer samples per
 * ray: while you are dragging, latency matters far more than sharpness.
 */
const LADDER: Array<{ size: number; samples: number }> = [
  { size: 64, samples: 24 },
  { size: 112, samples: 32 },
  { size: 160, samples: 48 },
]
/** Resolution and sampling used for the downloadable still. */
const EXPORT_SIZE = 256
const EXPORT_SAMPLES = 48
/** Delay before starting the next, sharper pass. */
const REFINE_DELAY_MS = 120
/** Turntable rotation, radians per second of wall clock. */
const TURNTABLE_SPEED = 0.6
/** Cap on the angle the turntable advances after a single slow frame. */
const MAX_TURNTABLE_STEP = 0.35

interface Camera {
  azimuth: number
  elevation: number
  radius: number
}

export function Viewer({ nerf, fovDegrees, radius }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [camera, setCamera] = useState<Camera>({ azimuth: 0.9, elevation: 0.35, radius })
  const [level, setLevel] = useState(0)
  const [mode, setMode] = useState<'color' | 'depth'>('color')
  const [spinning, setSpinning] = useState(false)
  const [lastRender, setLastRender] = useState<{ ms: number; size: number } | null>(null)
  const [busy, setBusy] = useState(false)
  const [exporting, setExporting] = useState(false)

  // Mirror of the state the pump reads, so it never works from a stale closure.
  const target = useRef({ camera, level, mode, spinning })
  target.current = { camera, level, mode, spinning }

  const inFlight = useRef(false)
  const renderedKey = useRef('')
  const refineTimer = useRef<number | null>(null)
  const disposed = useRef(false)
  const dragState = useRef<{ x: number; y: number } | null>(null)

  const cancelRefine = () => {
    if (refineTimer.current !== null) {
      window.clearTimeout(refineTimer.current)
      refineTimer.current = null
    }
  }

  /** Renders the current target, unless a render is already running. */
  const pump = useCallback(() => {
    if (disposed.current || inFlight.current) return
    const { camera: cam, level: lvl, mode: md, spinning: spin } = target.current
    const key = [
      cam.azimuth.toFixed(4), cam.elevation.toFixed(4), cam.radius.toFixed(4), lvl, md,
    ].join('|')
    if (key === renderedKey.current) return

    const { size, samples } = LADDER[lvl]
    inFlight.current = true
    renderedKey.current = key
    setBusy(true)
    const startedAt = performance.now()

    nerf
      .renderView({
        pose: orbitPose(cam.azimuth, cam.elevation, cam.radius),
        fovDegrees,
        width: size,
        height: size,
        mode: md,
        samplesPerRay: samples,
      })
      .then((image) => {
        inFlight.current = false
        if (disposed.current) return
        if (canvasRef.current) {
          paintCanvas(canvasRef.current, image.rgba, image.width, image.height, true)
        }
        setLastRender({ ms: image.elapsedMs, size })
        setBusy(false)

        if (spin) {
          // Advance by real elapsed time so the rotation speed stays honest when
          // frames are slow, but never so far that it reads as a jump cut.
          const elapsed = (performance.now() - startedAt) / 1000
          setCamera((c) => ({
            ...c,
            azimuth: c.azimuth + Math.min(TURNTABLE_SPEED * elapsed, MAX_TURNTABLE_STEP),
          }))
        } else if (lvl < LADDER.length - 1) {
          cancelRefine()
          refineTimer.current = window.setTimeout(() => setLevel((l) => l + 1), REFINE_DELAY_MS)
        } else {
          // The camera may have moved while this frame was rendering.
          pump()
        }
      })
      .catch(() => {
        inFlight.current = false
        setBusy(false)
      })
  }, [fovDegrees, nerf])

  // Any change to the target starts a render, if the pump is free.
  useEffect(() => {
    pump()
  }, [camera, level, mode, spinning, pump])

  useEffect(() => {
    disposed.current = false
    return () => {
      disposed.current = true
      cancelRefine()
    }
  }, [])

  useEffect(() => {
    setCamera((c) => ({ ...c, radius }))
  }, [radius])

  /** Drops back to the coarse pass — used whenever the camera moves. */
  const restart = useCallback(() => {
    cancelRefine()
    setLevel(0)
  }, [])

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId)
    dragState.current = { x: e.clientX, y: e.clientY }
    setSpinning(false)
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragState.current) return
    const dx = e.clientX - dragState.current.x
    const dy = e.clientY - dragState.current.y
    dragState.current = { x: e.clientX, y: e.clientY }
    setCamera((c) => ({
      ...c,
      azimuth: c.azimuth + dx * 0.01,
      // Stop just short of the poles, where the up vector degenerates.
      elevation: Math.min(1.45, Math.max(-1.45, c.elevation + dy * 0.01)),
    }))
    restart()
  }

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    dragState.current = null
    e.currentTarget.releasePointerCapture(e.pointerId)
  }

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    setCamera((c) => ({
      ...c,
      radius: Math.min(9, Math.max(1.2, c.radius * (1 + Math.sign(e.deltaY) * 0.08))),
    }))
    restart()
  }

  const downloadFrame = async () => {
    // A full-quality still is a much bigger render than the interactive passes
    // and can take a while, so the button reports that it is working.
    setSpinning(false)
    setExporting(true)
    try {
      const image = await nerf.renderView({
        pose: orbitPose(camera.azimuth, camera.elevation, camera.radius),
        fovDegrees,
        width: EXPORT_SIZE,
        height: EXPORT_SIZE,
        mode,
        samplesPerRay: EXPORT_SAMPLES,
      })
      const canvas = document.createElement('canvas')
      paintCanvas(canvas, image.rgba, image.width, image.height)
      await downloadCanvasPng(canvas, `nerf-goruntu-${Date.now()}.png`)
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="viewer">
      <div className="viewer-stage">
        <canvas
          ref={canvasRef}
          className="viewer-canvas"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onWheel={onWheel}
        />
        {busy && <span className="viewer-busy">işleniyor…</span>}
      </div>

      <div className="viewer-controls">
        <button
          type="button"
          className="btn"
          disabled={exporting}
          onClick={() => {
            setSpinning((s) => !s)
            restart()
          }}
        >
          {spinning ? 'Döndürmeyi durdur' : 'Etrafında döndür'}
        </button>
        <button
          type="button"
          className="btn"
          disabled={exporting}
          onClick={() => {
            setMode((m) => (m === 'color' ? 'depth' : 'color'))
            restart()
          }}
        >
          {mode === 'color' ? 'Derinlik haritası' : 'Renkli görüntü'}
        </button>
        <button
          type="button"
          className="btn"
          disabled={exporting}
          onClick={() => void downloadFrame()}
        >
          {exporting ? `${EXPORT_SIZE}² kare üretiliyor…` : 'Bu kareyi PNG indir'}
        </button>
        <button
          type="button"
          className="btn btn--ghost"
          onClick={() => {
            nerf.setPreviewPose(
              orbitPose(camera.azimuth, camera.elevation, camera.radius),
              fovDegrees,
            )
          }}
        >
          Canlı önizlemeyi buraya al
        </button>
      </div>

      <p className="hint">
        Sürükleyerek döndürün, tekerlekle yaklaşın. Bu açıdan hiç fotoğraf yok — görüntü tamamen
        ağın öğrendiği hacimden üretiliyor.
        {lastRender && ` Son kare: ${lastRender.size}² piksel, ${lastRender.ms.toFixed(0)} ms.`}
      </p>
    </div>
  )
}
