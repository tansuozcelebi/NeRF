/**
 * Novel-view explorer: drag to orbit, wheel to zoom.
 *
 * Rendering a NeRF frame on a CPU costs hundreds of milliseconds, so the viewer
 * renders progressively — a coarse frame lands immediately while you are
 * dragging, and sharper passes replace it once the camera settles.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { orbitPose } from '../nerf/camera'
import type { NerfWorkerApi } from '../hooks/useNerfWorker'
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
/** Delay before starting the next, sharper pass. */
const REFINE_DELAY_MS = 120
const TURNTABLE_SPEED = 0.6 // radians per second

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
  const dragState = useRef<{ x: number; y: number } | null>(null)
  const refineTimer = useRef<number | null>(null)

  useEffect(() => {
    setCamera((c) => ({ ...c, radius }))
  }, [radius])

  const restart = useCallback(() => {
    setLevel(0)
  }, [])

  // Render whenever the camera, the detail level or the mode changes.
  useEffect(() => {
    let cancelled = false
    const { size, samples } = LADDER[level]
    const pose = orbitPose(camera.azimuth, camera.elevation, camera.radius)
    setBusy(true)
    nerf
      .renderView({ pose, fovDegrees, width: size, height: size, mode, samplesPerRay: samples })
      .then((image) => {
        if (cancelled) return
        if (canvasRef.current) {
          paintCanvas(canvasRef.current, image.rgba, image.width, image.height, true)
        }
        setLastRender({ ms: image.elapsedMs, size })
        setBusy(false)
        // Climb the ladder only while the camera is still.
        if (level < LADDER.length - 1 && !spinning) {
          refineTimer.current = window.setTimeout(() => setLevel((l) => l + 1), REFINE_DELAY_MS)
        }
      })
      .catch(() => setBusy(false))

    return () => {
      cancelled = true
      if (refineTimer.current !== null) {
        window.clearTimeout(refineTimer.current)
        refineTimer.current = null
      }
    }
  }, [camera, level, mode, fovDegrees, nerf, spinning])

  // Turntable animation.
  useEffect(() => {
    if (!spinning) return
    let frame = 0
    let previous = performance.now()
    const tick = (now: number) => {
      const dt = (now - previous) / 1000
      previous = now
      setCamera((c) => ({ ...c, azimuth: c.azimuth + TURNTABLE_SPEED * dt }))
      setLevel(0)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [spinning])

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId)
    dragState.current = { x: e.clientX, y: e.clientY }
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
    const size = 256
    const pose = orbitPose(camera.azimuth, camera.elevation, camera.radius)
    const image = await nerf.renderView({
      pose, fovDegrees, width: size, height: size, mode, samplesPerRay: 64,
    })
    const canvas = document.createElement('canvas')
    paintCanvas(canvas, image.rgba, image.width, image.height)
    canvas.toBlob((blob) => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `nerf-goruntu-${Date.now()}.png`
      link.click()
      URL.revokeObjectURL(url)
    }, 'image/png')
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
        <button type="button" className="btn" onClick={() => setSpinning((s) => !s)}>
          {spinning ? 'Döndürmeyi durdur' : 'Etrafında döndür'}
        </button>
        <button
          type="button"
          className="btn"
          onClick={() => {
            setMode((m) => (m === 'color' ? 'depth' : 'color'))
            restart()
          }}
        >
          {mode === 'color' ? 'Derinlik haritası' : 'Renkli görüntü'}
        </button>
        <button type="button" className="btn" onClick={() => void downloadFrame()}>
          Bu kareyi PNG indir
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
