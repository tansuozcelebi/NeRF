import { useEffect, useRef } from 'react'
import { paintCanvas } from '../utils/image'
import type { RenderedImage } from '../hooks/useNerfWorker'

interface Props {
  image: RenderedImage | null
  /** CSS class for the canvas element. */
  className?: string
  /** Placeholder shown until the first image arrives. */
  placeholder?: string
  /** Smooth the upscale. Off looks sharper for tiny previews. */
  smooth?: boolean
}

/** Paints a worker-rendered RGBA buffer into a canvas, scaled up by CSS. */
export function CanvasImage({ image, className, placeholder, smooth = true }: Props) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!ref.current || !image) return
    paintCanvas(ref.current, image.rgba, image.width, image.height, smooth)
  }, [image, smooth])

  if (!image) {
    return <div className={`canvas-placeholder ${className ?? ''}`}>{placeholder ?? '—'}</div>
  }
  return (
    <canvas
      ref={ref}
      className={className}
      style={{ imageRendering: smooth ? 'auto' : 'pixelated' }}
    />
  )
}
