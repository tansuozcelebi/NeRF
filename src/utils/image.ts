/** Turning uploaded photos into training-sized float buffers. */

export interface DecodedPhoto {
  id: string
  name: string
  width: number
  height: number
  /** RGB in [0,1], 3 floats per pixel. */
  pixels: Float32Array
  /** Object URL of the original file, for showing the thumbnail. */
  previewUrl: string
}

/**
 * Decodes a file and downscales it so its long side is `longSide` pixels.
 *
 * Training resolution is deliberately small: every step samples random pixels,
 * and a 4000x3000 photo would spend all its memory bandwidth for detail the
 * model cannot represent anyway. 96-160px is the sweet spot in a browser.
 */
export async function decodePhoto(file: File, longSide: number): Promise<DecodedPhoto> {
  const bitmap = await createImageBitmap(file)
  try {
    const scale = longSide / Math.max(bitmap.width, bitmap.height)
    const width = Math.max(1, Math.round(bitmap.width * Math.min(1, scale)))
    const height = Math.max(1, Math.round(bitmap.height * Math.min(1, scale)))

    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    if (!ctx) throw new Error('2D çizim bağlamı oluşturulamadı.')
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(bitmap, 0, 0, width, height)

    const data = ctx.getImageData(0, 0, width, height).data
    const pixels = new Float32Array(width * height * 3)
    for (let i = 0; i < width * height; i++) {
      pixels[i * 3] = data[i * 4] / 255
      pixels[i * 3 + 1] = data[i * 4 + 1] / 255
      pixels[i * 3 + 2] = data[i * 4 + 2] / 255
    }

    return {
      id: `${file.name}-${file.size}-${file.lastModified}`,
      name: file.name,
      width,
      height,
      pixels,
      previewUrl: URL.createObjectURL(file),
    }
  } finally {
    bitmap.close()
  }
}

/** Average brightness of a photo set, used to guess a background colour. */
export function averageColor(photos: DecodedPhoto[]): [number, number, number] {
  let r = 0, g = 0, b = 0, n = 0
  for (const photo of photos) {
    // Sample the border, which is usually background rather than subject.
    const { width, height, pixels } = photo
    for (let x = 0; x < width; x++) {
      for (const y of [0, height - 1]) {
        const o = (y * width + x) * 3
        r += pixels[o]; g += pixels[o + 1]; b += pixels[o + 2]; n++
      }
    }
    for (let y = 0; y < height; y++) {
      for (const x of [0, width - 1]) {
        const o = (y * width + x) * 3
        r += pixels[o]; g += pixels[o + 1]; b += pixels[o + 2]; n++
      }
    }
  }
  if (n === 0) return [0, 0, 0]
  return [r / n, g / n, b / n]
}

/** Draws RGBA bytes into a canvas, scaling to fit its CSS size. */
export function paintCanvas(
  canvas: HTMLCanvasElement,
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  smooth = false,
): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }
  ctx.imageSmoothingEnabled = smooth
  // Going through createImageData rather than `new ImageData(rgba, …)` keeps
  // this independent of how the buffer was allocated (transferred worker
  // buffers are not typed as plain ArrayBuffer).
  const image = ctx.createImageData(width, height)
  image.data.set(rgba)
  ctx.putImageData(image, 0, 0)
}

export function rgbToCss([r, g, b]: [number, number, number]): string {
  const to = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255)
  return `#${[to(r), to(g), to(b)].map((v) => v.toString(16).padStart(2, '0')).join('')}`
}

export function cssToRgb(hex: string): [number, number, number] {
  const value = hex.replace('#', '')
  const n = parseInt(value.length === 3 ? value.replace(/./g, (c) => c + c) : value, 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}
