/** Handing a generated file to the user. */

/**
 * Saves a blob under `filename`.
 *
 * Two details matter here and both have bitten this app:
 *
 *  - the anchor is attached to the document before it is clicked, which some
 *    browsers require for a programmatic download to start at all;
 *  - the object URL is revoked on a later turn of the event loop. Revoking it
 *    immediately after `click()` races the browser's own fetch of the URL and
 *    silently cancels the download.
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.style.display = 'none'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  window.setTimeout(() => URL.revokeObjectURL(url), 60_000)
}

/** Encodes a canvas as PNG and downloads it. */
export function downloadCanvasPng(canvas: HTMLCanvasElement, filename: string): Promise<void> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error('Görüntü PNG olarak kodlanamadı.'))
        return
      }
      downloadBlob(blob, filename)
      resolve()
    }, 'image/png')
  })
}
