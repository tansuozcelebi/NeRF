import { useCallback, useRef, useState } from 'react'
import { decodePhoto, type DecodedPhoto } from '../utils/image'

interface Props {
  photos: DecodedPhoto[]
  onChange: (photos: DecodedPhoto[]) => void
  /** Long side of the downscaled training image, in pixels. */
  trainingResolution: number
}

export function PhotoUploader({ photos, onChange, trainingResolution }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)

  const addFiles = useCallback(
    async (files: FileList | null) => {
      if (!files || files.length === 0) return
      setBusy(true)
      setProblem(null)
      const accepted: DecodedPhoto[] = []
      const rejected: string[] = []
      for (const file of Array.from(files)) {
        if (!file.type.startsWith('image/')) {
          rejected.push(file.name)
          continue
        }
        try {
          accepted.push(await decodePhoto(file, trainingResolution))
        } catch {
          rejected.push(file.name)
        }
      }
      // Sort by file name so "IMG_001, IMG_002, ..." keeps capture order, which
      // is what the orbit assumption relies on.
      const merged = [...photos, ...accepted]
      merged.sort((a, b) => a.name.localeCompare(b.name, 'tr', { numeric: true }))
      onChange(merged)
      if (rejected.length > 0) {
        setProblem(`Şu dosyalar okunamadı: ${rejected.join(', ')}`)
      }
      setBusy(false)
    },
    [onChange, photos, trainingResolution],
  )

  const removeAt = (index: number) => {
    const photo = photos[index]
    URL.revokeObjectURL(photo.previewUrl)
    onChange(photos.filter((_, i) => i !== index))
  }

  const clearAll = () => {
    photos.forEach((p) => URL.revokeObjectURL(p.previewUrl))
    onChange([])
  }

  return (
    <div className="uploader">
      <div
        className={`dropzone ${dragging ? 'dropzone--active' : ''}`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          void addFiles(e.dataTransfer.files)
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click()
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(e) => {
            void addFiles(e.target.files)
            e.target.value = ''
          }}
        />
        <strong>{busy ? 'Fotoğraflar işleniyor…' : 'Fotoğrafları buraya sürükleyin'}</strong>
        <span>
          veya tıklayıp seçin — eğitim için uzun kenarı {trainingResolution} piksele küçültülür
        </span>
      </div>

      {problem && <p className="warning">{problem}</p>}

      {photos.length > 0 && (
        <>
          <div className="uploader-toolbar">
            <span>
              <strong>{photos.length}</strong> fotoğraf · dosya adına göre sıralı
            </span>
            <button type="button" className="btn btn--ghost" onClick={clearAll}>
              Tümünü kaldır
            </button>
          </div>
          <ul className="thumb-grid">
            {photos.map((photo, index) => (
              <li key={photo.id} className="thumb">
                <img src={photo.previewUrl} alt={photo.name} />
                <span className="thumb-index">{index + 1}</span>
                <button
                  type="button"
                  className="thumb-remove"
                  onClick={() => removeAt(index)}
                  aria-label={`${photo.name} fotoğrafını kaldır`}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  )
}
