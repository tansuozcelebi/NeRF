/**
 * Trimming the reconstructed volume down to the subject.
 *
 * NeRFs almost always leave debris — semi-transparent blobs floating in empty space
 * where too few photos disagreed to rule anything out. Cropping is the standard
 * cleanup: it costs nothing, needs no retraining, and it is usually the
 * difference between an export that looks like the object and one that looks
 * like the object inside a cloud.
 */
export interface CropBox {
  min: [number, number, number]
  max: [number, number, number]
}

interface Props {
  aabbSize: number
  crop: CropBox
  onChange: (crop: CropBox) => void
}

const AXES: Array<{ index: 0 | 1 | 2; label: string }> = [
  { index: 0, label: 'X — sağ/sol' },
  { index: 1, label: 'Y — yukarı/aşağı' },
  { index: 2, label: 'Z — ön/arka' },
]

export function fullCrop(aabbSize: number): CropBox {
  return { min: [-aabbSize, -aabbSize, -aabbSize], max: [aabbSize, aabbSize, aabbSize] }
}

export function CropControls({ aabbSize, crop, onChange }: Props) {
  const setAxis = (axis: 0 | 1 | 2, which: 'min' | 'max', value: number) => {
    const next: CropBox = { min: [...crop.min], max: [...crop.max] }
    if (which === 'min') next.min[axis] = Math.min(value, crop.max[axis] - 0.05)
    else next.max[axis] = Math.max(value, crop.min[axis] + 0.05)
    onChange(next)
  }

  const trimmed =
    crop.min.some((v) => v > -aabbSize + 1e-6) || crop.max.some((v) => v < aabbSize - 1e-6)

  return (
    <div className="crop">
      <div className="crop-header">
        <strong>Kırpma kutusu</strong>
        <button
          type="button"
          className="btn btn--ghost"
          disabled={!trimmed}
          onClick={() => onChange(fullCrop(aabbSize))}
        >
          Sıfırla
        </button>
      </div>
      {AXES.map(({ index, label }) => (
        <div key={index} className="crop-axis">
          <label>
            {label}
            <output>
              {crop.min[index].toFixed(2)} … {crop.max[index].toFixed(2)}
            </output>
          </label>
          <div className="crop-sliders">
            <input
              type="range"
              min={-aabbSize}
              max={aabbSize}
              step={0.05}
              value={crop.min[index]}
              aria-label={`${label} alt sınır`}
              onChange={(e) => setAxis(index, 'min', Number(e.target.value))}
            />
            <input
              type="range"
              min={-aabbSize}
              max={aabbSize}
              step={0.05}
              value={crop.max[index]}
              aria-label={`${label} üst sınır`}
              onChange={(e) => setAxis(index, 'max', Number(e.target.value))}
            />
          </div>
        </div>
      ))}
      <p className="hint">
        Görüntüleyiciyi ve dışa aktarılan modeli birlikte etkiler. Zeminden kurtulmak için genellikle
        Y alt sınırını yükseltmek yeterlidir.
      </p>
    </div>
  )
}
