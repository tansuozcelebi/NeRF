import { useRef, useState } from 'react'
import { poseOrigin } from '../nerf/camera'
import type { Mat4 } from '../nerf/types'
import { parseTransforms, type ImportedPoses } from '../utils/transforms'
import type { PoseConfig, PoseMode } from '../state/poseConfig'

interface Props {
  config: PoseConfig
  onChange: (config: PoseConfig) => void
  poses: Mat4[]
  note: string
  problem: boolean
  imported: ImportedPoses | null
  onImport: (imported: ImportedPoses | null) => void
}

const MODE_LABELS: Record<PoseMode, string> = {
  halka: 'Halka (sabit yükseklik)',
  kubbe: 'Kubbe (değişen yükseklik)',
  dosya: 'transforms.json içe aktar',
}

export function CameraSetup({
  config, onChange, poses, note, problem, imported, onImport,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const [importError, setImportError] = useState<string | null>(null)

  const set = <K extends keyof PoseConfig>(key: K, value: PoseConfig[K]) =>
    onChange({ ...config, [key]: value })

  const loadTransforms = async (file: File | undefined) => {
    if (!file) return
    setImportError(null)
    try {
      const parsed = parseTransforms(await file.text())
      onImport(parsed)
      if (parsed.fovDegrees) set('fovDegrees', Math.round(parsed.fovDegrees))
    } catch (error) {
      onImport(null)
      setImportError(error instanceof Error ? error.message : String(error))
    }
  }

  return (
    <div className="camera-setup">
      <div className="field">
        <label htmlFor="pose-mode">Kamera konumları</label>
        <select
          id="pose-mode"
          value={config.mode}
          onChange={(e) => set('mode', e.target.value as PoseMode)}
        >
          {Object.entries(MODE_LABELS).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </div>

      {config.mode === 'dosya' ? (
        <div className="field">
          <label>Poz dosyası</label>
          <input
            ref={fileRef}
            type="file"
            accept=".json,application/json"
            onChange={(e) => void loadTransforms(e.target.files?.[0])}
          />
          {imported && (
            <p className="hint">
              {imported.poses.length} kare okundu
              {imported.fovDegrees ? `, görüş açısı ${imported.fovDegrees.toFixed(1)}°` : ''}.
            </p>
          )}
          {importError && <p className="warning">{importError}</p>}
        </div>
      ) : (
        <>
          <Slider
            label="Özneye uzaklık"
            value={config.radius}
            min={1.5}
            max={6}
            step={0.1}
            suffix=" birim"
            onChange={(v) => set('radius', v)}
          />
          <Slider
            label={config.mode === 'kubbe' ? 'En düşük yükseklik açısı' : 'Yükseklik açısı'}
            value={config.elevationDeg}
            min={-20}
            max={70}
            step={1}
            suffix="°"
            onChange={(v) => set('elevationDeg', v)}
          />
          {config.mode === 'kubbe' && (
            <Slider
              label="En yüksek yükseklik açısı"
              value={config.elevationMaxDeg}
              min={0}
              max={85}
              step={1}
              suffix="°"
              onChange={(v) => set('elevationMaxDeg', Math.max(v, config.elevationDeg))}
            />
          )}
        </>
      )}

      <Slider
        label="Kamera görüş açısı"
        value={config.fovDegrees}
        min={20}
        max={100}
        step={1}
        suffix="°"
        onChange={(v) => set('fovDegrees', v)}
      />
      <Slider
        label="Sahne kutusu yarıçapı"
        value={config.aabbSize}
        min={0.6}
        max={4}
        step={0.1}
        suffix=" birim"
        onChange={(v) => set('aabbSize', v)}
      />

      <PoseDiagram poses={poses} aabbSize={config.aabbSize} />
      <p className={problem ? 'warning' : 'hint'}>{note}</p>
    </div>
  )
}

interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  suffix?: string
  onChange: (value: number) => void
}

function Slider({ label, value, min, max, step, suffix, onChange }: SliderProps) {
  return (
    <div className="field field--slider">
      <label>
        {label}
        <output>
          {value}
          {suffix}
        </output>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}

/** Top-down map of where the app thinks the cameras were. */
function PoseDiagram({ poses, aabbSize }: { poses: Mat4[]; aabbSize: number }) {
  const size = 200
  const centre = size / 2
  const points = poses.map((p) => poseOrigin(p))
  const extent = Math.max(aabbSize, ...points.map(([x, , z]) => Math.hypot(x, z)), 1) * 1.15
  const scale = (size / 2 - 12) / extent
  const boxHalf = aabbSize * scale

  return (
    <svg className="pose-diagram" viewBox={`0 0 ${size} ${size}`} role="img"
      aria-label="Kamera konumlarının üstten görünümü">
      <rect
        x={centre - boxHalf} y={centre - boxHalf} width={boxHalf * 2} height={boxHalf * 2}
        className="diagram-box"
      />
      <circle cx={centre} cy={centre} r={3} className="diagram-origin" />
      {points.map(([x, y, z], i) => (
        <circle
          key={i}
          cx={centre + x * scale}
          // Screen Y grows downwards; world +Z points towards the viewer.
          cy={centre + z * scale}
          r={3}
          className="diagram-camera"
          // Higher cameras render brighter, so elevation is visible from above.
          opacity={0.45 + 0.55 * Math.min(1, Math.max(0, (y + extent) / (2 * extent)))}
        />
      ))}
      <text x={6} y={size - 6} className="diagram-label">
        üstten görünüm · {poses.length} kamera
      </text>
    </svg>
  )
}
