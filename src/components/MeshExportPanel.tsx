/**
 * Pulling a real 3D model out of the trained field.
 *
 * This is the step that makes a NeRF usable outside this app: the density
 * function is sampled on a grid, the surface where it crosses a threshold is
 * turned into triangles, and each vertex is coloured by asking the colour
 * network what that point looks like head-on. The result opens in Blender,
 * MeshLab or anything else that reads PLY.
 */
import { useState } from 'react'
import type { CropBox } from './CropControls'
import type { NerfWorkerApi } from '../hooks/useNerfWorker'
import { downloadBlob } from '../utils/download'
import { encodePly } from '../utils/ply'

interface Props {
  nerf: NerfWorkerApi
  crop: CropBox
}

const RESOLUTIONS = [64, 96, 128, 160]

interface Result {
  vertexCount: number
  triangleCount: number
  isoLevel: number
}

export function MeshExportPanel({ nerf, crop }: Props) {
  const [resolution, setResolution] = useState(96)
  const [isoScale, setIsoScale] = useState(1)
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<Result | null>(null)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setBusy(true)
    setProgress(0)
    setError(null)
    setResult(null)
    try {
      const mesh = await nerf.extractMesh(
        { resolution, isoScale, boundsMin: crop.min, boundsMax: crop.max },
        setProgress,
      )
      setResult({
        vertexCount: mesh.vertexCount,
        triangleCount: mesh.triangleCount,
        isoLevel: mesh.isoLevel,
      })
      if (mesh.triangleCount === 0) {
        setError(
          'Bu eşikte hiç yüzey bulunamadı. Yüzey eşiğini düşürün veya modeli biraz daha eğitin.',
        )
        return
      }
      downloadBlob(
        encodePly(
          mesh,
          `KREA NeRF Studyo — ${resolution}^3 izgara, esik ${mesh.isoLevel.toFixed(3)}`,
        ),
        `krea-nerf-model-${mesh.triangleCount}ucgen.ply`,
      )
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="mesh-export">
      <h3>3B model olarak dışa aktar</h3>
      <p className="hint">
        Yoğunluk alanından üçgen yüzey çıkarılır, her köşe renk ağına sorularak boyanır ve renkli
        PLY olarak indirilir. Kırpma kutusu buraya da uygulanır.
      </p>

      <div className="mesh-export-controls">
        <div className="field">
          <label htmlFor="mesh-res">Izgara çözünürlüğü</label>
          <select
            id="mesh-res"
            value={resolution}
            disabled={busy}
            onChange={(e) => setResolution(Number(e.target.value))}
          >
            {RESOLUTIONS.map((r) => (
              <option key={r} value={r}>
                {r}³ ({((r ** 3) / 1e6).toFixed(1)} milyon örnek)
              </option>
            ))}
          </select>
        </div>

        <div className="field field--slider">
          <label>
            Yüzey eşiği
            <output>{isoScale.toFixed(2)}×</output>
          </label>
          <input
            type="range"
            min={0.25}
            max={4}
            step={0.05}
            value={isoScale}
            disabled={busy}
            onChange={(e) => setIsoScale(Number(e.target.value))}
          />
          <p className="hint">
            Otomatik seçilen eşiği ölçekler. Düşürmek sisi de dahil eder, yükseltmek yalnızca katı
            gövdeyi bırakır.
          </p>
        </div>
      </div>

      <div className="button-row">
        <button type="button" className="btn btn--primary" disabled={busy} onClick={() => void run()}>
          {busy ? `Yüzey çıkarılıyor… %${(progress * 100).toFixed(0)}` : 'Modeli çıkar ve indir (.ply)'}
        </button>
        {result && (
          <span className="hint">
            {result.triangleCount.toLocaleString('tr')} üçgen ·{' '}
            {result.vertexCount.toLocaleString('tr')} köşe · eşik {result.isoLevel.toFixed(3)}
          </span>
        )}
      </div>
      {error && <p className="warning">{error}</p>}
    </div>
  )
}
