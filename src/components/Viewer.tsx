/**
 * Picks the novel-view explorer to use.
 *
 * The GPU path is the one worth showing — it renders interactively. The CPU
 * path stays as a fallback so the app still works on machines without WebGL2
 * rather than showing an error where the result should be.
 */
import { useMemo, useState } from 'react'
import { CpuViewer } from './CpuViewer'
import { GpuViewer } from './GpuViewer'
import { describeWebglSupport } from '../gpu/GpuNerfRenderer'
import type { NerfWorkerApi } from '../hooks/useNerfWorker'

interface Props {
  nerf: NerfWorkerApi
  fovDegrees: number
  radius: number
}

export function Viewer({ nerf, fovDegrees, radius }: Props) {
  const support = useMemo(() => describeWebglSupport(), [])
  const [useGpu, setUseGpu] = useState(true)

  if (!support.supported) {
    return (
      <>
        <p className="warning">
          {support.reason} Görüntüler işlemci üzerinde üretiliyor; bu belirgin şekilde yavaştır.
        </p>
        <CpuViewer nerf={nerf} fovDegrees={fovDegrees} radius={radius} />
      </>
    )
  }

  return (
    <>
      {useGpu ? (
        <GpuViewer nerf={nerf} fovDegrees={fovDegrees} radius={radius} />
      ) : (
        <CpuViewer nerf={nerf} fovDegrees={fovDegrees} radius={radius} />
      )}
      <p className="hint">
        <button type="button" className="link-button" onClick={() => setUseGpu((v) => !v)}>
          {useGpu ? 'İşlemci (CPU) görüntüleyiciye geç' : 'GPU görüntüleyiciye dön'}
        </button>
        {useGpu
          ? ' — karşılaştırma için; CPU yolu aynı modeli çok daha yavaş üretir.'
          : ' — GPU yolu gerçek zamanlı çalışır.'}
      </p>
    </>
  )
}
