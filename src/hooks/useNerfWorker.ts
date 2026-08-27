/**
 * React binding for the training worker: owns its lifecycle, mirrors its state
 * and turns render requests into promises.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { ExportedMesh, GpuSnapshot, MeshExportOptions } from '../nerf/trainer'
import type { Mat4, QualityPreset, TrainStats } from '../nerf/types'
import type { SerializedView, SyntheticOptions, WorkerRequest, WorkerResponse } from '../worker/protocol'

export type WorkerStatus = 'bos' | 'hazirlaniyor' | 'hazir' | 'egitiliyor' | 'hata'

export interface RenderedImage {
  width: number
  height: number
  rgba: Uint8ClampedArray
}

export interface Thumbnail extends RenderedImage {
  id: string
}

/** Number of points kept in the loss history shown on the chart. */
const HISTORY_LIMIT = 400

export interface NerfWorkerApi {
  status: WorkerStatus
  error: string | null
  stats: TrainStats | null
  history: Array<{ step: number; loss: number; psnr: number; validationPsnr: number | null }>
  thumbnails: Thumbnail[]
  preview: RenderedImage | null
  previewStep: number
  paramCount: number
  viewCount: number
  pointsPerStep: number
  initSynthetic: (preset: QualityPreset, options: SyntheticOptions) => void
  initPhotos: (
    preset: QualityPreset,
    views: SerializedView[],
    aabbSize: number,
    background: [number, number, number],
  ) => void
  start: () => void
  pause: () => void
  reset: () => void
  setPreviewPose: (pose: Mat4, fovDegrees: number) => void
  renderView: (options: {
    pose: Mat4
    fovDegrees: number
    width: number
    height: number
    mode?: 'color' | 'depth'
    samplesPerRay?: number
  }) => Promise<RenderedImage & { elapsedMs: number }>
  exportWeights: () => Promise<{ data: Float32Array; step: number; paramCount: number }>
  /** Latest weights uploaded to the GPU renderer; null until one is requested. */
  snapshot: GpuSnapshot | null
  requestSnapshot: () => void
  /** Pulls a triangle mesh out of the density field. Reports sampling progress. */
  extractMesh: (
    options: Omit<MeshExportOptions, 'onProgress'>,
    onProgress?: (fraction: number) => void,
  ) => Promise<ExportedMesh>
}

export function useNerfWorker(): NerfWorkerApi {
  const workerRef = useRef<Worker | null>(null)
  const requestIdRef = useRef(0)
  const pendingRenders = useRef(
    new Map<
      number,
      {
        resolve: (value: RenderedImage & { elapsedMs: number }) => void
        reject: (reason: Error) => void
      }
    >(),
  )
  const pendingMesh = useRef<{
    resolve: (mesh: ExportedMesh) => void
    reject: (reason: Error) => void
    onProgress?: (fraction: number) => void
  } | null>(null)
  const pendingWeights = useRef<
    ((value: { data: Float32Array; step: number; paramCount: number }) => void) | null
  >(null)

  const [status, setStatus] = useState<WorkerStatus>('bos')
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<TrainStats | null>(null)
  const [history, setHistory] = useState<Array<{ step: number; loss: number; psnr: number; validationPsnr: number | null }>>([])
  const [thumbnails, setThumbnails] = useState<Thumbnail[]>([])
  const [preview, setPreview] = useState<RenderedImage | null>(null)
  const [previewStep, setPreviewStep] = useState(0)
  const [paramCount, setParamCount] = useState(0)
  const [viewCount, setViewCount] = useState(0)
  const [pointsPerStep, setPointsPerStep] = useState(0)
  const [snapshot, setSnapshot] = useState<GpuSnapshot | null>(null)

  /** Fails every outstanding render so no caller waits on a dead request. */
  const rejectPendingRenders = useCallback((reason: string) => {
    const pending = Array.from(pendingRenders.current.values())
    pendingRenders.current.clear()
    for (const { reject } of pending) reject(new Error(reason))
  }, [])

  useEffect(() => {
    const worker = new Worker(new URL('../worker/nerfWorker.ts', import.meta.url), {
      type: 'module',
    })
    workerRef.current = worker

    worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const message = event.data
      switch (message.type) {
        case 'ready':
          // A rebuild throws away the model the in-flight renders belonged to.
          rejectPendingRenders('Model yeniden kuruldu.')
          setSnapshot(null)
          setThumbnails(message.thumbnails)
          setParamCount(message.paramCount)
          setViewCount(message.viewCount)
          setStatus('hazir')
          setStats(null)
          setHistory([])
          setPreview(null)
          setPreviewStep(0)
          setError(null)
          break
        case 'stats':
          setStats(message.stats)
          setPointsPerStep(message.pointsPerStep)
          setStatus(message.running ? 'egitiliyor' : 'hazir')
          setHistory((previous) => {
            const next = [
              ...previous,
              {
                step: message.stats.step,
                loss: message.stats.loss,
                psnr: message.stats.psnr,
                validationPsnr: message.stats.validationPsnr,
              },
            ]
            return next.length > HISTORY_LIMIT ? next.slice(next.length - HISTORY_LIMIT) : next
          })
          break
        case 'preview':
          setPreview({ width: message.width, height: message.height, rgba: message.rgba })
          setPreviewStep(message.step)
          break
        case 'render': {
          const pending = pendingRenders.current.get(message.requestId)
          if (pending) {
            pendingRenders.current.delete(message.requestId)
            pending.resolve({
              width: message.width,
              height: message.height,
              rgba: message.rgba,
              elapsedMs: message.elapsedMs,
            })
          }
          break
        }
        case 'snapshot':
          setSnapshot(message.snapshot)
          break
        case 'meshProgress':
          pendingMesh.current?.onProgress?.(message.fraction)
          break
        case 'mesh':
          pendingMesh.current?.resolve(message.mesh)
          pendingMesh.current = null
          break
        case 'weights':
          pendingWeights.current?.({
            data: message.data,
            step: message.step,
            paramCount: message.paramCount,
          })
          pendingWeights.current = null
          break
        case 'error':
          // Nothing will answer these now; let the callers recover.
          pendingMesh.current?.reject(new Error(message.message))
          pendingMesh.current = null
          rejectPendingRenders(message.message)
          setError(message.message)
          setStatus('hata')
          break
      }
    }

    worker.onerror = (event) => {
      rejectPendingRenders(event.message || 'Eğitim çalışanı beklenmedik şekilde durdu.')
      setError(event.message || 'Eğitim çalışanında beklenmeyen bir hata oluştu.')
      setStatus('hata')
    }

    return () => {
      worker.terminate()
      workerRef.current = null
      rejectPendingRenders('Eğitim çalışanı kapatıldı.')
    }
  }, [rejectPendingRenders])

  const send = useCallback((message: WorkerRequest, transfer: Transferable[] = []) => {
    workerRef.current?.postMessage(message, transfer)
  }, [])

  const initSynthetic = useCallback(
    (preset: QualityPreset, options: SyntheticOptions) => {
      setStatus('hazirlaniyor')
      setError(null)
      send({ type: 'initSynthetic', preset, options })
    },
    [send],
  )

  const initPhotos = useCallback(
    (
      preset: QualityPreset,
      views: SerializedView[],
      aabbSize: number,
      background: [number, number, number],
    ) => {
      setStatus('hazirlaniyor')
      setError(null)
      // Buffers are transferred; the caller must not reuse them afterwards.
      const transfer = views.flatMap((v) => [v.pixels.buffer, v.pose.buffer])
      send({ type: 'initPhotos', preset, views, aabbSize, background }, transfer)
    },
    [send],
  )

  const start = useCallback(() => {
    setStatus('egitiliyor')
    send({ type: 'start' })
  }, [send])

  const pause = useCallback(() => {
    setStatus('hazir')
    send({ type: 'pause' })
  }, [send])

  const reset = useCallback(() => {
    send({ type: 'reset' })
    setStatus('bos')
    setStats(null)
    setHistory([])
    setThumbnails([])
    setPreview(null)
    setPreviewStep(0)
    setParamCount(0)
    setViewCount(0)
    setSnapshot(null)
  }, [send])

  const setPreviewPose = useCallback(
    (pose: Mat4, fovDegrees: number) => {
      send({ type: 'setPreviewPose', pose: new Float32Array(pose), fovDegrees })
    },
    [send],
  )

  const renderView = useCallback<NerfWorkerApi['renderView']>(
    ({ pose, fovDegrees, width, height, mode = 'color', samplesPerRay }) =>
      new Promise((resolve, reject) => {
        const requestId = ++requestIdRef.current
        pendingRenders.current.set(requestId, { resolve, reject })
        send({
          type: 'render',
          requestId,
          pose: new Float32Array(pose),
          fovDegrees,
          width,
          height,
          mode,
          samplesPerRay,
        })
      }),
    [send],
  )

  const requestSnapshot = useCallback(() => {
    send({ type: 'gpuSnapshot' })
  }, [send])

  const extractMesh = useCallback<NerfWorkerApi['extractMesh']>(
    (options, onProgress) =>
      new Promise((resolve, reject) => {
        pendingMesh.current = { resolve, reject, onProgress }
        send({ type: 'extractMesh', options })
      }),
    [send],
  )

  const exportWeights = useCallback<NerfWorkerApi['exportWeights']>(
    () =>
      new Promise((resolve) => {
        pendingWeights.current = resolve
        send({ type: 'exportWeights' })
      }),
    [send],
  )

  return {
    status,
    error,
    stats,
    history,
    thumbnails,
    preview,
    previewStep,
    paramCount,
    viewCount,
    pointsPerStep,
    initSynthetic,
    initPhotos,
    start,
    pause,
    reset,
    setPreviewPose,
    renderView,
    exportWeights,
    snapshot,
    requestSnapshot,
    extractMesh,
  }
}
