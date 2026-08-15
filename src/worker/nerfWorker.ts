/// <reference lib="webworker" />
/**
 * Training worker.
 *
 * Everything expensive lives here so the UI thread never blocks: the model, the
 * optimiser and all rendering. The loop runs a handful of steps, then yields to
 * the event loop so incoming messages (pause, a new camera angle to render) are
 * handled promptly instead of after the whole run.
 */
import { defaultIntrinsics, orbitPose } from '../nerf/camera'
import { NerfTrainer } from '../nerf/trainer'
import { buildSyntheticDataset } from '../nerf/syntheticScene'
import type { Dataset, Intrinsics, Mat4, TrainingView, TrainStats } from '../nerf/types'
import { QUALITY_PRESETS } from '../nerf/types'
import type { SerializedView, WorkerRequest, WorkerResponse } from './protocol'

/** Steps between yields. Small enough that pausing feels instant. */
const STEPS_PER_SLICE = 4
/** Minimum gap between stats messages, milliseconds. */
const STATS_INTERVAL_MS = 200
/** Steps between automatic training-progress previews. */
const PREVIEW_EVERY = 25
const PREVIEW_SIZE = 72

let trainer: NerfTrainer | null = null
let running = false
let loopScheduled = false
let lastStatsAt = 0
let lastStats: TrainStats | null = null
let previewPose: Mat4 = orbitPose(0.9, 0.35, 3.6)
let previewFov = 42
/** Render requests queued while a training slice was in flight. */
const renderQueue: Array<Extract<WorkerRequest, { type: 'render' }>> = []

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  ;(self as unknown as Worker).postMessage(message, transfer)
}

function fail(error: unknown): void {
  post({ type: 'error', message: error instanceof Error ? error.message : String(error) })
}

/** Converts float RGB pixels into RGBA bytes for a canvas. */
function toRgba(pixels: Float32Array, width: number, height: number): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < width * height; i++) {
    rgba[i * 4] = pixels[i * 3] * 255
    rgba[i * 4 + 1] = pixels[i * 3 + 1] * 255
    rgba[i * 4 + 2] = pixels[i * 3 + 2] * 255
    rgba[i * 4 + 3] = 255
  }
  return rgba
}

function announceReady(dataset: Dataset): void {
  const thumbnails = dataset.views.slice(0, 60).map((view) => ({
    id: view.id,
    width: view.width,
    height: view.height,
    rgba: toRgba(view.pixels, view.width, view.height),
  }))
  post(
    {
      type: 'ready',
      viewCount: dataset.views.length,
      paramCount: trainer?.field.paramCount ?? 0,
      resolution: dataset.views[0]?.width ?? 0,
      thumbnails,
    },
    thumbnails.map((t) => t.rgba.buffer),
  )
}

function deserializeView(v: SerializedView): TrainingView {
  const intrinsics: Intrinsics = { focal: v.focal, cx: v.cx, cy: v.cy }
  return {
    id: v.id,
    width: v.width,
    height: v.height,
    pixels: v.pixels,
    pose: v.pose,
    intrinsics,
  }
}

function handleRender(request: Extract<WorkerRequest, { type: 'render' }>): void {
  if (!trainer) return
  const started = performance.now()
  const intrinsics = defaultIntrinsics(request.width, request.height, request.fovDegrees)
  const result = trainer.render(request.pose, intrinsics, {
    width: request.width,
    height: request.height,
    mode: request.mode,
    samplesPerRay: request.samplesPerRay,
  })
  post(
    {
      type: 'render',
      requestId: request.requestId,
      width: result.width,
      height: result.height,
      rgba: result.rgba,
      elapsedMs: performance.now() - started,
    },
    [result.rgba.buffer],
  )
}

/**
 * Renders everything queued while the last training slice was running.
 *
 * Every request gets a reply, including ones that look superseded: the UI holds
 * a promise per request, and silently dropping one would leave the viewer
 * waiting forever. The client keeps at most one request in flight, so this
 * cannot pile up.
 */
function flushRenderQueue(): void {
  while (renderQueue.length > 0) {
    handleRender(renderQueue.shift()!)
  }
}

function sendPreview(): void {
  if (!trainer) return
  const intrinsics = defaultIntrinsics(PREVIEW_SIZE, PREVIEW_SIZE, previewFov)
  const result = trainer.render(previewPose, intrinsics, {
    width: PREVIEW_SIZE,
    height: PREVIEW_SIZE,
  })
  post(
    {
      type: 'preview',
      step: trainer.step,
      width: result.width,
      height: result.height,
      rgba: result.rgba,
    },
    [result.rgba.buffer],
  )
}

function runSlice(): void {
  loopScheduled = false
  if (!trainer || !running) return

  try {
    let stats = null
    for (let i = 0; i < STEPS_PER_SLICE && running; i++) {
      const previousStep = trainer.step
      stats = trainer.trainStep()
      lastStats = stats
      if (Math.floor(previousStep / PREVIEW_EVERY) !== Math.floor(trainer.step / PREVIEW_EVERY)) {
        sendPreview()
      }
    }

    const now = performance.now()
    if (stats && now - lastStatsAt >= STATS_INTERVAL_MS) {
      lastStatsAt = now
      post({ type: 'stats', stats, pointsPerStep: trainer.lastPointCount, running })
    }

    flushRenderQueue()
  } catch (error) {
    running = false
    fail(error)
    return
  }

  scheduleLoop()
}

function scheduleLoop(): void {
  if (loopScheduled || !running) return
  loopScheduled = true
  // setTimeout rather than a tight loop: it lets queued messages run first.
  setTimeout(runSlice, 0)
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data
  try {
    switch (request.type) {
      case 'initSynthetic': {
        const preset = QUALITY_PRESETS[request.preset]
        const dataset = buildSyntheticDataset(request.options)
        trainer = new NerfTrainer(dataset, preset.model, preset.train)
        running = false
        previewPose = orbitPose(0.9, 0.35, request.options.radius)
        previewFov = request.options.fovDegrees
        announceReady(dataset)
        break
      }
      case 'initPhotos': {
        const preset = QUALITY_PRESETS[request.preset]
        const dataset: Dataset = {
          views: request.views.map(deserializeView),
          aabbSize: request.aabbSize,
          background: request.background,
        }
        if (dataset.views.length === 0) throw new Error('Eğitim için en az bir fotoğraf gerekiyor.')
        trainer = new NerfTrainer(dataset, preset.model, preset.train)
        running = false
        announceReady(dataset)
        break
      }
      case 'start': {
        if (!trainer) throw new Error('Model henüz hazır değil.')
        running = true
        scheduleLoop()
        break
      }
      case 'pause': {
        running = false
        // Echo the last real measurement rather than inventing zeros, so the
        // loss chart does not get a spurious dip at every pause.
        if (trainer && lastStats) {
          post({
            type: 'stats',
            stats: lastStats,
            pointsPerStep: trainer.lastPointCount,
            running: false,
          })
        }
        break
      }
      case 'reset': {
        running = false
        trainer = null
        lastStats = null
        renderQueue.length = 0
        break
      }
      case 'setPreviewPose': {
        previewPose = request.pose
        previewFov = request.fovDegrees
        break
      }
      case 'render': {
        if (!trainer) return
        // While training, queue it so the current slice finishes first; when
        // idle, render immediately.
        if (running) renderQueue.push(request)
        else handleRender(request)
        break
      }
      case 'gpuSnapshot': {
        if (!trainer) return
        const snapshot = trainer.exportGpuSnapshot()
        // The arrays are fresh copies, so transferring them is safe and keeps
        // multi-megabyte weight updates off the structured-clone path.
        post({ type: 'snapshot', snapshot }, [
          snapshot.gridParams.buffer,
          snapshot.layerWeights.buffer,
          snapshot.occupancy.buffer,
        ])
        break
      }
      case 'exportWeights': {
        if (!trainer) throw new Error('Dışa aktarılacak model yok.')
        const data = trainer.field.exportWeights()
        post(
          { type: 'weights', data, step: trainer.step, paramCount: trainer.field.paramCount },
          [data.buffer],
        )
        break
      }
    }
  } catch (error) {
    fail(error)
  }
}
