/** Message contract between the UI and the training worker. */
import type { QualityPreset, TrainStats } from '../nerf/types'

/** A training image shipped to the worker. Buffers are transferred, not copied. */
export interface SerializedView {
  id: string
  width: number
  height: number
  /** RGB in [0,1], 3 floats per pixel. */
  pixels: Float32Array
  /** Row-major camera-to-world 4x4. */
  pose: Float32Array
  focal: number
  cx: number
  cy: number
}

export interface SyntheticOptions {
  viewCount: number
  resolution: number
  radius: number
  fovDegrees: number
}

export type WorkerRequest =
  | { type: 'initSynthetic'; preset: QualityPreset; options: SyntheticOptions }
  | {
      type: 'initPhotos'
      preset: QualityPreset
      views: SerializedView[]
      aabbSize: number
      background: [number, number, number]
    }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'reset' }
  | {
      type: 'render'
      requestId: number
      pose: Float32Array
      fovDegrees: number
      width: number
      height: number
      mode: 'color' | 'depth'
      samplesPerRay?: number
    }
  | { type: 'setPreviewPose'; pose: Float32Array; fovDegrees: number }
  | { type: 'exportWeights' }

export type WorkerResponse =
  | {
      type: 'ready'
      viewCount: number
      paramCount: number
      resolution: number
      thumbnails: Array<{ id: string; width: number; height: number; rgba: Uint8ClampedArray }>
    }
  | { type: 'stats'; stats: TrainStats; pointsPerStep: number; running: boolean }
  | {
      type: 'render'
      requestId: number
      width: number
      height: number
      rgba: Uint8ClampedArray
      elapsedMs: number
    }
  | { type: 'preview'; step: number; width: number; height: number; rgba: Uint8ClampedArray }
  | { type: 'weights'; data: Float32Array; step: number; paramCount: number }
  | { type: 'error'; message: string }
