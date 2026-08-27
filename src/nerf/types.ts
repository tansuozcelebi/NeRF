/** Shared type definitions for the NeRF pipeline. */

/** A 4x4 camera-to-world matrix, row-major, 16 floats. */
export type Mat4 = Float32Array

/** Pinhole camera intrinsics for a single training image. */
export interface Intrinsics {
  /** Focal length in pixels (shared for x/y). */
  focal: number
  /** Principal point, in pixels. Defaults to the image centre. */
  cx: number
  cy: number
}

/**
 * A training view: a downsampled image plus the camera pose it was taken from.
 * `pixels` holds linear RGB in [0,1], row-major, 3 floats per pixel.
 */
export interface TrainingView {
  id: string
  width: number
  height: number
  pixels: Float32Array
  pose: Mat4
  intrinsics: Intrinsics
}

/** Everything the trainer needs to fit a scene. */
export interface Dataset {
  views: TrainingView[]
  /** Half-size of the axis-aligned bounding box the scene is fitted into. */
  aabbSize: number
  /** Background colour composited behind the volume, RGB in [0,1]. */
  background: [number, number, number]
}

export interface HashGridConfig {
  levels: number
  featuresPerLevel: number
  /** log2 of the hash table size per level. */
  log2TableSize: number
  baseResolution: number
  maxResolution: number
}

export interface TrainConfig {
  /** Rays drawn per optimisation step. */
  raysPerStep: number
  /** Samples drawn along each ray. */
  samplesPerRay: number
  learningRate: number
  /** Weight of the density-sparsity regulariser (encourages empty space). */
  sparsityWeight: number
  /** Steps between occupancy-grid refreshes; 0 disables pruning. */
  occupancyRefreshInterval: number
  /**
   * Every Nth view is kept out of training and used only for evaluation.
   * 0 disables the split.
   *
   * Without this the only number on screen is the training loss, and a NeRF can
   * drive that down while producing a mess from any angle it was not shown —
   * which is exactly the failure the method is supposed to avoid.
   */
  holdOutEvery: number
}

export interface ModelConfig {
  grid: HashGridConfig
  /** Width of the hidden layer in both MLPs. */
  hiddenSize: number
  /** Size of the geometry feature vector handed to the colour MLP. */
  geoFeatureSize: number
  /** Spherical-harmonics degree used to encode the view direction. */
  shDegree: number
  seed: number
}

export interface TrainStats {
  step: number
  loss: number
  psnr: number
  /**
   * PSNR on views the model has never been trained on, or null when the dataset
   * is too small to hold any out. This is the number that says whether the
   * reconstruction generalises; `psnr` above only says it memorised.
   */
  validationPsnr: number | null
  /** Fraction of occupancy-grid cells considered non-empty. */
  occupancy: number
  /** Milliseconds spent in the last optimisation step. */
  stepMs: number
}

/** A rendered image handed back from the worker. */
export interface RenderResult {
  width: number
  height: number
  /** RGBA bytes, ready for ImageData. */
  rgba: Uint8ClampedArray
}

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  grid: {
    levels: 12,
    featuresPerLevel: 2,
    log2TableSize: 15,
    baseResolution: 16,
    maxResolution: 256,
  },
  hiddenSize: 32,
  geoFeatureSize: 15,
  shDegree: 3,
  seed: 1337,
}

export const DEFAULT_TRAIN_CONFIG: TrainConfig = {
  raysPerStep: 512,
  samplesPerRay: 32,
  learningRate: 0.01,
  sparsityWeight: 5e-4,
  occupancyRefreshInterval: 16,
  holdOutEvery: 8,
}

export type QualityPreset = 'hizli' | 'dengeli' | 'kaliteli'

export interface PresetDefinition {
  label: string
  description: string
  model: ModelConfig
  train: TrainConfig
}

/**
 * Three points on the speed/quality curve. Measured on a mid-range laptop core:
 * roughly 10, 5 and 2 steps per second respectively. Everything runs on the CPU
 * in a worker, so these are honest single-thread numbers.
 */
export const QUALITY_PRESETS: Record<QualityPreset, PresetDefinition> = {
  hizli: {
    label: 'Hızlı',
    description: 'Küçük ağ, hızlı geri bildirim. Denemeler ve zayıf cihazlar için.',
    model: {
      ...DEFAULT_MODEL_CONFIG,
      hiddenSize: 24,
      grid: { ...DEFAULT_MODEL_CONFIG.grid, levels: 10 },
    },
    train: { ...DEFAULT_TRAIN_CONFIG, raysPerStep: 384, samplesPerRay: 24 },
  },
  dengeli: {
    label: 'Dengeli',
    description: 'Önerilen ayar. Makul sürede belirgin şekilde daha keskin sonuç.',
    model: DEFAULT_MODEL_CONFIG,
    train: DEFAULT_TRAIN_CONFIG,
  },
  kaliteli: {
    label: 'Kaliteli',
    description: 'Daha geniş ağ ve daha yoğun örnekleme. Sabır ister.',
    model: {
      ...DEFAULT_MODEL_CONFIG,
      hiddenSize: 48,
      grid: { ...DEFAULT_MODEL_CONFIG.grid, levels: 14, log2TableSize: 16, maxResolution: 384 },
    },
    train: { ...DEFAULT_TRAIN_CONFIG, raysPerStep: 640, samplesPerRay: 48 },
  },
}
