/**
 * Ties the pieces together: samples rays from the training photos, renders
 * them, computes the photometric loss and updates the field.
 */
import { rayForPixel } from './camera'
import { RadianceField } from './field'
import { OccupancyGrid } from './occupancy'
import { makeRng } from './random'
import type {
  Dataset,
  Intrinsics,
  Mat4,
  ModelConfig,
  RenderResult,
  TrainConfig,
  TrainStats,
} from './types'
import { VolumeRenderer } from './volumeRender'

/** Rays rendered per chunk when synthesising a full image. */
const RENDER_CHUNK = 512

/**
 * Steps of training before empty-space skipping is allowed to kick in. Pruning
 * a volume the model has not begun to explain yet risks deleting geometry it
 * was about to grow.
 */
const OCCUPANCY_WARMUP_STEPS = 64

export interface RenderOptions {
  width: number
  height: number
  /** Samples per ray; higher is sharper and slower. */
  samplesPerRay?: number
  /** Return a colour-mapped depth image instead of the radiance. */
  mode?: 'color' | 'depth'
}

export class NerfTrainer {
  readonly field: RadianceField
  readonly occupancy: OccupancyGrid
  readonly dataset: Dataset
  config: TrainConfig
  step = 0

  private readonly renderer: VolumeRenderer
  private readonly renderRenderers = new Map<number, VolumeRenderer>()
  private readonly rng: () => number

  private readonly origins: Float32Array
  private readonly dirs: Float32Array
  private readonly targets: Float32Array
  private readonly gradColor: Float32Array
  private gradSigma: Float32Array
  private gradRgb: Float32Array

  private readonly rayOrigin = new Float32Array(3)
  private readonly rayDir = new Float32Array(3)

  /** Rolling average of samples that survived pruning, for the UI. */
  lastPointCount = 0

  constructor(dataset: Dataset, modelConfig: ModelConfig, trainConfig: TrainConfig) {
    this.dataset = dataset
    this.config = trainConfig
    this.field = new RadianceField(modelConfig)
    this.occupancy = new OccupancyGrid(32, 0.8)
    this.rng = makeRng(modelConfig.seed ^ 0x9e3779b9)

    const R = trainConfig.raysPerStep
    this.renderer = new VolumeRenderer(
      R, trainConfig.samplesPerRay, dataset.aabbSize, dataset.background,
    )
    this.origins = new Float32Array(R * 3)
    this.dirs = new Float32Array(R * 3)
    this.targets = new Float32Array(R * 3)
    this.gradColor = new Float32Array(R * 3)
    const maxPoints = R * trainConfig.samplesPerRay
    this.gradSigma = new Float32Array(maxPoints)
    this.gradRgb = new Float32Array(maxPoints * 3)
  }

  /** True once the occupancy grid holds a meaningful estimate of the scene. */
  private get occupancyActive(): boolean {
    return this.config.occupancyRefreshInterval > 0 && this.step >= OCCUPANCY_WARMUP_STEPS
  }

  /** Learning rate with a short warm-up followed by exponential decay. */
  private currentLr(): number {
    const base = this.config.learningRate
    const warmup = Math.min(1, (this.step + 1) / 100)
    const decay = Math.pow(0.33, this.step / 4000)
    return base * warmup * decay
  }

  /** Picks `count` random pixels across all views and builds their rays. */
  private sampleRays(count: number): void {
    const views = this.dataset.views
    for (let r = 0; r < count; r++) {
      const view = views[(this.rng() * views.length) | 0]
      const px = (this.rng() * view.width) | 0
      const py = (this.rng() * view.height) | 0
      rayForPixel(view.pose, view.intrinsics, px, py, this.rayOrigin, this.rayDir)
      this.origins[r * 3] = this.rayOrigin[0]
      this.origins[r * 3 + 1] = this.rayOrigin[1]
      this.origins[r * 3 + 2] = this.rayOrigin[2]
      this.dirs[r * 3] = this.rayDir[0]
      this.dirs[r * 3 + 1] = this.rayDir[1]
      this.dirs[r * 3 + 2] = this.rayDir[2]
      const pixelBase = (py * view.width + px) * 3
      this.targets[r * 3] = view.pixels[pixelBase]
      this.targets[r * 3 + 1] = view.pixels[pixelBase + 1]
      this.targets[r * 3 + 2] = view.pixels[pixelBase + 2]
    }
  }

  /** One optimisation step over a fresh batch of rays. */
  trainStep(): TrainStats {
    const t0 = performance.now()
    const R = this.config.raysPerStep
    this.sampleRays(R)

    const useOccupancy = this.occupancyActive
    const stats = this.renderer.sample(
      this.origins, this.dirs, R, useOccupancy ? this.occupancy : null, this.rng,
    )
    const n = stats.points
    this.lastPointCount = n

    this.field.forward(this.renderer.pos, this.renderer.dir, n)
    this.renderer.composite(this.field.sigma, this.field.rgb, R)

    // Mean squared error over rays and channels.
    let sse = 0
    const scale = 2 / (R * 3)
    for (let i = 0; i < R * 3; i++) {
      const diff = this.renderer.color[i] - this.targets[i]
      sse += diff * diff
      this.gradColor[i] = scale * diff
    }
    const mse = sse / (R * 3)

    if (n > 0) {
      if (this.gradSigma.length < n) {
        this.gradSigma = new Float32Array(n)
        this.gradRgb = new Float32Array(n * 3)
      }
      this.gradSigma.fill(0, 0, n)
      this.gradRgb.fill(0, 0, n * 3)
      this.renderer.backward(this.gradColor, R, this.field.rgb, this.gradSigma, this.gradRgb)

      // Cauchy sparsity prior: punishes faint fog everywhere, barely touches
      // solid surfaces. Without it the model happily explains photos with a
      // cloud of semi-transparent "floaters" that look terrible from new angles.
      const lambda = this.config.sparsityWeight
      if (lambda > 0) {
        const k = lambda / n
        for (let i = 0; i < n; i++) {
          const s = this.field.sigma[i]
          this.gradSigma[i] += (k * 4 * s) / (1 + 2 * s * s)
        }
      }

      this.field.zeroGrad()
      this.field.backward(this.gradSigma, this.gradRgb, n)
      this.field.step(this.currentLr())
    }

    this.step++
    if (
      this.config.occupancyRefreshInterval > 0 &&
      this.step >= OCCUPANCY_WARMUP_STEPS &&
      this.step % this.config.occupancyRefreshInterval === 0
    ) {
      const sampleDelta = (2 * this.dataset.aabbSize) / this.config.samplesPerRay
      this.occupancy.refresh(this.field, sampleDelta, this.rng)
    }

    return {
      step: this.step,
      loss: mse,
      psnr: mse > 0 ? -10 * Math.log10(mse) : 99,
      occupancy: this.occupancy.occupiedFraction,
      stepMs: performance.now() - t0,
    }
  }

  /** Synthesises a novel view from an arbitrary camera pose. */
  render(pose: Mat4, intrinsics: Intrinsics, opts: RenderOptions): RenderResult {
    const { width, height } = opts
    const samples = opts.samplesPerRay ?? this.config.samplesPerRay
    // The viewer alternates between a few sample counts as it refines a frame,
    // so keep one renderer per count instead of reallocating megabytes per pass.
    let rr = this.renderRenderers.get(samples)
    if (!rr) {
      rr = new VolumeRenderer(RENDER_CHUNK, samples, this.dataset.aabbSize, this.dataset.background)
      this.renderRenderers.set(samples, rr)
    }
    rr.aabbSize = this.dataset.aabbSize
    rr.background.set(this.dataset.background)

    const rgba = new Uint8ClampedArray(width * height * 4)
    const origins = new Float32Array(RENDER_CHUNK * 3)
    const dirs = new Float32Array(RENDER_CHUNK * 3)
    const useOccupancy = this.occupancyActive
    const total = width * height
    const depthValues = opts.mode === 'depth' ? new Float32Array(total) : null
    const accValues = opts.mode === 'depth' ? new Float32Array(total) : null
    // Deterministic mid-stratum sampling: no noise between frames.
    const midpoint = () => 0.5

    for (let start = 0; start < total; start += RENDER_CHUNK) {
      const count = Math.min(RENDER_CHUNK, total - start)
      for (let i = 0; i < count; i++) {
        const pixel = start + i
        const px = pixel % width
        const py = (pixel / width) | 0
        rayForPixel(pose, intrinsics, px, py, this.rayOrigin, this.rayDir)
        origins[i * 3] = this.rayOrigin[0]
        origins[i * 3 + 1] = this.rayOrigin[1]
        origins[i * 3 + 2] = this.rayOrigin[2]
        dirs[i * 3] = this.rayDir[0]
        dirs[i * 3 + 1] = this.rayDir[1]
        dirs[i * 3 + 2] = this.rayDir[2]
      }
      const stats = rr.sample(origins, dirs, count, useOccupancy ? this.occupancy : null, midpoint)
      this.field.forward(rr.pos, rr.dir, stats.points)
      rr.composite(this.field.sigma, this.field.rgb, count, true)

      for (let i = 0; i < count; i++) {
        const pixel = start + i
        if (depthValues && accValues) {
          depthValues[pixel] = rr.depth[i]
          accValues[pixel] = rr.acc[i]
        } else {
          const o = pixel * 4
          rgba[o] = rr.color[i * 3] * 255
          rgba[o + 1] = rr.color[i * 3 + 1] * 255
          rgba[o + 2] = rr.color[i * 3 + 2] * 255
          rgba[o + 3] = 255
        }
      }
    }

    if (depthValues && accValues) {
      writeDepthImage(depthValues, accValues, rgba)
    }
    return { width, height, rgba }
  }

  /** Re-renders one of the training views, for side-by-side comparison. */
  renderTrainingView(index: number, opts: RenderOptions): RenderResult {
    const view = this.dataset.views[index]
    const scaleX = opts.width / view.width
    const scaleY = opts.height / view.height
    return this.render(
      view.pose,
      {
        focal: view.intrinsics.focal * scaleX,
        cx: view.intrinsics.cx * scaleX,
        cy: view.intrinsics.cy * scaleY,
      },
      opts,
    )
  }
}

/** Normalises depth to the visible range and applies a turbo-ish ramp. */
function writeDepthImage(
  depth: Float32Array,
  acc: Float32Array,
  rgba: Uint8ClampedArray,
): void {
  let min = Infinity
  let max = -Infinity
  for (let i = 0; i < depth.length; i++) {
    if (acc[i] < 0.1) continue
    const d = depth[i] / Math.max(acc[i], 1e-6)
    if (d < min) min = d
    if (d > max) max = d
  }
  if (!isFinite(min) || max <= min) {
    min = 0
    max = 1
  }
  for (let i = 0; i < depth.length; i++) {
    const o = i * 4
    if (acc[i] < 0.1) {
      rgba[o] = 12; rgba[o + 1] = 14; rgba[o + 2] = 22; rgba[o + 3] = 255
      continue
    }
    const d = depth[i] / Math.max(acc[i], 1e-6)
    const t = 1 - Math.min(1, Math.max(0, (d - min) / (max - min)))
    // Simple blue -> cyan -> yellow ramp: near is warm, far is cold.
    rgba[o] = 255 * Math.min(1, Math.max(0, 1.5 * t - 0.4))
    rgba[o + 1] = 255 * Math.min(1, Math.max(0, 1.4 * t - 0.1))
    rgba[o + 2] = 255 * Math.min(1, Math.max(0, 1.2 - 1.4 * t))
    rgba[o + 3] = 255
  }
}
