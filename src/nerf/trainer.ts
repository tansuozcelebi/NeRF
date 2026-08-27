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
import type { GridLayout } from './hashGrid'
import { extractSurface, sealBoundary, type ExtractedMesh } from './meshExtract'
import { VolumeRenderer } from './volumeRender'

/** A mesh pulled out of the density field, ready to be written to a file. */
export interface ExportedMesh extends ExtractedMesh {
  /** Per-vertex colour, 3 bytes each. */
  colors: Uint8Array
  /** Density threshold the surface was taken at. */
  isoLevel: number
  boundsMin: [number, number, number]
  boundsMax: [number, number, number]
}

export interface MeshExportOptions {
  /** Samples along each axis. Cost grows with the cube of this. */
  resolution: number
  /**
   * Scales the automatically chosen density threshold. 1 keeps the suggestion;
   * lower values include more of the faint haze, higher values keep only the
   * solid core.
   */
  isoScale: number
  boundsMin: [number, number, number]
  boundsMax: [number, number, number]
  /** Called with progress in [0,1] while the grid is being sampled. */
  onProgress?: (fraction: number) => void
}

/** A trained model in the form the GPU renderer consumes. */
export interface GpuSnapshot {
  step: number
  modelConfig: ModelConfig
  gridLayout: GridLayout
  gridParams: Float32Array
  layerWeights: Float32Array
  occupancy: Uint8Array
  occupancyResolution: number
  useOccupancy: boolean
  aabbSize: number
  background: [number, number, number]
}

/** Rays rendered per chunk when synthesising a full image. */
const RENDER_CHUNK = 512

/**
 * Steps of training before empty-space skipping is allowed to kick in. Pruning
 * a volume the model has not begun to explain yet risks deleting geometry it
 * was about to grow.
 */
const OCCUPANCY_WARMUP_STEPS = 64

/** Rays drawn per held-out evaluation. Enough to be stable, cheap enough to run often. */
const VALIDATION_RAYS = 512
/** Steps between held-out evaluations. */
const VALIDATION_INTERVAL = 25
/** Below this many views a hold-out split would starve training, so it is skipped. */
const MIN_VIEWS_FOR_HOLDOUT = 6

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

  /** Views the optimiser is allowed to see, and the ones kept back from it. */
  readonly trainViews: number[]
  readonly holdOutViews: number[]

  private readonly valRenderer: VolumeRenderer
  private readonly valOrigins = new Float32Array(VALIDATION_RAYS * 3)
  private readonly valDirs = new Float32Array(VALIDATION_RAYS * 3)
  private readonly valTargets = new Float32Array(VALIDATION_RAYS * 3)
  private readonly valRng: () => number
  private lastValidationMse: number | null = null

  constructor(dataset: Dataset, modelConfig: ModelConfig, trainConfig: TrainConfig) {
    this.dataset = dataset
    this.config = trainConfig
    this.field = new RadianceField(modelConfig)
    this.occupancy = new OccupancyGrid(32, 0.8)
    this.rng = makeRng(modelConfig.seed ^ 0x9e3779b9)

    // Split the views before anything else touches them.
    const holdOutEvery = trainConfig.holdOutEvery
    const canHoldOut = holdOutEvery > 1 && dataset.views.length >= MIN_VIEWS_FOR_HOLDOUT
    this.trainViews = []
    this.holdOutViews = []
    dataset.views.forEach((_, i) => {
      // Offset by half a period so the first view — often the one users look at
      // first — stays in the training set.
      const isHeldOut = canHoldOut && i % holdOutEvery === (holdOutEvery >> 1)
      ;(isHeldOut ? this.holdOutViews : this.trainViews).push(i)
    })

    const R = trainConfig.raysPerStep
    this.renderer = new VolumeRenderer(
      R, trainConfig.samplesPerRay, dataset.aabbSize, dataset.background,
    )
    this.valRenderer = new VolumeRenderer(
      VALIDATION_RAYS, trainConfig.samplesPerRay, dataset.aabbSize, dataset.background,
    )
    this.valRng = makeRng(modelConfig.seed ^ 0x51ed270b)
    this.origins = new Float32Array(R * 3)
    this.dirs = new Float32Array(R * 3)
    this.targets = new Float32Array(R * 3)
    this.gradColor = new Float32Array(R * 3)
    const maxPoints = R * trainConfig.samplesPerRay
    this.gradSigma = new Float32Array(maxPoints)
    this.gradRgb = new Float32Array(maxPoints * 3)
  }

  /** True once the occupancy grid holds a meaningful estimate of the scene. */
  get occupancyActive(): boolean {
    return this.config.occupancyRefreshInterval > 0 && this.step >= OCCUPANCY_WARMUP_STEPS
  }

  /** Learning rate with a short warm-up followed by exponential decay. */
  private currentLr(): number {
    const base = this.config.learningRate
    const warmup = Math.min(1, (this.step + 1) / 100)
    const decay = Math.pow(0.33, this.step / 4000)
    return base * warmup * decay
  }

  /**
   * Picks `count` random pixels from the given views and builds their rays.
   * Writes into the supplied buffers so training and evaluation can share it.
   */
  private sampleRays(
    count: number,
    viewIndices: number[],
    rng: () => number,
    origins: Float32Array,
    dirs: Float32Array,
    targets: Float32Array,
  ): void {
    const views = this.dataset.views
    for (let r = 0; r < count; r++) {
      const view = views[viewIndices[(rng() * viewIndices.length) | 0]]
      const px = (rng() * view.width) | 0
      const py = (rng() * view.height) | 0
      rayForPixel(view.pose, view.intrinsics, px, py, this.rayOrigin, this.rayDir)
      origins[r * 3] = this.rayOrigin[0]
      origins[r * 3 + 1] = this.rayOrigin[1]
      origins[r * 3 + 2] = this.rayOrigin[2]
      dirs[r * 3] = this.rayDir[0]
      dirs[r * 3 + 1] = this.rayDir[1]
      dirs[r * 3 + 2] = this.rayDir[2]
      const pixelBase = (py * view.width + px) * 3
      targets[r * 3] = view.pixels[pixelBase]
      targets[r * 3 + 1] = view.pixels[pixelBase + 1]
      targets[r * 3 + 2] = view.pixels[pixelBase + 2]
    }
  }

  /**
   * Mean squared error on rays from views the optimiser never sees.
   * Forward pass only — no gradients, nothing learned from these pixels.
   */
  validate(): number | null {
    if (this.holdOutViews.length === 0) return null
    const R = VALIDATION_RAYS
    this.sampleRays(R, this.holdOutViews, this.valRng, this.valOrigins, this.valDirs, this.valTargets)
    const stats = this.valRenderer.sample(
      this.valOrigins, this.valDirs, R,
      this.occupancyActive ? this.occupancy : null,
      // Mid-stratum: the metric should not jitter with the sampling noise.
      () => 0.5,
    )
    this.field.forward(this.valRenderer.pos, this.valRenderer.dir, stats.points)
    this.valRenderer.composite(this.field.sigma, this.field.rgb, R, true)

    let sse = 0
    for (let i = 0; i < R * 3; i++) {
      const diff = this.valRenderer.color[i] - this.valTargets[i]
      sse += diff * diff
    }
    return sse / (R * 3)
  }

  /** One optimisation step over a fresh batch of rays. */
  trainStep(): TrainStats {
    const t0 = performance.now()
    const R = this.config.raysPerStep
    this.sampleRays(R, this.trainViews, this.rng, this.origins, this.dirs, this.targets)

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

    if (this.holdOutViews.length > 0 && this.step % VALIDATION_INTERVAL === 0) {
      this.lastValidationMse = this.validate()
    }
    const valMse = this.lastValidationMse

    return {
      step: this.step,
      loss: mse,
      psnr: mse > 0 ? -10 * Math.log10(mse) : 99,
      validationPsnr: valMse === null ? null : valMse > 0 ? -10 * Math.log10(valMse) : 99,
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

  /**
   * Everything the GPU renderer needs to reproduce this model: the hash-grid
   * features, the dense layers and the occupancy mask. All copies — the worker
   * transfers these away and must keep its own.
   */
  exportGpuSnapshot(): GpuSnapshot {
    return {
      step: this.step,
      modelConfig: this.field.config,
      gridLayout: this.field.grid.describeLayout(),
      gridParams: new Float32Array(this.field.grid.params),
      layerWeights: this.field.exportLayerWeights(),
      occupancy: new Uint8Array(this.occupancy.occupied),
      occupancyResolution: this.occupancy.resolution,
      useOccupancy: this.occupancyActive,
      aabbSize: this.dataset.aabbSize,
      background: this.dataset.background,
    }
  }

  /**
   * Samples the density field on a regular grid.
   *
   * Evaluated one z-slab at a time: a 128^3 grid is two million network
   * evaluations, and doing it in one batch would allocate hundreds of megabytes
   * of intermediate activations.
   */
  private sampleDensityGrid(options: MeshExportOptions): Float32Array {
    const { resolution: res, boundsMin, boundsMax } = options
    const values = new Float32Array(res * res * res)
    const slab = new Float32Array(res * res * 3)
    const out = new Float32Array(res * res)
    const size = this.dataset.aabbSize
    const invExtent = 1 / (2 * size)

    for (let z = 0; z < res; z++) {
      let p = 0
      const wz = boundsMin[2] + ((boundsMax[2] - boundsMin[2]) * z) / (res - 1)
      for (let y = 0; y < res; y++) {
        const wy = boundsMin[1] + ((boundsMax[1] - boundsMin[1]) * y) / (res - 1)
        for (let x = 0; x < res; x++) {
          const wx = boundsMin[0] + ((boundsMax[0] - boundsMin[0]) * x) / (res - 1)
          // The field works in normalised [0,1] box coordinates.
          slab[p++] = (wx + size) * invExtent
          slab[p++] = (wy + size) * invExtent
          slab[p++] = (wz + size) * invExtent
        }
      }
      this.field.densityOnly(slab, res * res, out)
      values.set(out.subarray(0, res * res), z * res * res)
      options.onProgress?.((z + 1) / res)
    }

    sealBoundary(values, res)
    return values
  }

  /**
   * Picks a density threshold from the samples themselves.
   *
   * A fixed number cannot work across scenes: density is an unbounded quantity
   * and its scale depends on how the model happened to fit. Taking a high
   * percentile instead means "keep the densest couple of percent of space",
   * which is a reasonable description of a surface in almost any scene.
   */
  private suggestIsoLevel(values: Float32Array): number {
    const sample: number[] = []
    // A stride keeps the sort cheap on large grids without biasing the estimate.
    const stride = Math.max(1, Math.floor(values.length / 200000))
    for (let i = 0; i < values.length; i += stride) sample.push(values[i])
    sample.sort((a, b) => a - b)
    const index = Math.floor(sample.length * 0.985)
    return Math.max(1e-3, sample[Math.min(index, sample.length - 1)])
  }

  /** Extracts a coloured triangle mesh from the learned density field. */
  extractMesh(options: MeshExportOptions): ExportedMesh {
    const values = this.sampleDensityGrid(options)
    const isoLevel = this.suggestIsoLevel(values) * options.isoScale
    const mesh = extractSurface(values, {
      resolution: options.resolution,
      isoLevel,
      boundsMin: options.boundsMin,
      boundsMax: options.boundsMax,
    })

    return {
      ...mesh,
      colors: this.colourVertices(mesh),
      isoLevel,
      boundsMin: options.boundsMin,
      boundsMax: options.boundsMax,
    }
  }

  /**
   * Asks the colour network what each vertex looks like, viewed head-on.
   *
   * NeRF colour is view-dependent, so a mesh has to commit to one direction.
   * Looking straight down the surface normal is the natural choice: it is the
   * angle that shows the surface's own colour rather than a glancing highlight.
   */
  private colourVertices(mesh: ExtractedMesh): Uint8Array {
    const count = mesh.vertexCount
    const colors = new Uint8Array(count * 3)
    if (count === 0) return colors

    // Area-weighted vertex normals, accumulated from the faces.
    const normals = new Float32Array(count * 3)
    for (let t = 0; t < mesh.triangleCount; t++) {
      const ia = mesh.indices[t * 3], ib = mesh.indices[t * 3 + 1], ic = mesh.indices[t * 3 + 2]
      const ax = mesh.positions[ia * 3], ay = mesh.positions[ia * 3 + 1], az = mesh.positions[ia * 3 + 2]
      const e1x = mesh.positions[ib * 3] - ax
      const e1y = mesh.positions[ib * 3 + 1] - ay
      const e1z = mesh.positions[ib * 3 + 2] - az
      const e2x = mesh.positions[ic * 3] - ax
      const e2y = mesh.positions[ic * 3 + 1] - ay
      const e2z = mesh.positions[ic * 3 + 2] - az
      const nx = e1y * e2z - e1z * e2y
      const ny = e1z * e2x - e1x * e2z
      const nz = e1x * e2y - e1y * e2x
      for (const index of [ia, ib, ic]) {
        normals[index * 3] += nx
        normals[index * 3 + 1] += ny
        normals[index * 3 + 2] += nz
      }
    }

    const size = this.dataset.aabbSize
    const invExtent = 1 / (2 * size)
    const CHUNK = 4096
    const pos = new Float32Array(CHUNK * 3)
    const dir = new Float32Array(CHUNK * 3)

    for (let start = 0; start < count; start += CHUNK) {
      const n = Math.min(CHUNK, count - start)
      for (let i = 0; i < n; i++) {
        const v = start + i
        pos[i * 3] = (mesh.positions[v * 3] + size) * invExtent
        pos[i * 3 + 1] = (mesh.positions[v * 3 + 1] + size) * invExtent
        pos[i * 3 + 2] = (mesh.positions[v * 3 + 2] + size) * invExtent
        let nx = normals[v * 3], ny = normals[v * 3 + 1], nz = normals[v * 3 + 2]
        const len = Math.hypot(nx, ny, nz)
        if (len < 1e-9) { nx = 0; ny = 0; nz = 1 } else { nx /= len; ny /= len; nz /= len }
        // Viewing direction points from the eye towards the surface.
        dir[i * 3] = -nx
        dir[i * 3 + 1] = -ny
        dir[i * 3 + 2] = -nz
      }
      this.field.forward(pos, dir, n)
      for (let i = 0; i < n * 3; i++) {
        colors[start * 3 + i] = Math.round(Math.min(1, Math.max(0, this.field.rgb[i])) * 255)
      }
    }
    return colors
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
      rgba[o] = 12; rgba[o + 1] = 8; rgba[o + 2] = 10; rgba[o + 3] = 255
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
