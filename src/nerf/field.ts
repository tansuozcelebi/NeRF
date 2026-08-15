/**
 * The radiance field itself: position -> (density, view-dependent colour).
 *
 *   position --[hash grid]--> features --[density MLP]--> density + geo-feature
 *   geo-feature + SH(direction) --[colour MLP]--> RGB
 *
 * The split follows Instant-NGP: density must not depend on the viewing
 * direction (otherwise geometry changes as you walk around), while colour must.
 */
import { Adam } from './adam'
import { HashGrid } from './hashGrid'
import { Linear, relu, reluBackward, sigmoid } from './mlp'
import { encodeDirection, shDim } from './sphericalHarmonics'
import type { ModelConfig } from './types'

/** Density is exp(raw); the raw value is clamped to keep exp finite. */
const RAW_DENSITY_CLAMP = 15

/**
 * exp(-1.5) ~ 0.22: a light haze the optimiser can carve geometry out of.
 *
 * The value has a hard lower bound: it must stay clearly above the occupancy
 * grid's cutoff (~0.08 at default settings), or the very first refresh prunes
 * the entire volume, after which no samples are drawn, no gradients flow, and
 * training is dead with no way back.
 */
const INITIAL_RAW_DENSITY = -1.5

export class RadianceField {
  readonly config: ModelConfig
  readonly grid: HashGrid
  private readonly densityHidden: Linear
  private readonly densityOut: Linear
  private readonly colorHidden: Linear
  private readonly colorOut: Linear

  private readonly gridOpt: Adam
  private readonly layerOpts: Adam[]
  private readonly layers: Linear[]

  readonly encDim: number
  readonly shDim: number
  readonly geoDim: number
  readonly colorInDim: number

  /** Density of the last forward pass, one value per point. */
  sigma = new Float32Array(0)
  /** RGB of the last forward pass, three values per point. */
  rgb = new Float32Array(0)

  private capacity = 0
  private enc = new Float32Array(0)
  private h1 = new Float32Array(0)
  private dout = new Float32Array(0)
  private colorIn = new Float32Array(0)
  private c1 = new Float32Array(0)
  private cpre = new Float32Array(0)
  private gradEnc = new Float32Array(0)
  private gradH1 = new Float32Array(0)
  private gradDOut = new Float32Array(0)
  private gradC1 = new Float32Array(0)
  private gradColorIn = new Float32Array(0)
  private gradCPre = new Float32Array(0)

  constructor(config: ModelConfig) {
    this.config = config
    this.grid = new HashGrid(config.grid, config.seed)
    this.encDim = this.grid.outputDim
    this.shDim = shDim(config.shDegree)
    this.geoDim = config.geoFeatureSize
    this.colorInDim = this.geoDim + this.shDim

    const h = config.hiddenSize
    this.densityHidden = new Linear(this.encDim, h, config.seed + 11)
    this.densityOut = new Linear(h, 1 + this.geoDim, config.seed + 23)
    this.colorHidden = new Linear(this.colorInDim, h, config.seed + 37)
    // Small last-layer weights keep the initial image a flat mid-grey rather
    // than saturated noise, which speeds up the first few hundred steps.
    this.colorOut = new Linear(h, 3, config.seed + 51, 0.05)
    // Start the volume almost transparent. Density is exp(raw), so a raw bias of
    // 0 would fill the whole box with sigma = 1 fog: the first images would be
    // an opaque soup, the occupancy grid could never prune anything, and the
    // optimiser would waste hundreds of steps digging back out.
    this.densityOut.bias[0] = INITIAL_RAW_DENSITY

    this.layers = [this.densityHidden, this.densityOut, this.colorHidden, this.colorOut]
    this.gridOpt = new Adam(this.grid.params, this.grid.grads)
    this.layerOpts = this.layers.map((l) => new Adam(l.params, l.grads))
    this.ensureCapacity(1024)
  }

  get paramCount(): number {
    return this.grid.params.length + this.layers.reduce((n, l) => n + l.paramCount, 0)
  }

  /** The four dense layers, in optimiser order. Exposed for tests and stats. */
  get modules(): readonly Linear[] {
    return this.layers
  }

  ensureCapacity(n: number): void {
    if (n <= this.capacity) return
    const cap = Math.max(n, Math.ceil(this.capacity * 1.5))
    const h = this.config.hiddenSize
    const dOutDim = 1 + this.geoDim
    this.enc = new Float32Array(cap * this.encDim)
    this.h1 = new Float32Array(cap * h)
    this.dout = new Float32Array(cap * dOutDim)
    this.colorIn = new Float32Array(cap * this.colorInDim)
    this.c1 = new Float32Array(cap * h)
    this.cpre = new Float32Array(cap * 3)
    this.rgb = new Float32Array(cap * 3)
    this.sigma = new Float32Array(cap)
    this.gradEnc = new Float32Array(cap * this.encDim)
    this.gradH1 = new Float32Array(cap * h)
    this.gradDOut = new Float32Array(cap * dOutDim)
    this.gradC1 = new Float32Array(cap * h)
    this.gradColorIn = new Float32Array(cap * this.colorInDim)
    this.gradCPre = new Float32Array(cap * 3)
    this.capacity = cap
  }

  /**
   * Evaluates density and colour for `n` points.
   * @param pos Normalised positions in [0,1]^3, 3 floats per point.
   * @param dir Unit view directions, 3 floats per point.
   */
  forward(pos: Float32Array, dir: Float32Array, n: number): void {
    this.ensureCapacity(n)
    const h = this.config.hiddenSize
    const dOutDim = 1 + this.geoDim
    const { encDim, geoDim, colorInDim } = this
    const degree = this.config.shDegree

    this.grid.ensureCache(n)

    for (let i = 0; i < n; i++) {
      this.grid.encode(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2], this.enc, i * encDim, i)
    }

    this.densityHidden.forward(this.enc, this.h1, n)
    relu(this.h1, n * h)
    this.densityOut.forward(this.h1, this.dout, n)

    for (let i = 0; i < n; i++) {
      let raw = this.dout[i * dOutDim]
      if (raw > RAW_DENSITY_CLAMP) raw = RAW_DENSITY_CLAMP
      else if (raw < -RAW_DENSITY_CLAMP) raw = -RAW_DENSITY_CLAMP
      this.sigma[i] = Math.exp(raw)

      const ciBase = i * colorInDim
      for (let k = 0; k < geoDim; k++) this.colorIn[ciBase + k] = this.dout[i * dOutDim + 1 + k]
      encodeDirection(dir[i * 3], dir[i * 3 + 1], dir[i * 3 + 2], degree, this.colorIn, ciBase + geoDim)
    }

    this.colorHidden.forward(this.colorIn, this.c1, n)
    relu(this.c1, n * h)
    this.colorOut.forward(this.c1, this.cpre, n)
    for (let i = 0; i < n * 3; i++) this.rgb[i] = sigmoid(this.cpre[i])
  }

  /**
   * Backpropagates dL/dsigma and dL/drgb through the network, accumulating into
   * every parameter gradient buffer. Must follow a `forward` with the same `n`.
   */
  backward(gradSigma: Float32Array, gradRgb: Float32Array, n: number): void {
    const h = this.config.hiddenSize
    const dOutDim = 1 + this.geoDim
    const { encDim, geoDim, colorInDim } = this

    // Colour branch: sigmoid -> colourOut -> ReLU -> colourHidden.
    for (let i = 0; i < n * 3; i++) {
      const s = this.rgb[i]
      this.gradCPre[i] = gradRgb[i] * s * (1 - s)
    }
    this.colorOut.backward(this.c1, this.gradCPre, this.gradC1, n)
    reluBackward(this.c1, this.gradC1, n * h)
    this.colorHidden.backward(this.colorIn, this.gradC1, this.gradColorIn, n)

    // Density branch: exp -> densityOut (+ geo-feature grads from colour).
    this.gradDOut.fill(0, 0, n * dOutDim)
    for (let i = 0; i < n; i++) {
      const raw = this.dout[i * dOutDim]
      // Gradient vanishes where exp was clamped.
      const gSigma =
        raw >= RAW_DENSITY_CLAMP || raw <= -RAW_DENSITY_CLAMP ? 0 : gradSigma[i] * this.sigma[i]
      this.gradDOut[i * dOutDim] = gSigma
      const ciBase = i * colorInDim
      for (let k = 0; k < geoDim; k++) {
        this.gradDOut[i * dOutDim + 1 + k] = this.gradColorIn[ciBase + k]
      }
    }
    this.densityOut.backward(this.h1, this.gradDOut, this.gradH1, n)
    reluBackward(this.h1, this.gradH1, n * h)
    this.densityHidden.backward(this.enc, this.gradH1, this.gradEnc, n)

    for (let i = 0; i < n; i++) {
      this.grid.accumulateCached(i, this.gradEnc, i * encDim)
    }
  }

  /** Density only — used to refresh the occupancy grid, where colour is irrelevant. */
  densityOnly(pos: Float32Array, n: number, out: Float32Array): void {
    this.ensureCapacity(n)
    const h = this.config.hiddenSize
    const dOutDim = 1 + this.geoDim
    for (let i = 0; i < n; i++) {
      this.grid.encode(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2], this.enc, i * this.encDim)
    }
    this.densityHidden.forward(this.enc, this.h1, n)
    relu(this.h1, n * h)
    this.densityOut.forward(this.h1, this.dout, n)
    for (let i = 0; i < n; i++) {
      let raw = this.dout[i * dOutDim]
      if (raw > RAW_DENSITY_CLAMP) raw = RAW_DENSITY_CLAMP
      else if (raw < -RAW_DENSITY_CLAMP) raw = -RAW_DENSITY_CLAMP
      out[i] = Math.exp(raw)
    }
  }

  zeroGrad(): void {
    this.grid.zeroGrad()
    for (const l of this.layers) l.zeroGrad()
  }

  step(lr: number): void {
    const { list, count } = this.grid.touched
    this.gridOpt.stepSparse(list, count, this.config.grid.featuresPerLevel, lr)
    for (const opt of this.layerOpts) opt.step(lr)
  }

  /** Flat snapshot of every trainable parameter, for saving a trained scene. */
  exportWeights(): Float32Array {
    const total = this.paramCount
    const out = new Float32Array(total)
    let offset = 0
    out.set(this.grid.params, offset)
    offset += this.grid.params.length
    for (const l of this.layers) {
      out.set(l.params, offset)
      offset += l.params.length
    }
    return out
  }

  /** Just the four dense layers, concatenated — the GPU keeps them in their own texture. */
  exportLayerWeights(): Float32Array {
    const total = this.layers.reduce((n, l) => n + l.paramCount, 0)
    const out = new Float32Array(total)
    let offset = 0
    for (const l of this.layers) {
      out.set(l.params, offset)
      offset += l.params.length
    }
    return out
  }

  /** Restores parameters produced by `exportWeights` on an identically configured field. */
  importWeights(data: Float32Array): void {
    if (data.length !== this.paramCount) {
      throw new Error(
        `Ağırlık boyutu uyuşmuyor: ${data.length} geldi, ${this.paramCount} bekleniyordu`,
      )
    }
    let offset = 0
    this.grid.params.set(data.subarray(offset, offset + this.grid.params.length))
    offset += this.grid.params.length
    for (const l of this.layers) {
      l.params.set(data.subarray(offset, offset + l.params.length))
      offset += l.params.length
    }
  }
}
