/**
 * Multiresolution hash encoding (Müller et al., "Instant Neural Graphics
 * Primitives", 2022).
 *
 * A 3D position is looked up in L grids of growing resolution. Each grid stores
 * F trainable features per vertex in a hash table of T entries; the features of
 * the 8 surrounding vertices are trilinearly interpolated and the L results are
 * concatenated. This is what lets a *tiny* MLP represent a detailed scene, and
 * it is the reason training converges in seconds rather than hours.
 */
import { makeRng } from './random'
import type { HashGridConfig } from './types'

export const PRIME_Y = 2654435761
export const PRIME_Z = 805459861

/** Everything needed to reproduce this grid's addressing elsewhere. */
export interface GridLayout {
  levels: number
  featuresPerLevel: number
  tableSize: number
  tableMask: number
  /** Grid resolution per level. */
  resolutions: number[]
  /** Start of each level in the table, in entries. */
  levelOffsets: number[]
  /** Side length when a level is addressed densely, 0 when it is hashed. */
  denseSides: number[]
  entryCount: number
}

export class HashGrid {
  readonly config: HashGridConfig
  /** levels * tableSize * featuresPerLevel trainable features. */
  readonly params: Float32Array
  /** Gradient buffer, same layout as `params`. */
  readonly grads: Float32Array
  readonly outputDim: number

  private readonly resolutions: Int32Array
  /** Per level: offset in *entries* (not floats) into the parameter array. */
  private readonly levelOffset: Int32Array
  /** Per level: 0 = hashed, otherwise the side length of the dense grid. */
  private readonly denseSide: Int32Array
  private readonly tableMask: number
  private readonly tableSize: number

  /** Sparse-gradient bookkeeping: which entries were touched this step. */
  private readonly stamp: Int32Array
  private touchedList: Int32Array
  private touchedCount = 0
  private currentStamp = 0

  /**
   * Corner indices and interpolation fractions saved during the forward pass.
   * The backward pass would otherwise redo every floor and hash — about a fifth
   * of the total training cost.
   */
  private cacheCapacity = 0
  private idxCache = new Int32Array(0)
  private fracCache = new Float32Array(0)

  constructor(config: HashGridConfig, seed: number) {
    this.config = config
    const { levels, featuresPerLevel, log2TableSize, baseResolution, maxResolution } = config
    this.tableSize = 1 << log2TableSize
    this.tableMask = this.tableSize - 1
    this.outputDim = levels * featuresPerLevel

    const totalEntries = levels * this.tableSize
    this.params = new Float32Array(totalEntries * featuresPerLevel)
    this.grads = new Float32Array(totalEntries * featuresPerLevel)
    this.stamp = new Int32Array(totalEntries).fill(-1)
    this.touchedList = new Int32Array(1 << 16)

    this.resolutions = new Int32Array(levels)
    this.levelOffset = new Int32Array(levels)
    this.denseSide = new Int32Array(levels)

    // Geometric progression from baseResolution to maxResolution.
    const growth =
      levels > 1
        ? Math.exp((Math.log(maxResolution) - Math.log(baseResolution)) / (levels - 1))
        : 1
    for (let l = 0; l < levels; l++) {
      const res = Math.max(1, Math.floor(baseResolution * Math.pow(growth, l)))
      this.resolutions[l] = res
      this.levelOffset[l] = l * this.tableSize
      // Small levels are addressed directly — collision-free and faster.
      const side = res + 1
      this.denseSide[l] = side * side * side <= this.tableSize ? side : 0
    }

    // Instant-NGP initialises features uniformly in a narrow band around zero.
    const rng = makeRng(seed)
    for (let i = 0; i < this.params.length; i++) {
      this.params[i] = (rng() * 2 - 1) * 1e-4
    }
  }

  get levels(): number {
    return this.config.levels
  }

  resolutionAt(level: number): number {
    return this.resolutions[level]
  }

  /**
   * The per-level addressing scheme, so another implementation (the GPU shader)
   * can reproduce these lookups exactly. Any drift here shows up as a GPU render
   * that disagrees with the CPU one.
   */
  describeLayout(): GridLayout {
    return {
      levels: this.config.levels,
      featuresPerLevel: this.config.featuresPerLevel,
      tableSize: this.tableSize,
      tableMask: this.tableMask,
      resolutions: Array.from(this.resolutions),
      levelOffsets: Array.from(this.levelOffset),
      denseSides: Array.from(this.denseSide),
      entryCount: this.config.levels * this.tableSize,
    }
  }

  /** Index of the vertex (x, y, z) at `level`, in entries. */
  private vertexIndex(level: number, x: number, y: number, z: number): number {
    const side = this.denseSide[level]
    if (side > 0) {
      return this.levelOffset[level] + x + side * (y + side * z)
    }
    const h = (x ^ Math.imul(y, PRIME_Y) ^ Math.imul(z, PRIME_Z)) >>> 0
    return this.levelOffset[level] + (h & this.tableMask)
  }

  ensureCache(points: number): void {
    if (points <= this.cacheCapacity) return
    const cap = Math.max(points, Math.ceil(this.cacheCapacity * 1.5))
    this.idxCache = new Int32Array(cap * this.config.levels * 8)
    this.fracCache = new Float32Array(cap * this.config.levels * 3)
    this.cacheCapacity = cap
  }

  /**
   * Encodes a position given in normalised [0,1]^3 coordinates.
   * Writes `outputDim` floats starting at `out[outOffset]`.
   *
   * @param cacheIndex Point slot to remember the corner lookups in, so
   *                   `accumulateCached` can skip recomputing them. Pass -1 for
   *                   inference-only calls.
   */
  encode(
    px: number, py: number, pz: number,
    out: Float32Array, outOffset: number,
    cacheIndex = -1,
  ): void {
    const F = this.config.featuresPerLevel
    const levels = this.config.levels
    const params = this.params
    const idxCache = this.idxCache
    const fracCache = this.fracCache
    const idxBase = cacheIndex * levels * 8
    const fracBase = cacheIndex * levels * 3

    for (let l = 0; l < levels; l++) {
      const res = this.resolutions[l]
      const gx = px * res
      const gy = py * res
      const gz = pz * res
      // Positions are non-negative, so a truncating cast matches Math.floor.
      let x0 = gx | 0
      let y0 = gy | 0
      let z0 = gz | 0
      if (x0 < 0) x0 = 0; else if (x0 > res - 1) x0 = res - 1
      if (y0 < 0) y0 = 0; else if (y0 > res - 1) y0 = res - 1
      if (z0 < 0) z0 = 0; else if (z0 > res - 1) z0 = res - 1
      const fx = gx - x0
      const fy = gy - y0
      const fz = gz - z0

      if (cacheIndex >= 0) {
        fracCache[fracBase + l * 3] = fx
        fracCache[fracBase + l * 3 + 1] = fy
        fracCache[fracBase + l * 3 + 2] = fz
      }

      const base = outOffset + l * F
      for (let f = 0; f < F; f++) out[base + f] = 0

      for (let c = 0; c < 8; c++) {
        const entry = this.vertexIndex(l, x0 + (c & 1), y0 + ((c >> 1) & 1), z0 + ((c >> 2) & 1))
        if (cacheIndex >= 0) idxCache[idxBase + l * 8 + c] = entry
        const w =
          ((c & 1) ? fx : 1 - fx) *
          (((c >> 1) & 1) ? fy : 1 - fy) *
          (((c >> 2) & 1) ? fz : 1 - fz)
        if (w === 0) continue
        const idx = entry * F
        for (let f = 0; f < F; f++) out[base + f] += w * params[idx + f]
      }
    }
  }

  /**
   * Scatters `gradOut` (outputDim values at `gradOffset`) back into the feature
   * table, reusing the corners saved by `encode` for the same `cacheIndex`.
   */
  accumulateCached(cacheIndex: number, gradOut: Float32Array, gradOffset: number): void {
    const F = this.config.featuresPerLevel
    const levels = this.config.levels
    const grads = this.grads
    const idxCache = this.idxCache
    const fracCache = this.fracCache
    const idxBase = cacheIndex * levels * 8
    const fracBase = cacheIndex * levels * 3

    for (let l = 0; l < levels; l++) {
      const fx = fracCache[fracBase + l * 3]
      const fy = fracCache[fracBase + l * 3 + 1]
      const fz = fracCache[fracBase + l * 3 + 2]
      const base = gradOffset + l * F

      let allZero = true
      for (let f = 0; f < F; f++) {
        if (gradOut[base + f] !== 0) { allZero = false; break }
      }
      if (allZero) continue

      for (let c = 0; c < 8; c++) {
        const w =
          ((c & 1) ? fx : 1 - fx) *
          (((c >> 1) & 1) ? fy : 1 - fy) *
          (((c >> 2) & 1) ? fz : 1 - fz)
        if (w === 0) continue
        const entry = idxCache[idxBase + l * 8 + c]
        this.markTouched(entry)
        const idx = entry * F
        for (let f = 0; f < F; f++) grads[idx + f] += w * gradOut[base + f]
      }
    }
  }

  /**
   * Standalone gradient scatter for a position, recomputing the corners.
   * Used by tests and by any caller that did not populate the cache.
   */
  accumulate(
    px: number, py: number, pz: number,
    gradOut: Float32Array, gradOffset: number,
  ): void {
    this.ensureCache(1)
    const scratch = new Float32Array(this.outputDim)
    this.encode(px, py, pz, scratch, 0, 0)
    this.accumulateCached(0, gradOut, gradOffset)
  }

  private markTouched(entry: number): void {
    if (this.stamp[entry] === this.currentStamp) return
    this.stamp[entry] = this.currentStamp
    if (this.touchedCount === this.touchedList.length) {
      const bigger = new Int32Array(this.touchedList.length * 2)
      bigger.set(this.touchedList)
      this.touchedList = bigger
    }
    this.touchedList[this.touchedCount++] = entry
  }

  /** Clears gradients for the entries touched since the last call. */
  zeroGrad(): void {
    const F = this.config.featuresPerLevel
    for (let e = 0; e < this.touchedCount; e++) {
      const idx = this.touchedList[e] * F
      for (let f = 0; f < F; f++) this.grads[idx + f] = 0
    }
    this.touchedCount = 0
    this.currentStamp++
  }

  get touched(): { list: Int32Array; count: number } {
    return { list: this.touchedList, count: this.touchedCount }
  }
}
