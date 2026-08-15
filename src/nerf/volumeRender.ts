/**
 * Differentiable volume rendering: turn a ray into a pixel.
 *
 * Along each ray we take stratified samples, ask the field for density and
 * colour, and alpha-composite front to back:
 *
 *   alpha_i = 1 - exp(-sigma_i * delta_i)
 *   w_i     = T_i * alpha_i,   T_i = prod_{j<i} (1 - alpha_j)
 *   C       = sum_i w_i * c_i + T_end * background
 *
 * The backward pass below is the analytic derivative of that expression, which
 * is why training needs no autograd framework.
 */
import { intersectAabb } from './camera'
import { OccupancyGrid } from './occupancy'

export interface SampleStats {
  /** Compact points actually handed to the network. */
  points: number
  /** Points that would have been generated without empty-space skipping. */
  pointsBeforePruning: number
}

export class VolumeRenderer {
  readonly maxRays: number
  readonly samplesPerRay: number
  aabbSize: number
  background: Float32Array

  /** Compact sample buffers — only points inside occupied space are stored. */
  readonly pos: Float32Array
  readonly dir: Float32Array
  readonly delta: Float32Array
  readonly tval: Float32Array
  private readonly weight: Float32Array
  private readonly trans: Float32Array

  /** rayStart[i] .. rayStart[i+1] delimits ray i's samples. */
  readonly rayStart: Int32Array
  private readonly rayTend: Float32Array

  readonly color: Float32Array
  readonly depth: Float32Array
  readonly acc: Float32Array

  pointCount = 0

  constructor(maxRays: number, samplesPerRay: number, aabbSize: number, background: [number, number, number]) {
    this.maxRays = maxRays
    this.samplesPerRay = samplesPerRay
    this.aabbSize = aabbSize
    this.background = new Float32Array(background)

    const maxPoints = maxRays * samplesPerRay
    this.pos = new Float32Array(maxPoints * 3)
    this.dir = new Float32Array(maxPoints * 3)
    this.delta = new Float32Array(maxPoints)
    this.tval = new Float32Array(maxPoints)
    this.weight = new Float32Array(maxPoints)
    this.trans = new Float32Array(maxPoints)
    this.rayStart = new Int32Array(maxRays + 1)
    this.rayTend = new Float32Array(maxRays)
    this.color = new Float32Array(maxRays * 3)
    this.depth = new Float32Array(maxRays)
    this.acc = new Float32Array(maxRays)
  }

  /**
   * Places stratified samples along each ray, skipping empty cells.
   * @param jitter Returns a value in [0,1); pass `() => 0.5` for deterministic
   *               mid-stratum sampling when rendering a still image.
   */
  sample(
    origins: Float32Array,
    dirs: Float32Array,
    rayCount: number,
    occupancy: OccupancyGrid | null,
    jitter: () => number,
  ): SampleStats {
    const S = this.samplesPerRay
    const size = this.aabbSize
    const invExtent = 1 / (2 * size)
    let p = 0
    let before = 0

    for (let r = 0; r < rayCount; r++) {
      this.rayStart[r] = p
      const ox = origins[r * 3], oy = origins[r * 3 + 1], oz = origins[r * 3 + 2]
      const dx = dirs[r * 3], dy = dirs[r * 3 + 1], dz = dirs[r * 3 + 2]
      const hit = intersectAabb(ox, oy, oz, dx, dy, dz, size)
      if (!hit) continue

      const [tn, tf] = hit
      const step = (tf - tn) / S
      if (step <= 0) continue
      before += S

      for (let k = 0; k < S; k++) {
        const t = tn + (k + jitter()) * step
        const wx = ox + t * dx
        const wy = oy + t * dy
        const wz = oz + t * dz
        // World [-size, size] -> normalised [0, 1].
        const nx = (wx + size) * invExtent
        const ny = (wy + size) * invExtent
        const nz = (wz + size) * invExtent
        if (occupancy && !occupancy.isOccupied(nx, ny, nz)) continue

        this.pos[p * 3] = nx
        this.pos[p * 3 + 1] = ny
        this.pos[p * 3 + 2] = nz
        this.dir[p * 3] = dx
        this.dir[p * 3 + 1] = dy
        this.dir[p * 3 + 2] = dz
        this.delta[p] = step
        this.tval[p] = t
        p++
      }
    }
    this.rayStart[rayCount] = p
    this.pointCount = p
    return { points: p, pointsBeforePruning: before }
  }

  /**
   * Front-to-back compositing of the field outputs into per-ray colours.
   * @param earlyStop Stop a ray once it is effectively opaque. Great for
   *                  previews, off during training so gradients stay exact.
   */
  composite(sigma: Float32Array, rgb: Float32Array, rayCount: number, earlyStop = false): void {
    const bg = this.background
    for (let r = 0; r < rayCount; r++) {
      const start = this.rayStart[r]
      const end = this.rayStart[r + 1]
      let T = 1
      let cr = 0, cg = 0, cb = 0
      let d = 0
      for (let j = start; j < end; j++) {
        if (earlyStop && T < 1e-4) {
          this.weight[j] = 0
          this.trans[j] = 0
          continue
        }
        const alpha = 1 - Math.exp(-sigma[j] * this.delta[j])
        const w = T * alpha
        this.weight[j] = w
        this.trans[j] = T
        cr += w * rgb[j * 3]
        cg += w * rgb[j * 3 + 1]
        cb += w * rgb[j * 3 + 2]
        d += w * this.tval[j]
        T -= w
      }
      this.rayTend[r] = T
      this.acc[r] = 1 - T
      this.depth[r] = d
      this.color[r * 3] = cr + T * bg[0]
      this.color[r * 3 + 1] = cg + T * bg[1]
      this.color[r * 3 + 2] = cb + T * bg[2]
    }
  }

  /**
   * Turns dL/dC (3 per ray) into dL/dsigma and dL/drgb for every sample.
   *
   * Walking each ray backwards, `Sc` accumulates the colour contributed by the
   * samples *behind* the current one; raising sigma here dims all of them,
   * which is exactly the second term of the derivative.
   */
  backward(
    gradColor: Float32Array,
    rayCount: number,
    rgb: Float32Array,
    gradSigma: Float32Array,
    gradRgb: Float32Array,
  ): void {
    const bg = this.background
    for (let r = 0; r < rayCount; r++) {
      const start = this.rayStart[r]
      const end = this.rayStart[r + 1]
      const gr = gradColor[r * 3]
      const gg = gradColor[r * 3 + 1]
      const gb = gradColor[r * 3 + 2]
      const gDotBg = gr * bg[0] + gg * bg[1] + gb * bg[2]
      const tEnd = this.rayTend[r]

      let scR = 0, scG = 0, scB = 0
      for (let j = end - 1; j >= start; j--) {
        const w = this.weight[j]
        const cr = rgb[j * 3], cg = rgb[j * 3 + 1], cb = rgb[j * 3 + 2]
        const gDotC = gr * cr + gg * cg + gb * cb
        const gDotSc = gr * scR + gg * scG + gb * scB
        const tNext = this.trans[j] - w

        gradSigma[j] = this.delta[j] * (tNext * gDotC - gDotSc - gDotBg * tEnd)
        gradRgb[j * 3] = w * gr
        gradRgb[j * 3 + 1] = w * gg
        gradRgb[j * 3 + 2] = w * gb

        scR += w * cr
        scG += w * cg
        scB += w * cb
      }
    }
  }
}
