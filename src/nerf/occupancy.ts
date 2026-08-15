/**
 * Coarse occupancy grid used to skip empty space.
 *
 * Most of the volume around an object is air. Evaluating the network there is
 * pure waste, so every few steps we probe the density on a coarse grid and
 * remember which cells are worth sampling. This is the single biggest speed-up
 * in the whole pipeline — typically 3-6x once the scene starts to take shape.
 */
import type { RadianceField } from './field'

export class OccupancyGrid {
  readonly resolution: number
  /**
   * Per-refresh decay applied to a cell's remembered density. A cell only dies
   * once it has read low several refreshes in a row, which protects real
   * geometry from a single unlucky probe — but too slow a decay leaves the grid
   * remembering the dense fog the model started with, and nothing gets pruned.
   */
  readonly decay: number
  /** Running (decayed) maximum density seen per cell. */
  readonly density: Float32Array
  readonly occupied: Uint8Array
  private readonly probePos: Float32Array
  private readonly probeOut: Float32Array
  private lastThreshold = 0

  constructor(resolution = 32, decay = 0.8) {
    this.resolution = resolution
    this.decay = decay
    const cells = resolution * resolution * resolution
    this.density = new Float32Array(cells)
    this.occupied = new Uint8Array(cells).fill(1)
    this.probePos = new Float32Array(resolution * resolution * 3)
    this.probeOut = new Float32Array(resolution * resolution)
  }

  /** Fraction of cells currently marked occupied. */
  get occupiedFraction(): number {
    let n = 0
    for (let i = 0; i < this.occupied.length; i++) n += this.occupied[i]
    return n / this.occupied.length
  }

  get threshold(): number {
    return this.lastThreshold
  }

  /**
   * Re-probes the field and updates the occupancy mask.
   * @param sampleDelta Typical distance between two ray samples; a cell counts
   *                    as occupied when a sample in it would be at least
   *                    marginally opaque.
   * @param jitter      Random offset in [0,1) per refresh so cell centres do not
   *                    always land on the same points.
   */
  refresh(field: RadianceField, sampleDelta: number, jitter: () => number): void {
    const res = this.resolution
    const inv = 1 / res
    // alpha = 1 - exp(-sigma * delta); 0.01 alpha is the cutoff for "matters".
    const threshold = 0.01 / Math.max(sampleDelta, 1e-6)
    this.lastThreshold = threshold

    let occupiedCount = 0
    // Probe one XY slab at a time to keep the batch small.
    for (let z = 0; z < res; z++) {
      let p = 0
      for (let y = 0; y < res; y++) {
        for (let x = 0; x < res; x++) {
          this.probePos[p++] = (x + jitter()) * inv
          this.probePos[p++] = (y + jitter()) * inv
          this.probePos[p++] = (z + jitter()) * inv
        }
      }
      const count = res * res
      field.densityOnly(this.probePos, count, this.probeOut)
      const base = z * res * res
      for (let i = 0; i < count; i++) {
        // Decayed max: a cell stays alive for a while after a low reading, so a
        // single unlucky probe cannot delete real geometry.
        const decayed = this.density[base + i] * this.decay
        const value = Math.max(decayed, this.probeOut[i])
        this.density[base + i] = value
        const isOccupied = value > threshold ? 1 : 0
        this.occupied[base + i] = isOccupied
        occupiedCount += isOccupied
      }
    }

    // Safety net. An empty mask is an absorbing state: no cells occupied means
    // no samples, which means no gradients, which means the density can never
    // rise above the threshold again. If we ever land there, open the volume
    // back up and let the model rebuild.
    if (occupiedCount === 0) {
      this.reset()
    }
  }

  /** Marks everything occupied again (used when training restarts). */
  reset(): void {
    this.density.fill(0)
    this.occupied.fill(1)
  }

  /** Lookup for a position in normalised [0,1]^3 coordinates. */
  isOccupied(px: number, py: number, pz: number): boolean {
    const res = this.resolution
    let x = (px * res) | 0
    let y = (py * res) | 0
    let z = (pz * res) | 0
    if (x < 0) x = 0; else if (x >= res) x = res - 1
    if (y < 0) y = 0; else if (y >= res) y = res - 1
    if (z < 0) z = 0; else if (z >= res) z = res - 1
    return this.occupied[(z * res + y) * res + x] === 1
  }
}
