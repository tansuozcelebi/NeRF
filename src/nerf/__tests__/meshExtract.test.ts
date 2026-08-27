/**
 * The mesh exporter is checked against a shape whose answer is known exactly:
 * a sphere. Every vertex must land on it, the surface must be closed, and the
 * faces must point outwards — the three ways a surface extractor goes wrong.
 */
import { describe, expect, it } from 'vitest'
import { extractSurface } from '../meshExtract'

const RES = 40
const RADIUS = 0.6
const ISO = 1

/** Density that crosses `ISO` exactly on a sphere of radius RADIUS. */
function sampleSphere(res: number): Float32Array {
  const values = new Float32Array(res * res * res)
  for (let z = 0; z < res; z++) {
    for (let y = 0; y < res; y++) {
      for (let x = 0; x < res; x++) {
        const px = (x / (res - 1)) * 2 - 1
        const py = (y / (res - 1)) * 2 - 1
        const pz = (z / (res - 1)) * 2 - 1
        const distance = Math.hypot(px, py, pz)
        // Decreasing outwards, equal to ISO at the sphere surface.
        values[(z * res + y) * res + x] = ISO + (RADIUS - distance)
      }
    }
  }
  return values
}

const OPTIONS = {
  resolution: RES,
  isoLevel: ISO,
  boundsMin: [-1, -1, -1] as [number, number, number],
  boundsMax: [1, 1, 1] as [number, number, number],
}

describe('surface extraction', () => {
  const mesh = extractSurface(sampleSphere(RES), OPTIONS)

  it('produces a mesh', () => {
    expect(mesh.triangleCount).toBeGreaterThan(500)
    expect(mesh.vertexCount).toBeGreaterThan(200)
  })

  it('places every vertex on the sphere', () => {
    let worst = 0
    for (let i = 0; i < mesh.vertexCount; i++) {
      const r = Math.hypot(
        mesh.positions[i * 3], mesh.positions[i * 3 + 1], mesh.positions[i * 3 + 2],
      )
      worst = Math.max(worst, Math.abs(r - RADIUS))
    }
    // Linear interpolation along grid edges; error is bounded by the cell size.
    expect(worst).toBeLessThan((2 / (RES - 1)) * 0.6)
  })

  it('is watertight: every edge is shared by exactly two triangles', () => {
    const edgeUse = new Map<string, number>()
    for (let t = 0; t < mesh.triangleCount; t++) {
      const corner = [mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]]
      for (let e = 0; e < 3; e++) {
        const a = corner[e]
        const b = corner[(e + 1) % 3]
        const key = a < b ? `${a}_${b}` : `${b}_${a}`
        edgeUse.set(key, (edgeUse.get(key) ?? 0) + 1)
      }
    }
    const boundaryEdges = [...edgeUse.values()].filter((n) => n !== 2)
    expect(boundaryEdges).toHaveLength(0)
  })

  it('orients faces outwards', () => {
    let outward = 0
    for (let t = 0; t < mesh.triangleCount; t++) {
      const [ia, ib, ic] = [mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]]
      const a = mesh.positions.subarray(ia * 3, ia * 3 + 3)
      const b = mesh.positions.subarray(ib * 3, ib * 3 + 3)
      const c = mesh.positions.subarray(ic * 3, ic * 3 + 3)
      const e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
      const e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]]
      const n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
      ]
      // On a sphere centred at the origin the outward normal is the position.
      const centroid = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3, (a[2] + b[2] + c[2]) / 3]
      if (n[0] * centroid[0] + n[1] * centroid[1] + n[2] * centroid[2] > 0) outward++
    }
    expect(outward).toBe(mesh.triangleCount)
  })

  it('encloses the right volume', () => {
    // Divergence theorem: summing the signed tetrahedron volumes to the origin
    // gives the enclosed volume, which is another check that the surface closes.
    let volume = 0
    for (let t = 0; t < mesh.triangleCount; t++) {
      const [ia, ib, ic] = [mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]]
      const a = mesh.positions.subarray(ia * 3, ia * 3 + 3)
      const b = mesh.positions.subarray(ib * 3, ib * 3 + 3)
      const c = mesh.positions.subarray(ic * 3, ic * 3 + 3)
      volume +=
        (a[0] * (b[1] * c[2] - b[2] * c[1]) -
          a[1] * (b[0] * c[2] - b[2] * c[0]) +
          a[2] * (b[0] * c[1] - b[1] * c[0])) / 6
    }
    const expected = (4 / 3) * Math.PI * RADIUS ** 3
    expect(volume).toBeGreaterThan(expected * 0.95)
    expect(volume).toBeLessThan(expected * 1.05)
  })

  it('returns nothing when the field never crosses the threshold', () => {
    const flat = new Float32Array(RES * RES * RES).fill(ISO - 1)
    expect(extractSurface(flat, OPTIONS).triangleCount).toBe(0)
  })

  it('rejects a sample array that is too small', () => {
    expect(() => extractSurface(new Float32Array(8), OPTIONS)).toThrow()
  })
})

describe('boundary sealing', () => {
  it('closes a surface that runs off the edge of the sampled box', async () => {
    const { sealBoundary } = await import('../meshExtract')
    // Dense everywhere: without sealing this produces no surface at all, and a
    // half-open one for anything that merely touches the wall.
    const values = new Float32Array(RES * RES * RES).fill(ISO + 1)
    sealBoundary(values, RES)
    const mesh = extractSurface(values, OPTIONS)

    expect(mesh.triangleCount).toBeGreaterThan(0)
    const edgeUse = new Map<string, number>()
    for (let t = 0; t < mesh.triangleCount; t++) {
      const corner = [mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]]
      for (let e = 0; e < 3; e++) {
        const a = corner[e]
        const b = corner[(e + 1) % 3]
        const key = a < b ? `${a}_${b}` : `${b}_${a}`
        edgeUse.set(key, (edgeUse.get(key) ?? 0) + 1)
      }
    }
    expect([...edgeUse.values()].filter((n) => n !== 2)).toHaveLength(0)
  })
})
