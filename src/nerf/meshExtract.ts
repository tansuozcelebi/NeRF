/**
 * Turning the learned density field into a triangle mesh.
 *
 * A NeRF stores geometry as a continuous density function rather than as
 * surfaces — which is exactly what makes it good at fog, hair and glass, and
 * exactly what stops you from opening the result in Blender. This module is the
 * bridge: pick a density threshold and emit the surface where the field crosses
 * it.
 *
 * It uses **marching tetrahedra** rather than the more famous marching cubes.
 * Marching cubes needs a 256x16 triangle lookup table, and a single mistyped row
 * is a hole in the mesh that no test is likely to catch. Splitting each cube
 * into six tetrahedra removes the table entirely: a tetrahedron has only four
 * corners, so the three possible cases (one, two or three corners inside) can be
 * derived in code. Every cube is split the same way around the same diagonal,
 * so neighbouring cubes agree on how their shared face is cut and the result is
 * watertight.
 */

/** Cube corners, indexed by (x, y, z) bits. */
const CORNERS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
]

/**
 * Six tetrahedra filling the cube, all sharing the 0-6 main diagonal.
 *
 * Using the same decomposition for every cube is what makes the mesh
 * watertight: two neighbouring cubes cut their shared face along the same
 * diagonal, so the triangles on either side meet exactly.
 */
const TETRAHEDRA: ReadonlyArray<readonly [number, number, number, number]> = [
  [0, 5, 1, 6],
  [0, 1, 2, 6],
  [0, 2, 3, 6],
  [0, 3, 7, 6],
  [0, 7, 4, 6],
  [0, 4, 5, 6],
]

/**
 * Forces the outermost shell of samples to be empty.
 *
 * Where the surface runs into the wall of the sampled box — the ground plane
 * reaching the edge of the scene, or anything the crop cuts through — the
 * extracted mesh would otherwise simply stop, leaving an open boundary. Marking
 * the shell as outside the surface caps those places instead, so the export is
 * a closed solid that other tools can boolean, print or fill.
 *
 * The cost is that geometry touching the wall is trimmed by one cell.
 */
export function sealBoundary(values: Float32Array, resolution: number, emptyValue = 0): void {
  const res = resolution
  const at = (x: number, y: number, z: number) => (z * res + y) * res + x
  for (let z = 0; z < res; z++) {
    for (let y = 0; y < res; y++) {
      for (let x = 0; x < res; x++) {
        const onShell =
          x === 0 || y === 0 || z === 0 || x === res - 1 || y === res - 1 || z === res - 1
        if (onShell) values[at(x, y, z)] = emptyValue
      }
    }
  }
}

export interface ExtractOptions {
  /** Number of sample points along each axis. */
  resolution: number
  /** The surface sits where density crosses this value. */
  isoLevel: number
  /** World-space extent the samples cover. */
  boundsMin: readonly [number, number, number]
  boundsMax: readonly [number, number, number]
}

export interface ExtractedMesh {
  /** Vertex positions in world space, 3 floats each. */
  positions: Float32Array
  /** Triangle corner indices, 3 per face. */
  indices: Uint32Array
  vertexCount: number
  triangleCount: number
}

/**
 * Extracts the iso-surface of a scalar field sampled on a regular grid.
 *
 * @param values `resolution^3` samples, ordered x fastest then y then z.
 */
export function extractSurface(values: Float32Array, options: ExtractOptions): ExtractedMesh {
  const { resolution: res, isoLevel, boundsMin, boundsMax } = options
  if (values.length < res * res * res) {
    throw new Error(`Örnek dizisi çok kısa: ${values.length}, ${res ** 3} bekleniyordu`)
  }

  const step: [number, number, number] = [
    (boundsMax[0] - boundsMin[0]) / (res - 1),
    (boundsMax[1] - boundsMin[1]) / (res - 1),
    (boundsMax[2] - boundsMin[2]) / (res - 1),
  ]
  const nodeIndex = (x: number, y: number, z: number) => (z * res + y) * res + x

  const positions: number[] = []
  const indices: number[] = []
  /**
   * One vertex per grid edge, shared by every tetrahedron that cuts it. Without
   * this the mesh would be a soup of unconnected triangles.
   */
  const edgeVertices = new Map<number, number>()

  /** Interpolates the crossing point on the edge between two grid nodes. */
  const vertexOnEdge = (
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
  ): number => {
    const ia = nodeIndex(ax, ay, az)
    const ib = nodeIndex(bx, by, bz)
    const key = ia < ib ? ia * (res * res * res) + ib : ib * (res * res * res) + ia
    const existing = edgeVertices.get(key)
    if (existing !== undefined) return existing

    const va = values[ia]
    const vb = values[ib]
    const denom = vb - va
    // Guard the degenerate case where both ends sit exactly on the iso level.
    const t = Math.abs(denom) < 1e-12 ? 0.5 : Math.min(1, Math.max(0, (isoLevel - va) / denom))
    const index = positions.length / 3
    positions.push(
      boundsMin[0] + (ax + t * (bx - ax)) * step[0],
      boundsMin[1] + (ay + t * (by - ay)) * step[1],
      boundsMin[2] + (az + t * (bz - az)) * step[2],
    )
    edgeVertices.set(key, index)
    return index
  }

  /** Central-difference gradient, used only to orient faces outwards. */
  const gradientAt = (x: number, y: number, z: number, out: [number, number, number]): void => {
    const clamp = (v: number) => Math.min(res - 1, Math.max(0, v))
    out[0] = values[nodeIndex(clamp(x + 1), y, z)] - values[nodeIndex(clamp(x - 1), y, z)]
    out[1] = values[nodeIndex(x, clamp(y + 1), z)] - values[nodeIndex(x, clamp(y - 1), z)]
    out[2] = values[nodeIndex(x, y, clamp(z + 1))] - values[nodeIndex(x, y, clamp(z - 1))]
  }

  const gradient: [number, number, number] = [0, 0, 0]

  /**
   * Emits a triangle with its normal pointing away from the dense region.
   *
   * Deriving the winding per case is fiddly and easy to get subtly wrong; since
   * the density gradient already points *into* the object, comparing against it
   * fixes the orientation for every case at once.
   */
  const addTriangle = (a: number, b: number, c: number, cx: number, cy: number, cz: number): void => {
    const ax = positions[a * 3], ay = positions[a * 3 + 1], az = positions[a * 3 + 2]
    const bx = positions[b * 3], by = positions[b * 3 + 1], bz = positions[b * 3 + 2]
    const px = positions[c * 3], py = positions[c * 3 + 1], pz = positions[c * 3 + 2]
    const e1 = [bx - ax, by - ay, bz - az]
    const e2 = [px - ax, py - ay, pz - az]
    const nx = e1[1] * e2[2] - e1[2] * e2[1]
    const ny = e1[2] * e2[0] - e1[0] * e2[2]
    const nz = e1[0] * e2[1] - e1[1] * e2[0]

    gradientAt(cx, cy, cz, gradient)
    const facingInwards = nx * gradient[0] + ny * gradient[1] + nz * gradient[2] > 0
    if (facingInwards) indices.push(a, c, b)
    else indices.push(a, b, c)
  }

  const inside = new Array<boolean>(4)
  const cornerX = new Array<number>(4)
  const cornerY = new Array<number>(4)
  const cornerZ = new Array<number>(4)

  for (let z = 0; z < res - 1; z++) {
    for (let y = 0; y < res - 1; y++) {
      for (let x = 0; x < res - 1; x++) {
        for (const tet of TETRAHEDRA) {
          let insideCount = 0
          for (let i = 0; i < 4; i++) {
            const offset = CORNERS[tet[i]]
            const px = x + offset[0]
            const py = y + offset[1]
            const pz = z + offset[2]
            cornerX[i] = px
            cornerY[i] = py
            cornerZ[i] = pz
            inside[i] = values[nodeIndex(px, py, pz)] > isoLevel
            if (inside[i]) insideCount++
          }
          if (insideCount === 0 || insideCount === 4) continue

          // Split the four corners into the ones inside the surface and out.
          const ins: number[] = []
          const outs: number[] = []
          for (let i = 0; i < 4; i++) (inside[i] ? ins : outs).push(i)

          const cut = (a: number, b: number) =>
            vertexOnEdge(
              cornerX[a], cornerY[a], cornerZ[a],
              cornerX[b], cornerY[b], cornerZ[b],
            )

          if (insideCount === 1 || insideCount === 3) {
            // One corner is alone on its side of the surface: the three edges
            // leaving it are cut, giving a single triangle.
            const lone = insideCount === 1 ? ins[0] : outs[0]
            const others = insideCount === 1 ? outs : ins
            addTriangle(
              cut(lone, others[0]), cut(lone, others[1]), cut(lone, others[2]),
              cornerX[lone], cornerY[lone], cornerZ[lone],
            )
          } else {
            // Two in, two out: four edges are cut and form a quad. Walking
            // in0-out0, in0-out1, in1-out1, in1-out0 traces its boundary.
            const q0 = cut(ins[0], outs[0])
            const q1 = cut(ins[0], outs[1])
            const q2 = cut(ins[1], outs[1])
            const q3 = cut(ins[1], outs[0])
            addTriangle(q0, q1, q2, cornerX[ins[0]], cornerY[ins[0]], cornerZ[ins[0]])
            addTriangle(q0, q2, q3, cornerX[ins[0]], cornerY[ins[0]], cornerZ[ins[0]])
          }
        }
      }
    }
  }

  return {
    positions: Float32Array.from(positions),
    indices: Uint32Array.from(indices),
    vertexCount: positions.length / 3,
    triangleCount: indices.length / 3,
  }
}
