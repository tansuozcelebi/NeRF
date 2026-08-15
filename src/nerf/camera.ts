/**
 * Camera model, pose helpers and ray geometry.
 *
 * Convention (the one the original NeRF / Blender datasets use):
 * a pose is a row-major camera-to-world 4x4 matrix, the camera looks down its
 * own -Z axis, +X points right and +Y points up in image space.
 */
import type { Intrinsics, Mat4 } from './types'

export type Vec3 = [number, number, number]

export function identity(): Mat4 {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ])
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / len, v[1] / len, v[2] / len]
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}

/**
 * Builds a camera-to-world matrix for a camera at `eye` looking at `target`.
 * `up` only needs to be roughly correct; it is re-orthogonalised.
 */
export function lookAt(eye: Vec3, target: Vec3, up: Vec3 = [0, 1, 0]): Mat4 {
  // Camera looks down -Z, so the third basis column points *away* from target.
  const back = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]])
  let right = cross(up, back)
  if (Math.hypot(right[0], right[1], right[2]) < 1e-6) {
    // Degenerate: looking straight along `up`. Pick any perpendicular axis.
    right = cross([0, 0, 1], back)
  }
  right = normalize(right)
  const trueUp = cross(back, right)

  const m = identity()
  m[0] = right[0]; m[1] = trueUp[0]; m[2] = back[0]; m[3] = eye[0]
  m[4] = right[1]; m[5] = trueUp[1]; m[6] = back[1]; m[7] = eye[1]
  m[8] = right[2]; m[9] = trueUp[2]; m[10] = back[2]; m[11] = eye[2]
  return m
}

/**
 * Camera placed on a sphere around `target`.
 * @param azimuth   Angle around the +Y axis, radians.
 * @param elevation Angle above the XZ plane, radians.
 */
export function orbitPose(
  azimuth: number,
  elevation: number,
  radius: number,
  target: Vec3 = [0, 0, 0],
): Mat4 {
  const cosE = Math.cos(elevation)
  const eye: Vec3 = [
    target[0] + radius * cosE * Math.sin(azimuth),
    target[1] + radius * Math.sin(elevation),
    target[2] + radius * cosE * Math.cos(azimuth),
  ]
  return lookAt(eye, target)
}

/** Focal length in pixels for a horizontal field of view given in radians. */
export function focalFromFov(fovRadians: number, width: number): number {
  return width / (2 * Math.tan(fovRadians / 2))
}

/** Horizontal field of view in radians for a focal length in pixels. */
export function fovFromFocal(focal: number, width: number): number {
  return 2 * Math.atan(width / (2 * focal))
}

export function defaultIntrinsics(width: number, height: number, fovDegrees = 50): Intrinsics {
  return {
    focal: focalFromFov((fovDegrees * Math.PI) / 180, width),
    cx: width / 2,
    cy: height / 2,
  }
}

/** Camera position encoded in a camera-to-world matrix. */
export function poseOrigin(pose: Mat4): Vec3 {
  return [pose[3], pose[7], pose[11]]
}

/**
 * World-space ray through pixel centre (px, py).
 * Writes the origin into `outO` and the *normalised* direction into `outD`.
 */
export function rayForPixel(
  pose: Mat4,
  intr: Intrinsics,
  px: number,
  py: number,
  outO: Float32Array,
  outD: Float32Array,
): void {
  // Pixel -> camera space. Y is flipped because image rows go downwards.
  const cx = (px + 0.5 - intr.cx) / intr.focal
  const cy = -(py + 0.5 - intr.cy) / intr.focal
  const cz = -1

  let dx = pose[0] * cx + pose[1] * cy + pose[2] * cz
  let dy = pose[4] * cx + pose[5] * cy + pose[6] * cz
  let dz = pose[8] * cx + pose[9] * cy + pose[10] * cz
  const len = Math.hypot(dx, dy, dz) || 1
  dx /= len; dy /= len; dz /= len

  outO[0] = pose[3]; outO[1] = pose[7]; outO[2] = pose[11]
  outD[0] = dx; outD[1] = dy; outD[2] = dz
}

/**
 * Slab intersection of a ray with the axis-aligned box [-size, size]^3.
 * Returns `[tNear, tFar]`, or `null` when the ray misses the box.
 * `tNear` is clamped to 0 so cameras inside the box still work.
 */
export function intersectAabb(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  size: number,
): [number, number] | null {
  let tMin = -Infinity
  let tMax = Infinity

  const o = [ox, oy, oz]
  const d = [dx, dy, dz]
  for (let axis = 0; axis < 3; axis++) {
    const dir = d[axis]
    const org = o[axis]
    if (Math.abs(dir) < 1e-8) {
      if (org < -size || org > size) return null
      continue
    }
    const inv = 1 / dir
    let t0 = (-size - org) * inv
    let t1 = (size - org) * inv
    if (t0 > t1) { const tmp = t0; t0 = t1; t1 = tmp }
    if (t0 > tMin) tMin = t0
    if (t1 < tMax) tMax = t1
    if (tMin > tMax) return null
  }
  if (tMax <= 0) return null
  return [Math.max(tMin, 0), tMax]
}

/**
 * Recentres and rescales a set of poses so the cameras look at the origin and
 * the scene comfortably fits inside the unit box. Returns new matrices.
 */
export function normalizePoses(poses: Mat4[], targetRadius = 2.2): Mat4[] {
  if (poses.length === 0) return []
  let cx = 0, cy = 0, cz = 0
  for (const p of poses) {
    cx += p[3]; cy += p[7]; cz += p[11]
  }
  cx /= poses.length; cy /= poses.length; cz /= poses.length

  let maxDist = 0
  for (const p of poses) {
    maxDist = Math.max(maxDist, Math.hypot(p[3] - cx, p[7] - cy, p[11] - cz))
  }
  const scale = maxDist > 1e-6 ? targetRadius / maxDist : 1

  return poses.map((p) => {
    const q = new Float32Array(p)
    q[3] = (p[3] - cx) * scale
    q[7] = (p[7] - cy) * scale
    q[11] = (p[11] - cz) * scale
    return q
  })
}

/**
 * Camera positions spread over a spherical cap using a Fibonacci lattice —
 * much more even coverage than a naive ring, which matters a lot for NeRF.
 */
export function fibonacciOrbit(
  count: number,
  radius: number,
  minElevationDeg = 5,
  maxElevationDeg = 62,
): Mat4[] {
  const golden = Math.PI * (3 - Math.sqrt(5))
  const minE = (minElevationDeg * Math.PI) / 180
  const maxE = (maxElevationDeg * Math.PI) / 180
  const poses: Mat4[] = []
  for (let i = 0; i < count; i++) {
    const frac = count > 1 ? i / (count - 1) : 0.5
    poses.push(orbitPose(i * golden, minE + (maxE - minE) * frac, radius))
  }
  return poses
}

/** Camera positions on a single horizontal ring, in capture order. */
export function ringOrbit(count: number, radius: number, elevationDeg = 20): Mat4[] {
  const elevation = (elevationDeg * Math.PI) / 180
  const poses: Mat4[] = []
  for (let i = 0; i < count; i++) {
    poses.push(orbitPose((2 * Math.PI * i) / count, elevation, radius))
  }
  return poses
}
