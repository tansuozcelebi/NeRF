/**
 * A small procedural scene rendered with a classic ray tracer.
 *
 * Why this exists: a real NeRF needs to know exactly where each photo was taken
 * from, and recovering that from arbitrary phone snaps requires structure-from-
 * motion (COLMAP and friends), which does not run in a browser tab. The
 * synthetic scene sidesteps that — we *choose* the camera poses, so the poses
 * are exact and the reconstruction is limited only by the model. It is the
 * honest way to demonstrate that the NeRF implementation actually works, and it
 * gives new users something to press "train" on immediately.
 *
 * The shading deliberately includes sharp specular highlights: they move as the
 * camera moves, so the view-dependent half of the network has something real to
 * learn.
 */
import { defaultIntrinsics, fibonacciOrbit, rayForPixel } from './camera'
import type { Dataset, Intrinsics, Mat4, TrainingView } from './types'

interface Sphere {
  center: [number, number, number]
  radius: number
  albedo: [number, number, number]
  shininess: number
  specular: number
}

const SPHERES: Sphere[] = [
  { center: [0, 0.05, 0], radius: 0.72, albedo: [0.86, 0.26, 0.24], shininess: 64, specular: 0.55 },
  { center: [-0.95, -0.35, 0.62], radius: 0.36, albedo: [0.22, 0.52, 0.92], shininess: 120, specular: 0.8 },
  { center: [0.82, -0.45, -0.5], radius: 0.28, albedo: [0.95, 0.79, 0.22], shininess: 32, specular: 0.35 },
]

const GROUND_Y = -0.75
const GROUND_RADIUS = 1.7
const KEY_LIGHT: [number, number, number] = [0.55, 0.78, 0.3]
const FILL_LIGHT: [number, number, number] = [-0.6, 0.35, -0.72]

export const SYNTHETIC_BACKGROUND: [number, number, number] = [0.07, 0.08, 0.12]
export const SYNTHETIC_AABB = 1.9

function normalize3(v: [number, number, number]): [number, number, number] {
  const l = Math.hypot(v[0], v[1], v[2]) || 1
  return [v[0] / l, v[1] / l, v[2] / l]
}

const KEY = normalize3(KEY_LIGHT)
const FILL = normalize3(FILL_LIGHT)

/** Nearest sphere hit along the ray, or -1. Writes the distance into `out`. */
function hitSpheres(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  tMax: number,
  out: { t: number },
): number {
  let best = -1
  let bestT = tMax
  for (let s = 0; s < SPHERES.length; s++) {
    const c = SPHERES[s].center
    const r = SPHERES[s].radius
    const ex = ox - c[0], ey = oy - c[1], ez = oz - c[2]
    const b = ex * dx + ey * dy + ez * dz
    const cc = ex * ex + ey * ey + ez * ez - r * r
    const disc = b * b - cc
    if (disc <= 0) continue
    const sq = Math.sqrt(disc)
    let t = -b - sq
    if (t < 1e-4) t = -b + sq
    if (t > 1e-4 && t < bestT) {
      bestT = t
      best = s
    }
  }
  out.t = bestT
  return best
}

function inShadow(
  px: number, py: number, pz: number,
  lx: number, ly: number, lz: number,
): boolean {
  const scratch = { t: 0 }
  return hitSpheres(px + lx * 1e-3, py + ly * 1e-3, pz + lz * 1e-3, lx, ly, lz, 50, scratch) >= 0
}

function shade(
  nx: number, ny: number, nz: number,
  vx: number, vy: number, vz: number,
  albedo: [number, number, number],
  specular: number,
  shininess: number,
  px: number, py: number, pz: number,
  out: [number, number, number],
): void {
  // Ambient + two directional lights, Blinn-Phong specular.
  let r = albedo[0] * 0.18
  let g = albedo[1] * 0.18
  let b = albedo[2] * 0.2

  const lights: Array<{ dir: [number, number, number]; color: [number, number, number] }> = [
    { dir: KEY, color: [1.0, 0.97, 0.9] },
    { dir: FILL, color: [0.28, 0.36, 0.55] },
  ]

  for (const light of lights) {
    const [lx, ly, lz] = light.dir
    const ndl = nx * lx + ny * ly + nz * lz
    if (ndl <= 0) continue
    const shadowed = inShadow(px, py, pz, lx, ly, lz)
    const visibility = shadowed ? 0.15 : 1
    r += albedo[0] * light.color[0] * ndl * visibility
    g += albedo[1] * light.color[1] * ndl * visibility
    b += albedo[2] * light.color[2] * ndl * visibility

    // Half vector: this term is what makes the scene view-dependent.
    let hx = lx - vx, hy = ly - vy, hz = lz - vz
    const hl = Math.hypot(hx, hy, hz) || 1
    hx /= hl; hy /= hl; hz /= hl
    const ndh = Math.max(0, nx * hx + ny * hy + nz * hz)
    const spec = specular * Math.pow(ndh, shininess) * visibility
    r += light.color[0] * spec
    g += light.color[1] * spec
    b += light.color[2] * spec
  }

  out[0] = Math.min(1, r)
  out[1] = Math.min(1, g)
  out[2] = Math.min(1, b)
}

/** Traces one ray through the synthetic scene. */
function traceRay(
  ox: number, oy: number, oz: number,
  dx: number, dy: number, dz: number,
  out: [number, number, number],
): void {
  const scratch = { t: 0 }
  const sphere = hitSpheres(ox, oy, oz, dx, dy, dz, 1e9, scratch)
  const tSphere = sphere >= 0 ? scratch.t : Infinity

  // Ground disc.
  let tGround = Infinity
  if (Math.abs(dy) > 1e-6) {
    const t = (GROUND_Y - oy) / dy
    if (t > 1e-4) {
      const hx = ox + t * dx
      const hz = oz + t * dz
      if (hx * hx + hz * hz < GROUND_RADIUS * GROUND_RADIUS) tGround = t
    }
  }

  if (tSphere === Infinity && tGround === Infinity) {
    out[0] = SYNTHETIC_BACKGROUND[0]
    out[1] = SYNTHETIC_BACKGROUND[1]
    out[2] = SYNTHETIC_BACKGROUND[2]
    return
  }

  if (tGround < tSphere) {
    const hx = ox + tGround * dx
    const hy = GROUND_Y
    const hz = oz + tGround * dz
    const checker = ((Math.floor(hx * 2.5) + Math.floor(hz * 2.5)) & 1) === 0
    const base: [number, number, number] = checker ? [0.72, 0.72, 0.75] : [0.24, 0.25, 0.3]
    // Fade the disc edge into the background so there is no hard rim to fit.
    const edge = Math.min(1, (GROUND_RADIUS - Math.hypot(hx, hz)) / 0.35)
    shade(0, 1, 0, dx, dy, dz, base, 0.08, 16, hx, hy, hz, out)
    out[0] = out[0] * edge + SYNTHETIC_BACKGROUND[0] * (1 - edge)
    out[1] = out[1] * edge + SYNTHETIC_BACKGROUND[1] * (1 - edge)
    out[2] = out[2] * edge + SYNTHETIC_BACKGROUND[2] * (1 - edge)
    return
  }

  const s = SPHERES[sphere]
  const hx = ox + tSphere * dx
  const hy = oy + tSphere * dy
  const hz = oz + tSphere * dz
  const nx = (hx - s.center[0]) / s.radius
  const ny = (hy - s.center[1]) / s.radius
  const nz = (hz - s.center[2]) / s.radius
  shade(nx, ny, nz, dx, dy, dz, s.albedo, s.specular, s.shininess, hx, hy, hz, out)
}

/** Renders the scene from one camera pose into an RGB Float32Array. */
export function renderSyntheticView(
  pose: Mat4,
  intrinsics: Intrinsics,
  width: number,
  height: number,
): Float32Array {
  const pixels = new Float32Array(width * height * 3)
  const origin = new Float32Array(3)
  const dir = new Float32Array(3)
  const color: [number, number, number] = [0, 0, 0]
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      rayForPixel(pose, intrinsics, x, y, origin, dir)
      traceRay(origin[0], origin[1], origin[2], dir[0], dir[1], dir[2], color)
      const o = (y * width + x) * 3
      pixels[o] = color[0]
      pixels[o + 1] = color[1]
      pixels[o + 2] = color[2]
    }
  }
  return pixels
}

export interface SyntheticOptions {
  viewCount?: number
  resolution?: number
  radius?: number
  fovDegrees?: number
}

/** Builds a ready-to-train dataset from the procedural scene. */
export function buildSyntheticDataset(opts: SyntheticOptions = {}): Dataset {
  const viewCount = opts.viewCount ?? 40
  const resolution = opts.resolution ?? 96
  const radius = opts.radius ?? 3.6
  const fov = opts.fovDegrees ?? 42

  const poses = fibonacciOrbit(viewCount, radius)
  const intrinsics = defaultIntrinsics(resolution, resolution, fov)
  const views: TrainingView[] = poses.map((pose, i) => ({
    id: `sentetik-${i + 1}`,
    width: resolution,
    height: resolution,
    pixels: renderSyntheticView(pose, intrinsics, resolution, resolution),
    pose,
    intrinsics,
  }))

  return {
    views,
    aabbSize: SYNTHETIC_AABB,
    background: SYNTHETIC_BACKGROUND,
  }
}
