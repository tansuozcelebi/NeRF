import { describe, expect, it } from 'vitest'
import {
  defaultIntrinsics,
  focalFromFov,
  fovFromFocal,
  intersectAabb,
  lookAt,
  normalizePoses,
  orbitPose,
  poseOrigin,
  rayForPixel,
} from '../camera'

describe('camera', () => {
  it('places the camera at the eye point and looks at the target', () => {
    const pose = lookAt([0, 0, 4], [0, 0, 0])
    expect(Array.from(poseOrigin(pose))).toEqual([0, 0, 4])

    const o = new Float32Array(3)
    const d = new Float32Array(3)
    const intr = defaultIntrinsics(64, 64)
    rayForPixel(pose, intr, 31.5, 31.5, o, d)
    // Centre pixel of a camera at +Z looking at the origin points along -Z.
    expect(d[0]).toBeCloseTo(0, 5)
    expect(d[1]).toBeCloseTo(0, 5)
    expect(d[2]).toBeCloseTo(-1, 5)
  })

  it('keeps the image upright: the top row points upwards in world space', () => {
    const pose = lookAt([0, 0, 4], [0, 0, 0])
    const o = new Float32Array(3)
    const d = new Float32Array(3)
    rayForPixel(pose, defaultIntrinsics(64, 64), 31.5, 0, o, d)
    expect(d[1]).toBeGreaterThan(0)
  })

  it('orbits around the target at the requested radius', () => {
    for (const azimuth of [0, 1.2, 3.0, 5.5]) {
      const pose = orbitPose(azimuth, 0.4, 3)
      const [x, y, z] = poseOrigin(pose)
      expect(Math.hypot(x, y, z)).toBeCloseTo(3, 4)
      expect(y).toBeCloseTo(3 * Math.sin(0.4), 4)
    }
  })

  it('round-trips focal length and field of view', () => {
    const focal = focalFromFov((50 * Math.PI) / 180, 128)
    expect((fovFromFocal(focal, 128) * 180) / Math.PI).toBeCloseTo(50, 6)
  })

  it('intersects the bounding box only when the ray actually hits it', () => {
    const hit = intersectAabb(0, 0, 4, 0, 0, -1, 1)
    expect(hit).not.toBeNull()
    expect(hit![0]).toBeCloseTo(3, 5)
    expect(hit![1]).toBeCloseTo(5, 5)

    expect(intersectAabb(0, 5, 4, 0, 0, -1, 1)).toBeNull()
  })

  it('clamps tNear to zero for cameras inside the box', () => {
    const hit = intersectAabb(0, 0, 0, 0, 0, -1, 1)
    expect(hit![0]).toBe(0)
    expect(hit![1]).toBeCloseTo(1, 5)
  })

  it('recentres and rescales poses so they fit the unit box', () => {
    const poses = [
      lookAt([10, 0, 20], [10, 0, 10]),
      lookAt([10, 0, 0], [10, 0, 10]),
      lookAt([20, 0, 10], [10, 0, 10]),
    ]
    const normalized = normalizePoses(poses, 2)
    let maxDist = 0
    let cx = 0
    for (const p of normalized) {
      const [x, y, z] = poseOrigin(p)
      cx += x
      maxDist = Math.max(maxDist, Math.hypot(x, y, z))
    }
    expect(cx / normalized.length).toBeCloseTo(0, 4)
    expect(maxDist).toBeCloseTo(2, 4)
  })
})
