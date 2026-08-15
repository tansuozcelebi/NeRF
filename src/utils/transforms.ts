/**
 * Import of `transforms.json`, the pose file produced by COLMAP wrappers and
 * used by NeRF Studio / Instant-NGP.
 *
 * This is the path for people who *do* have real camera poses: run the
 * structure-from-motion step in a desktop tool, then bring the result here.
 */
import { fovFromFocal, normalizePoses } from '../nerf/camera'
import type { Mat4 } from '../nerf/types'

export interface ImportedPoses {
  /** Pose per frame, in file order, already recentred around the origin. */
  poses: Mat4[]
  /** File name of each frame, used to match uploaded photos. */
  names: string[]
  /** Horizontal field of view in degrees, if the file specified one. */
  fovDegrees: number | null
}

interface RawFrame {
  file_path?: string
  transform_matrix?: number[][]
}

interface RawTransforms {
  camera_angle_x?: number
  fl_x?: number
  w?: number
  frames?: RawFrame[]
}

function baseName(path: string): string {
  const file = path.split(/[\\/]/).pop() ?? path
  return file.replace(/\.[^.]+$/, '').toLowerCase()
}

/** Parses a transforms.json string. Throws with a readable Turkish message. */
export function parseTransforms(text: string): ImportedPoses {
  let raw: RawTransforms
  try {
    raw = JSON.parse(text) as RawTransforms
  } catch {
    return failure('Dosya geçerli bir JSON değil.')
  }
  if (!raw.frames || !Array.isArray(raw.frames) || raw.frames.length === 0) {
    return failure('Dosyada "frames" listesi bulunamadı.')
  }

  const poses: Mat4[] = []
  const names: string[] = []
  for (const frame of raw.frames) {
    const m = frame.transform_matrix
    if (!Array.isArray(m) || m.length < 4 || m.some((row) => !Array.isArray(row) || row.length < 4)) {
      return failure('Bir karenin "transform_matrix" değeri 4x4 değil.')
    }
    const pose = new Float32Array(16)
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) pose[r * 4 + c] = m[r][c]
    }
    poses.push(pose)
    names.push(baseName(frame.file_path ?? `kare-${poses.length}`))
  }

  let fovDegrees: number | null = null
  if (typeof raw.camera_angle_x === 'number') {
    fovDegrees = (raw.camera_angle_x * 180) / Math.PI
  } else if (typeof raw.fl_x === 'number' && typeof raw.w === 'number') {
    fovDegrees = (fovFromFocal(raw.fl_x, raw.w) * 180) / Math.PI
  }

  return { poses: normalizePoses(poses), names, fovDegrees }
}

function failure(message: string): never {
  throw new Error(`transforms.json okunamadı: ${message}`)
}

/**
 * Matches imported poses to uploaded photos by file name, falling back to
 * upload order when the names do not line up.
 */
export function matchPosesToPhotos(
  imported: ImportedPoses,
  photoNames: string[],
): { poses: Mat4[]; matchedByName: boolean; missing: string[] } {
  const index = new Map<string, number>()
  imported.names.forEach((name, i) => index.set(name, i))

  const missing: string[] = []
  const byName: Mat4[] = []
  for (const photoName of photoNames) {
    const i = index.get(baseName(photoName))
    if (i === undefined) missing.push(photoName)
    else byName.push(imported.poses[i])
  }

  if (missing.length === 0 && byName.length === photoNames.length) {
    return { poses: byName, matchedByName: true, missing: [] }
  }
  return {
    poses: photoNames.map((_, i) => imported.poses[Math.min(i, imported.poses.length - 1)]),
    matchedByName: false,
    missing,
  }
}
