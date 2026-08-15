/**
 * Where the app thinks each photo was taken from.
 *
 * NeRF cannot be trained without camera poses, and recovering them from loose
 * photos needs structure-from-motion, which is not something a browser tab can
 * do. So the app is explicit about it and offers three honest options:
 *
 *  - `halka`  the photos were taken walking a circle around the subject,
 *  - `kubbe`  the same but with the height varying over the sequence,
 *  - `dosya`  real poses imported from a transforms.json produced elsewhere.
 *
 * The first two are *assumptions*. They reconstruct beautifully when the
 * capture actually matched them and produce a smeared mess when it did not,
 * which the UI says out loud.
 */
import { fibonacciOrbit, ringOrbit } from '../nerf/camera'
import type { Mat4 } from '../nerf/types'
import type { ImportedPoses } from '../utils/transforms'
import { matchPosesToPhotos } from '../utils/transforms'

export type PoseMode = 'halka' | 'kubbe' | 'dosya'

export interface PoseConfig {
  mode: PoseMode
  /** Distance from the subject, in scene units. */
  radius: number
  /** Camera height for `halka`, and the lower bound for `kubbe`. */
  elevationDeg: number
  /** Upper elevation bound for `kubbe`. */
  elevationMaxDeg: number
  /** Horizontal field of view of the capturing camera. */
  fovDegrees: number
  /** Half-size of the box the scene is assumed to fit in. */
  aabbSize: number
}

export const DEFAULT_POSE_CONFIG: PoseConfig = {
  mode: 'halka',
  radius: 3.4,
  elevationDeg: 18,
  elevationMaxDeg: 55,
  fovDegrees: 50,
  aabbSize: 1.9,
}

export interface BuiltPoses {
  poses: Mat4[]
  /** Human-readable note about how the poses were obtained. */
  note: string
  /** True when something is off and training would likely fail. */
  problem: boolean
}

export function buildPoses(
  config: PoseConfig,
  photoNames: string[],
  imported: ImportedPoses | null,
): BuiltPoses {
  const count = photoNames.length
  if (count === 0) {
    return { poses: [], note: 'Henüz fotoğraf yüklenmedi.', problem: true }
  }

  if (config.mode === 'dosya') {
    if (!imported) {
      return {
        poses: [],
        note: 'Bir transforms.json dosyası seçin.',
        problem: true,
      }
    }
    const matched = matchPosesToPhotos(imported, photoNames)
    return {
      poses: matched.poses,
      note: matched.matchedByName
        ? `${count} fotoğrafın tamamı dosyadaki karelerle dosya adına göre eşleşti.`
        : `Dosya adları eşleşmedi (${matched.missing.length} fotoğraf), yükleme sırası kullanıldı.`,
      problem: !matched.matchedByName,
    }
  }

  const poses =
    config.mode === 'halka'
      ? ringOrbit(count, config.radius, config.elevationDeg)
      : fibonacciOrbit(count, config.radius, config.elevationDeg, config.elevationMaxDeg)

  return {
    poses,
    note:
      config.mode === 'halka'
        ? 'Fotoğrafların, özne etrafında sabit yükseklikte ve eşit aralıklı çekildiği varsayılıyor.'
        : 'Fotoğrafların, özne etrafında yükseklik değiştirerek bir kubbe boyunca çekildiği varsayılıyor.',
    problem: count < 8,
  }
}
