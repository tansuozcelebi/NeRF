/**
 * Real spherical-harmonics encoding of a unit view direction.
 *
 * This is what makes a NeRF view-dependent: the colour MLP sees *where the
 * camera is looking from*, so it can reproduce specular highlights and
 * reflections that change as you move around the object, instead of baking one
 * flat colour per point.
 */

/** Number of coefficients for a given degree (degree 3 -> 16). */
export function shDim(degree: number): number {
  const bands = degree + 1
  return bands * bands
}

/**
 * Evaluates the SH basis up to `degree` (max 3) for the unit vector (x, y, z),
 * writing `shDim(degree)` values at `out[offset]`.
 */
export function encodeDirection(
  x: number, y: number, z: number,
  degree: number,
  out: Float32Array,
  offset: number,
): void {
  out[offset] = 0.28209479177387814
  if (degree < 1) return

  out[offset + 1] = -0.48860251190291987 * y
  out[offset + 2] = 0.48860251190291987 * z
  out[offset + 3] = -0.48860251190291987 * x
  if (degree < 2) return

  const xx = x * x, yy = y * y, zz = z * z
  const xy = x * y, yz = y * z, xz = x * z
  out[offset + 4] = 1.0925484305920792 * xy
  out[offset + 5] = -1.0925484305920792 * yz
  out[offset + 6] = 0.94617469575755997 * zz - 0.31539156525251999
  out[offset + 7] = -1.0925484305920792 * xz
  out[offset + 8] = 0.54627421529603959 * (xx - yy)
  if (degree < 3) return

  out[offset + 9] = 0.59004358992664352 * y * (-3 * xx + yy)
  out[offset + 10] = 2.8906114426405538 * xy * z
  out[offset + 11] = 0.45704579946446572 * y * (1 - 5 * zz)
  out[offset + 12] = 0.3731763325901154 * z * (5 * zz - 3)
  out[offset + 13] = 0.45704579946446572 * x * (1 - 5 * zz)
  out[offset + 14] = 1.4453057213202769 * z * (xx - yy)
  out[offset + 15] = 0.59004358992664352 * x * (xx - 3 * yy)
}
