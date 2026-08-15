/** Small deterministic PRNG so runs are reproducible from a seed. */

/** mulberry32 — fast, good enough for weight init and ray sampling. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0
  return function next(): number {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Standard normal samples via Box–Muller. */
export function makeGaussian(rng: () => number): () => number {
  let spare: number | null = null
  return function gauss(): number {
    if (spare !== null) {
      const value = spare
      spare = null
      return value
    }
    let u = 0
    let v = 0
    let s = 0
    do {
      u = rng() * 2 - 1
      v = rng() * 2 - 1
      s = u * u + v * v
    } while (s === 0 || s >= 1)
    const factor = Math.sqrt((-2 * Math.log(s)) / s)
    spare = v * factor
    return u * factor
  }
}
