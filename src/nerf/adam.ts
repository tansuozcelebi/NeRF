/** Adam optimiser over a flat Float32Array of parameters. */

export interface AdamOptions {
  beta1?: number
  beta2?: number
  eps?: number
}

export class Adam {
  readonly params: Float32Array
  readonly grads: Float32Array
  private readonly m: Float32Array
  private readonly v: Float32Array
  private readonly beta1: number
  private readonly beta2: number
  private readonly eps: number
  /** Number of updates applied so far, used for bias correction. */
  t = 0

  /**
   * @param grads Gradient buffer to read from. Pass the owner's own buffer when
   *              it already has one (the hash grid and each layer do), so the
   *              optimiser and the module stay in sync.
   */
  constructor(params: Float32Array, grads?: Float32Array, opts: AdamOptions = {}) {
    this.params = params
    this.grads = grads ?? new Float32Array(params.length)
    this.m = new Float32Array(params.length)
    this.v = new Float32Array(params.length)
    this.beta1 = opts.beta1 ?? 0.9
    this.beta2 = opts.beta2 ?? 0.99
    this.eps = opts.eps ?? 1e-8
  }

  zeroGrad(): void {
    this.grads.fill(0)
  }

  /** Applies Adam to every parameter. */
  step(lr: number): void {
    this.t++
    const { params, grads, m, v, beta1, beta2, eps } = this
    const bc1 = 1 - Math.pow(beta1, this.t)
    const bc2 = 1 - Math.pow(beta2, this.t)
    for (let i = 0; i < params.length; i++) {
      const g = grads[i]
      const mi = beta1 * m[i] + (1 - beta1) * g
      const vi = beta2 * v[i] + (1 - beta2) * g * g
      m[i] = mi
      v[i] = vi
      params[i] -= (lr * (mi / bc1)) / (Math.sqrt(vi / bc2) + eps)
    }
  }

  /**
   * Applies Adam only to the entries listed in `entries[0..count)`, where each
   * entry covers `stride` consecutive parameters. Hash-grid features are touched
   * by a tiny fraction of each batch, so updating everything would dominate the
   * step cost (and would decay momentum for cells that saw no gradient).
   */
  stepSparse(entries: Int32Array, count: number, stride: number, lr: number): void {
    this.t++
    const { params, grads, m, v, beta1, beta2, eps } = this
    const bc1 = 1 - Math.pow(beta1, this.t)
    const bc2 = 1 - Math.pow(beta2, this.t)
    for (let e = 0; e < count; e++) {
      const base = entries[e] * stride
      for (let k = 0; k < stride; k++) {
        const i = base + k
        const g = grads[i]
        const mi = beta1 * m[i] + (1 - beta1) * g
        const vi = beta2 * v[i] + (1 - beta2) * g * g
        m[i] = mi
        v[i] = vi
        params[i] -= (lr * (mi / bc1)) / (Math.sqrt(vi / bc2) + eps)
      }
    }
  }
}
