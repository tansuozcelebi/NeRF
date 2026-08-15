/**
 * A batched fully-connected layer with hand-written forward and backward
 * passes. Everything is plain Float32Array arithmetic so it runs in a worker
 * with no dependencies and no GPU requirement.
 *
 * Layout: `weights` is row-major [outDim][inDim], inputs and outputs are
 * [batch][dim]. Weights and biases live in one flat `params` array so an
 * optimiser can own the layer with a single view.
 */
import { makeGaussian, makeRng } from './random'

export class Linear {
  readonly inDim: number
  readonly outDim: number
  /** Flat parameter storage: [weights (outDim*inDim), bias (outDim)]. */
  readonly params: Float32Array
  readonly grads: Float32Array
  readonly weights: Float32Array
  readonly bias: Float32Array
  private readonly weightGrads: Float32Array
  private readonly biasGrads: Float32Array

  constructor(inDim: number, outDim: number, seed: number, initScale?: number) {
    this.inDim = inDim
    this.outDim = outDim
    const wCount = inDim * outDim
    this.params = new Float32Array(wCount + outDim)
    this.grads = new Float32Array(wCount + outDim)
    this.weights = this.params.subarray(0, wCount)
    this.bias = this.params.subarray(wCount)
    this.weightGrads = this.grads.subarray(0, wCount)
    this.biasGrads = this.grads.subarray(wCount)

    // He initialisation — the hidden layers use ReLU.
    const gauss = makeGaussian(makeRng(seed))
    const scale = initScale ?? Math.sqrt(2 / inDim)
    for (let i = 0; i < wCount; i++) this.weights[i] = gauss() * scale
  }

  /** out[b][o] = bias[o] + sum_i w[o][i] * input[b][i] */
  forward(input: Float32Array, out: Float32Array, batch: number): void {
    const { inDim, outDim, weights, bias } = this
    for (let b = 0; b < batch; b++) {
      const inBase = b * inDim
      const outBase = b * outDim
      for (let o = 0; o < outDim; o++) {
        const wBase = o * inDim
        let acc = bias[o]
        for (let i = 0; i < inDim; i++) acc += weights[wBase + i] * input[inBase + i]
        out[outBase + o] = acc
      }
    }
  }

  /**
   * Accumulates weight/bias gradients and, when `gradIn` is given, propagates
   * the gradient to the layer input. `gradIn` is overwritten, not accumulated.
   */
  backward(
    input: Float32Array,
    gradOut: Float32Array,
    gradIn: Float32Array | null,
    batch: number,
  ): void {
    const { inDim, outDim, weights, weightGrads, biasGrads } = this
    // The `gradIn` test is hoisted out of the innermost loop: this runs once
    // per sample per unit per input and is the hottest loop in training.
    if (gradIn) {
      gradIn.fill(0, 0, batch * inDim)
      for (let b = 0; b < batch; b++) {
        const inBase = b * inDim
        const outBase = b * outDim
        for (let o = 0; o < outDim; o++) {
          const g = gradOut[outBase + o]
          if (g === 0) continue
          biasGrads[o] += g
          const wBase = o * inDim
          for (let i = 0; i < inDim; i++) {
            weightGrads[wBase + i] += g * input[inBase + i]
            gradIn[inBase + i] += g * weights[wBase + i]
          }
        }
      }
      return
    }
    for (let b = 0; b < batch; b++) {
      const inBase = b * inDim
      const outBase = b * outDim
      for (let o = 0; o < outDim; o++) {
        const g = gradOut[outBase + o]
        if (g === 0) continue
        biasGrads[o] += g
        const wBase = o * inDim
        for (let i = 0; i < inDim; i++) weightGrads[wBase + i] += g * input[inBase + i]
      }
    }
  }

  zeroGrad(): void {
    this.grads.fill(0)
  }

  get paramCount(): number {
    return this.params.length
  }
}

/** In-place ReLU over the first `count` values. */
export function relu(x: Float32Array, count: number): void {
  for (let i = 0; i < count; i++) if (x[i] < 0) x[i] = 0
}

/** Backward of an in-place ReLU, given the post-activation values. */
export function reluBackward(activated: Float32Array, grad: Float32Array, count: number): void {
  for (let i = 0; i < count; i++) if (activated[i] <= 0) grad[i] = 0
}

export function sigmoid(x: number): number {
  return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x))
}
