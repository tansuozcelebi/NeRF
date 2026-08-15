/**
 * Finite-difference gradient checks.
 *
 * Every derivative in this project is written by hand, so these tests are the
 * safety net: if a sign or a transmittance term were wrong, training would
 * still "run" and just quietly produce mush.
 */
import { describe, expect, it } from 'vitest'
import { RadianceField } from '../field'
import { HashGrid } from '../hashGrid'
import { Linear } from '../mlp'
import { makeRng } from '../random'
import type { ModelConfig } from '../types'
import { VolumeRenderer } from '../volumeRender'

/**
 * Central differences on float32 storage. The step is chosen near
 * cbrt(float32 epsilon) ~ 4e-3, which balances truncation against round-off.
 */
const EPS = 4e-3
/** Round-off floor: a float32 loss of order 1 resolves ~6e-8, amplified by 1/EPS. */
const ATOL = 3e-4
const RTOL = 6e-3

function expectGradClose(numeric: number, analytic: number, rtol = RTOL): void {
  const tolerance = ATOL + rtol * Math.max(Math.abs(numeric), Math.abs(analytic))
  expect(Math.abs(numeric - analytic)).toBeLessThanOrEqual(tolerance)
}

describe('Linear layer', () => {
  it('matches numeric gradients for weights, bias and input', () => {
    const rng = makeRng(7)
    const layer = new Linear(4, 3, 5)
    const batch = 2
    const input = new Float32Array(batch * 4).map(() => rng() * 2 - 1)
    const out = new Float32Array(batch * 3)
    const gradOut = new Float32Array(batch * 3).map(() => rng() * 2 - 1)
    const gradIn = new Float32Array(batch * 4)

    layer.forward(input, out, batch)
    layer.backward(input, gradOut, gradIn, batch)

    const loss = (): number => {
      const o = new Float32Array(batch * 3)
      layer.forward(input, o, batch)
      let s = 0
      for (let i = 0; i < o.length; i++) s += o[i] * gradOut[i]
      return s
    }

    for (let i = 0; i < layer.params.length; i++) {
      const original = layer.params[i]
      layer.params[i] = original + EPS
      const up = loss()
      layer.params[i] = original - EPS
      const down = loss()
      layer.params[i] = original
      expectGradClose((up - down) / (2 * EPS), layer.grads[i])
    }

    for (let i = 0; i < input.length; i++) {
      const original = input[i]
      input[i] = original + EPS
      const up = loss()
      input[i] = original - EPS
      const down = loss()
      input[i] = original
      expectGradClose((up - down) / (2 * EPS), gradIn[i])
    }
  })
})

describe('hash grid', () => {
  const config = {
    levels: 3,
    featuresPerLevel: 2,
    log2TableSize: 8,
    baseResolution: 4,
    maxResolution: 16,
  }

  it('interpolates trilinearly and sums to one across the 8 corners', () => {
    const grid = new HashGrid(config, 3)
    grid.params.fill(1)
    const out = new Float32Array(grid.outputDim)
    grid.encode(0.37, 0.62, 0.11, out, 0)
    // Every corner holds 1 and the weights partition unity.
    for (let i = 0; i < out.length; i++) expect(out[i]).toBeCloseTo(1, 5)
  })

  it('scatters gradients that match finite differences', () => {
    const grid = new HashGrid(config, 11)
    const rng = makeRng(99)
    for (let i = 0; i < grid.params.length; i++) grid.params[i] = rng() * 2 - 1

    const p: [number, number, number] = [0.31, 0.77, 0.44]
    const out = new Float32Array(grid.outputDim)
    const gradOut = new Float32Array(grid.outputDim).map(() => rng() * 2 - 1)

    grid.zeroGrad()
    grid.encode(p[0], p[1], p[2], out, 0)
    grid.accumulate(p[0], p[1], p[2], gradOut, 0)

    const loss = (): number => {
      const o = new Float32Array(grid.outputDim)
      grid.encode(p[0], p[1], p[2], o, 0)
      let s = 0
      for (let i = 0; i < o.length; i++) s += o[i] * gradOut[i]
      return s
    }

    const { list, count } = grid.touched
    expect(count).toBeGreaterThan(0)
    for (let e = 0; e < count; e++) {
      for (let f = 0; f < config.featuresPerLevel; f++) {
        const idx = list[e] * config.featuresPerLevel + f
        const original = grid.params[idx]
        grid.params[idx] = original + EPS
        const up = loss()
        grid.params[idx] = original - EPS
        const down = loss()
        grid.params[idx] = original
        expectGradClose((up - down) / (2 * EPS), grid.grads[idx])
      }
    }
  })

  it('only clears the entries it touched', () => {
    const grid = new HashGrid(config, 5)
    const gradOut = new Float32Array(grid.outputDim).fill(1)
    grid.accumulate(0.5, 0.5, 0.5, gradOut, 0)
    expect(grid.touched.count).toBeGreaterThan(0)
    grid.zeroGrad()
    expect(grid.touched.count).toBe(0)
    let sum = 0
    for (let i = 0; i < grid.grads.length; i++) sum += Math.abs(grid.grads[i])
    expect(sum).toBe(0)
  })
})

describe('volume rendering', () => {
  it('composites to the background when the volume is empty', () => {
    const renderer = new VolumeRenderer(1, 8, 1, [0.2, 0.4, 0.6])
    const origins = new Float32Array([0, 0, 3])
    const dirs = new Float32Array([0, 0, -1])
    const stats = renderer.sample(origins, dirs, 1, null, () => 0.5)
    const sigma = new Float32Array(stats.points) // all zero
    const rgb = new Float32Array(stats.points * 3).fill(1)
    renderer.composite(sigma, rgb, 1)
    expect(renderer.color[0]).toBeCloseTo(0.2, 5)
    expect(renderer.color[1]).toBeCloseTo(0.4, 5)
    expect(renderer.color[2]).toBeCloseTo(0.6, 5)
    expect(renderer.acc[0]).toBeCloseTo(0, 5)
  })

  it('composites to the sample colour when the volume is opaque', () => {
    const renderer = new VolumeRenderer(1, 8, 1, [0, 0, 0])
    const origins = new Float32Array([0, 0, 3])
    const dirs = new Float32Array([0, 0, -1])
    const stats = renderer.sample(origins, dirs, 1, null, () => 0.5)
    const sigma = new Float32Array(stats.points).fill(500)
    const rgb = new Float32Array(stats.points * 3).fill(0.75)
    renderer.composite(sigma, rgb, 1)
    expect(renderer.color[0]).toBeCloseTo(0.75, 4)
    expect(renderer.acc[0]).toBeCloseTo(1, 4)
  })

  it('matches numeric gradients for density and colour', () => {
    const rng = makeRng(21)
    const renderer = new VolumeRenderer(2, 6, 1, [0.3, 0.1, 0.8])
    const origins = new Float32Array([0, 0, 3, 0.2, 0.1, 3])
    const dirs = new Float32Array([0, 0, -1, -0.05, -0.02, -0.998])
    // Normalise the second direction.
    const l = Math.hypot(dirs[3], dirs[4], dirs[5])
    dirs[3] /= l; dirs[4] /= l; dirs[5] /= l

    const stats = renderer.sample(origins, dirs, 2, null, () => 0.5)
    const n = stats.points
    expect(n).toBe(12)

    const sigma = new Float32Array(n).map(() => rng() * 3)
    const rgb = new Float32Array(n * 3).map(() => rng())
    const gradColor = new Float32Array(6).map(() => rng() * 2 - 1)

    renderer.composite(sigma, rgb, 2)
    const gradSigma = new Float32Array(n)
    const gradRgb = new Float32Array(n * 3)
    renderer.backward(gradColor, 2, rgb, gradSigma, gradRgb)

    const loss = (): number => {
      renderer.composite(sigma, rgb, 2)
      let s = 0
      for (let i = 0; i < 6; i++) s += renderer.color[i] * gradColor[i]
      return s
    }

    for (let j = 0; j < n; j++) {
      const original = sigma[j]
      sigma[j] = original + EPS
      const up = loss()
      sigma[j] = original - EPS
      const down = loss()
      sigma[j] = original
      expectGradClose((up - down) / (2 * EPS), gradSigma[j])
    }

    for (let j = 0; j < n * 3; j++) {
      const original = rgb[j]
      rgb[j] = original + EPS
      const up = loss()
      rgb[j] = original - EPS
      const down = loss()
      rgb[j] = original
      expectGradClose((up - down) / (2 * EPS), gradRgb[j])
    }
  })
})

describe('radiance field', () => {
  const config: ModelConfig = {
    grid: {
      levels: 2,
      featuresPerLevel: 2,
      log2TableSize: 8,
      baseResolution: 4,
      maxResolution: 8,
    },
    hiddenSize: 8,
    geoFeatureSize: 4,
    shDegree: 2,
    seed: 4242,
  }

  it('backpropagates density and colour to every parameter', () => {
    const field = new RadianceField(config)
    const rng = makeRng(17)

    // Move away from the initialisation point before checking gradients. At
    // init the hash features are ~1e-4 and every bias is 0, so all hidden
    // pre-activations sit exactly on the ReLU kink: a central difference then
    // straddles the corner and reports roughly half the true derivative. That
    // is an artefact of finite differences, not of the backward pass.
    for (let i = 0; i < field.grid.params.length; i++) field.grid.params[i] = rng() * 2 - 1
    for (const layer of field.modules) {
      for (let i = 0; i < layer.bias.length; i++) layer.bias[i] = rng() * 0.6 - 0.3
    }

    const n = 3
    const pos = new Float32Array([0.3, 0.4, 0.5, 0.62, 0.15, 0.88, 0.5, 0.5, 0.5])
    const dir = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      const x = rng() * 2 - 1, y = rng() * 2 - 1, z = rng() * 2 - 1
      const len = Math.hypot(x, y, z)
      dir[i * 3] = x / len; dir[i * 3 + 1] = y / len; dir[i * 3 + 2] = z / len
    }
    const gradSigma = new Float32Array(n).map(() => rng() * 2 - 1)
    const gradRgb = new Float32Array(n * 3).map(() => rng() * 2 - 1)

    const loss = (): number => {
      field.forward(pos, dir, n)
      let s = 0
      for (let i = 0; i < n; i++) s += field.sigma[i] * gradSigma[i]
      for (let i = 0; i < n * 3; i++) s += field.rgb[i] * gradRgb[i]
      return s
    }

    field.zeroGrad()
    field.forward(pos, dir, n)
    field.backward(gradSigma, gradRgb, n)

    // Snapshot the analytic gradients before the numeric passes overwrite state.
    const gridGrads = Float32Array.from(field.grid.grads)
    const touched = { ...field.grid.touched, list: Int32Array.from(field.grid.touched.list) }
    const layerGrads = field.modules.map((l) => Float32Array.from(l.grads))

    const check = (params: Float32Array, grads: Float32Array, idx: number) => {
      const original = params[idx]
      params[idx] = original + EPS
      const up = loss()
      params[idx] = original - EPS
      const down = loss()
      params[idx] = original
      expectGradClose((up - down) / (2 * EPS), grads[idx])
    }

    // Hash-grid features that took part in this batch.
    for (let e = 0; e < touched.count; e++) {
      for (let f = 0; f < config.grid.featuresPerLevel; f++) {
        check(field.grid.params, gridGrads, touched.list[e] * config.grid.featuresPerLevel + f)
      }
    }

    // Every weight and bias of the four dense layers.
    field.modules.forEach((layer, li) => {
      for (let i = 0; i < layer.params.length; i++) check(layer.params, layerGrads[li], i)
    })
  })
})
