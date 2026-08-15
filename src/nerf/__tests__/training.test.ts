/**
 * End-to-end proof that the pipeline reconstructs a scene: train on the
 * synthetic dataset (where the camera poses are exact by construction) and
 * check that a *held-out* viewpoint gets closer to the ground truth.
 *
 * Held-out matters — fitting the training views is easy, synthesising a novel
 * view is the whole point of a NeRF.
 */
import { describe, expect, it } from 'vitest'
import { defaultIntrinsics, orbitPose } from '../camera'
import { NerfTrainer } from '../trainer'
import { buildSyntheticDataset, renderSyntheticView } from '../syntheticScene'
import { DEFAULT_MODEL_CONFIG, DEFAULT_TRAIN_CONFIG } from '../types'

describe('NeRF training', () => {
  it('reduces loss and generalises to an unseen camera angle', () => {
    const dataset = buildSyntheticDataset({ viewCount: 24, resolution: 40, radius: 3.6 })
    const trainer = new NerfTrainer(
      dataset,
      { ...DEFAULT_MODEL_CONFIG, grid: { ...DEFAULT_MODEL_CONFIG.grid, log2TableSize: 14 } },
      { ...DEFAULT_TRAIN_CONFIG, raysPerStep: 512, samplesPerRay: 32 },
    )

    // A pose that is deliberately not in the training set.
    const heldOutPose = orbitPose(0.77, 0.42, 3.6)
    const size = 32
    const intr = defaultIntrinsics(size, size, 42)
    const truth = renderSyntheticView(heldOutPose, intr, size, size)

    const noveltyError = (): number => {
      const { rgba } = trainer.render(heldOutPose, intr, { width: size, height: size })
      let sse = 0
      for (let i = 0; i < size * size; i++) {
        for (let c = 0; c < 3; c++) {
          const diff = rgba[i * 4 + c] / 255 - truth[i * 3 + c]
          sse += diff * diff
        }
      }
      return sse / (size * size * 3)
    }

    const errorBefore = noveltyError()

    let firstLoss = 0
    let lastLoss = 0
    for (let i = 0; i < 300; i++) {
      const stats = trainer.trainStep()
      if (i === 0) firstLoss = stats.loss
      lastLoss = stats.loss
      expect(Number.isFinite(stats.loss)).toBe(true)
    }

    const errorAfter = noveltyError()

    // Training loss must drop substantially.
    expect(lastLoss).toBeLessThan(firstLoss * 0.5)
    // And the novel view must actually get better, by a clear margin.
    expect(errorAfter).toBeLessThan(errorBefore * 0.5)
    // 20 dB PSNR on an unseen view after 300 steps is a low bar the model clears.
    expect(-10 * Math.log10(errorAfter)).toBeGreaterThan(20)
  }, 120_000)

  it('never prunes the volume down to nothing', () => {
    // Regression: the initial density used to sit just below the occupancy
    // cutoff, so the first refresh could mark every cell empty. With no cells
    // occupied no samples are taken, no gradients flow, and training is stuck
    // for good — the model can never climb back over the threshold.
    const dataset = buildSyntheticDataset({ viewCount: 8, resolution: 24 })
    const trainer = new NerfTrainer(dataset, DEFAULT_MODEL_CONFIG, {
      ...DEFAULT_TRAIN_CONFIG,
      raysPerStep: 128,
      samplesPerRay: 24,
      occupancyRefreshInterval: 4,
      // An aggressive prior makes the collapse far more likely to show up.
      sparsityWeight: 0.05,
    })
    for (let i = 0; i < 200; i++) {
      trainer.trainStep()
      expect(trainer.occupancy.occupiedFraction).toBeGreaterThan(0)
    }
    expect(trainer.lastPointCount).toBeGreaterThan(0)
  }, 120_000)

  it('prunes empty space once the occupancy grid warms up', () => {
    const dataset = buildSyntheticDataset({ viewCount: 12, resolution: 32 })
    const trainer = new NerfTrainer(dataset, DEFAULT_MODEL_CONFIG, {
      ...DEFAULT_TRAIN_CONFIG,
      raysPerStep: 256,
      samplesPerRay: 32,
      occupancyRefreshInterval: 8,
    })
    for (let i = 0; i < 120; i++) trainer.trainStep()
    expect(trainer.occupancy.occupiedFraction).toBeLessThan(0.9)
    expect(trainer.lastPointCount).toBeGreaterThan(0)
  }, 120_000)
})
