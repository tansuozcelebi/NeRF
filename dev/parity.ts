/**
 * Development harness: does the GPU shader agree with the CPU renderer?
 *
 * The shader in `src/gpu/nerfShader.ts` is a second, independent implementation
 * of the hash encoding, both MLPs and the volume compositing. Nothing stops the
 * two from drifting apart except a test that renders the same model from the
 * same camera through both paths and compares the pixels — which is what this
 * page does.
 *
 * It runs a single explicit frame rather than an animation loop, so it works
 * even where WebGL is emulated on the CPU and far too slow to be interactive.
 *
 * Open it with `npm run dev` at /dev/parity.html. It is not part of the
 * production build.
 */
import { defaultIntrinsics, orbitPose } from '../src/nerf/camera'
import { GpuNerfRenderer } from '../src/gpu/GpuNerfRenderer'
import { makeRng } from '../src/nerf/random'
import { buildSyntheticDataset } from '../src/nerf/syntheticScene'
import { NerfTrainer } from '../src/nerf/trainer'
import { DEFAULT_MODEL_CONFIG, DEFAULT_TRAIN_CONFIG } from '../src/nerf/types'
import { paintCanvas } from '../src/utils/image'

const SIZE = 48
const SAMPLES = 24
const FOV = 42
const RADIUS = 3.6
const AZIMUTH = 0.9
const ELEVATION = 0.35

export interface ParityResult {
  meanAbs: number
  maxAbs: number
  /** Share of channels within 8/255 of the reference. */
  agreement: number
  size: number
  trained: boolean
  error?: string
}

function compare(cpu: Uint8ClampedArray, gpu: Uint8ClampedArray): Omit<ParityResult, 'size' | 'trained'> {
  let sum = 0
  let max = 0
  let close = 0
  let count = 0
  for (let i = 0; i < cpu.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(cpu[i + c] - gpu[i + c])
      sum += d
      if (d > max) max = d
      if (d <= 8) close++
      count++
    }
  }
  return { meanAbs: sum / count, maxAbs: max, agreement: close / count }
}

/**
 * @param trained When false the model is randomised instead of trained. Random
 *                weights make the comparison far more discriminative — an
 *                untrained field is nearly uniform, and almost any bug would
 *                still look "close enough".
 */
async function runCase(trained: boolean): Promise<ParityResult> {
  const dataset = buildSyntheticDataset({
    viewCount: 8, resolution: 32, radius: RADIUS, fovDegrees: FOV,
  })
  const trainer = new NerfTrainer(dataset, DEFAULT_MODEL_CONFIG, {
    ...DEFAULT_TRAIN_CONFIG,
    raysPerStep: 256,
    samplesPerRay: SAMPLES,
    // Pruning off: it is a shared input to both paths, and leaving it out keeps
    // this test about the shader maths.
    occupancyRefreshInterval: 0,
  })

  if (trained) {
    for (let i = 0; i < 150; i++) trainer.trainStep()
  } else {
    const rng = makeRng(9)
    const grid = trainer.field.grid.params
    for (let i = 0; i < grid.length; i++) grid[i] = (rng() * 2 - 1) * 0.5
    for (const layer of trainer.field.modules) {
      for (let i = 0; i < layer.bias.length; i++) layer.bias[i] = rng() * 0.4 - 0.2
    }
  }

  const pose = orbitPose(AZIMUTH, ELEVATION, RADIUS)
  const cpu = trainer.render(pose, defaultIntrinsics(SIZE, SIZE, FOV), {
    width: SIZE, height: SIZE, samplesPerRay: SAMPLES,
  })
  paintCanvas(document.getElementById('cpu') as HTMLCanvasElement, cpu.rgba, SIZE, SIZE)

  const canvas = document.getElementById('gpu') as HTMLCanvasElement
  const renderer = new GpuNerfRenderer(canvas, { fovDegrees: FOV, radius: RADIUS })
  try {
    // Damping would interpolate the camera over several frames; this harness
    // renders exactly one.
    renderer.controls.enableDamping = false
    renderer.setSnapshot(trainer.exportGpuSnapshot())
    renderer.setSamples(SAMPLES)
    renderer.setSize(SIZE, SIZE, 1)
    renderer.setOrbit(AZIMUTH, ELEVATION, RADIUS)
    renderer.render()
    const gpu = renderer.readPixels()
    if (gpu.width !== SIZE || gpu.height !== SIZE) {
      return {
        ...compare(cpu.rgba, cpu.rgba), size: SIZE, trained,
        error: `GPU tamponu ${gpu.width}x${gpu.height}, ${SIZE}x${SIZE} bekleniyordu`,
      }
    }
    return { ...compare(cpu.rgba, gpu.rgba), size: SIZE, trained }
  } finally {
    renderer.dispose()
  }
}

async function main(): Promise<void> {
  const output = document.getElementById('result') as HTMLElement
  const results: Record<string, ParityResult> = {}
  try {
    results.random = await runCase(false)
    results.trained = await runCase(true)
  } catch (error) {
    output.textContent = String(error)
    ;(window as unknown as Record<string, unknown>).__parity = { error: String(error) }
    return
  }
  output.textContent = JSON.stringify(results, null, 2)
  ;(window as unknown as Record<string, unknown>).__parity = results
}

void main()
