/**
 * GLSL source for evaluating a trained NeRF entirely on the GPU.
 *
 * The fragment shader reimplements, per pixel, exactly what `trainer.render`
 * does on the CPU: intersect the scene box, march stratified samples, skip
 * empty cells with the occupancy grid, look the position up in the hash grid,
 * run both MLPs and alpha-composite front to back.
 *
 * The network shape (level count, resolutions, layer widths) is baked into the
 * source as constants so every loop can be unrolled and every array has a fixed
 * size — GLSL ES 3.0 needs that, and it is also much faster than reading the
 * shape from uniforms.
 *
 * Anything that changes here must stay bit-compatible with the CPU
 * implementation in `hashGrid.ts` / `field.ts`, otherwise the GPU preview stops
 * agreeing with what was actually trained. `gpuParity` in the browser test
 * suite is what keeps the two honest.
 */
import { PRIME_Y, PRIME_Z, type GridLayout } from '../nerf/hashGrid'
import type { ModelConfig } from '../nerf/types'
import { shDim } from '../nerf/sphericalHarmonics'

export interface ShaderLayout {
  /** Width of the RG32F texture holding the hash-grid features. */
  gridTextureWidth: number
  gridTextureHeight: number
  /** Width of the R32F texture holding all MLP weights. */
  weightTextureWidth: number
  weightTextureHeight: number
  /** Flat offsets of each layer inside the weight texture. */
  layerOffsets: number[]
  weightCount: number
}

export const VERTEX_SHADER = /* glsl */ `
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`

/** Texture widths are capped so we never exceed MAX_TEXTURE_SIZE on small GPUs. */
const GRID_TEXTURE_WIDTH = 2048
const WEIGHT_TEXTURE_WIDTH = 512

export function computeShaderLayout(config: ModelConfig, grid: GridLayout): ShaderLayout {
  const hidden = config.hiddenSize
  const geo = config.geoFeatureSize
  const encDim = grid.levels * grid.featuresPerLevel
  const colorInDim = geo + shDim(config.shDegree)

  const sizes = [
    encDim * hidden + hidden, // densityHidden
    hidden * (1 + geo) + (1 + geo), // densityOut
    colorInDim * hidden + hidden, // colorHidden
    hidden * 3 + 3, // colorOut
  ]
  const layerOffsets: number[] = []
  let offset = 0
  for (const size of sizes) {
    layerOffsets.push(offset)
    offset += size
  }

  return {
    gridTextureWidth: GRID_TEXTURE_WIDTH,
    gridTextureHeight: Math.ceil(grid.entryCount / GRID_TEXTURE_WIDTH),
    weightTextureWidth: WEIGHT_TEXTURE_WIDTH,
    weightTextureHeight: Math.ceil(offset / WEIGHT_TEXTURE_WIDTH),
    layerOffsets,
    weightCount: offset,
  }
}

function intArray(name: string, values: number[]): string {
  return `const int ${name}[${values.length}] = int[${values.length}](${values.join(', ')});`
}

/** Builds the fragment shader for one particular model configuration. */
export function buildFragmentShader(
  config: ModelConfig,
  grid: GridLayout,
  layout: ShaderLayout,
): string {
  const hidden = config.hiddenSize
  const geo = config.geoFeatureSize
  const encDim = grid.levels * grid.featuresPerLevel
  const shCount = shDim(config.shDegree)
  const colorInDim = geo + shCount
  const densityOutDim = 1 + geo
  const [densityHiddenOff, densityOutOff, colorHiddenOff, colorOutOff] = layout.layerOffsets

  return /* glsl */ `
precision highp float;
precision highp int;
precision highp sampler2D;

in vec2 vUv;
out vec4 fragColor;

// ---- baked model shape ------------------------------------------------------
#define LEVELS ${grid.levels}
#define FEATURES ${grid.featuresPerLevel}
#define ENC_DIM ${encDim}
#define HIDDEN ${hidden}
#define GEO_DIM ${geo}
#define DENSITY_OUT_DIM ${densityOutDim}
#define SH_DIM ${shCount}
#define COLOR_IN_DIM ${colorInDim}
#define GRID_TEX_W ${layout.gridTextureWidth}
#define W_TEX_W ${layout.weightTextureWidth}
#define TABLE_MASK ${grid.tableMask}u
#define DENSITY_HIDDEN_OFF ${densityHiddenOff}
#define DENSITY_OUT_OFF ${densityOutOff}
#define COLOR_HIDDEN_OFF ${colorHiddenOff}
#define COLOR_OUT_OFF ${colorOutOff}
#define RAW_DENSITY_CLAMP 15.0

${intArray('RESOLUTIONS', grid.resolutions)}
${intArray('LEVEL_OFFSETS', grid.levelOffsets)}
${intArray('DENSE_SIDES', grid.denseSides)}

// ---- inputs -----------------------------------------------------------------
uniform sampler2D uGrid;        // RG32F: hash-grid features
uniform sampler2D uWeights;     // R32F: every MLP weight and bias
uniform sampler2D uOccupancy;   // R8: coarse empty-space mask, flattened
uniform int uOccupancyRes;
uniform bool uUseOccupancy;

uniform vec3 uCameraPosition;
uniform mat3 uCameraBasis;      // camera -> world rotation
uniform float uTanHalfFovY;
uniform float uAspect;

uniform float uAabb;            // half-size of the scene box
uniform vec3 uBackground;
uniform int uSamples;           // samples per ray
uniform int uMode;              // 0 = colour, 1 = depth
uniform float uDepthNear;
uniform float uDepthFar;

// ---- helpers ----------------------------------------------------------------
float weightAt(int index) {
  return texelFetch(uWeights, ivec2(index % W_TEX_W, index / W_TEX_W), 0).r;
}

vec2 gridFeature(int entry) {
  return texelFetch(uGrid, ivec2(entry % GRID_TEX_W, entry / GRID_TEX_W), 0).rg;
}

bool occupiedAt(vec3 p) {
  if (!uUseOccupancy) return true;
  int res = uOccupancyRes;
  ivec3 c = clamp(ivec3(p * float(res)), ivec3(0), ivec3(res - 1));
  int idx = (c.z * res + c.y) * res + c.x;
  int width = res * res;
  return texelFetch(uOccupancy, ivec2(idx % width, idx / width), 0).r > 0.5;
}

// Mirrors HashGrid.vertexIndex: dense addressing for the coarse levels that fit
// in the table, a spatial hash for the rest.
int vertexIndex(int level, ivec3 c) {
  int side = DENSE_SIDES[level];
  if (side > 0) {
    return LEVEL_OFFSETS[level] + c.x + side * (c.y + side * c.z);
  }
  uint h = uint(c.x) ^ (uint(c.y) * ${PRIME_Y}u) ^ (uint(c.z) * ${PRIME_Z}u);
  return LEVEL_OFFSETS[level] + int(h & TABLE_MASK);
}

void encodePosition(vec3 p, out float enc[ENC_DIM]) {
  for (int l = 0; l < LEVELS; l++) {
    int res = RESOLUTIONS[l];
    vec3 g = p * float(res);
    ivec3 c0 = clamp(ivec3(floor(g)), ivec3(0), ivec3(res - 1));
    vec3 f = g - vec3(c0);

    vec2 acc = vec2(0.0);
    for (int corner = 0; corner < 8; corner++) {
      ivec3 o = ivec3(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
      vec3 wv = mix(vec3(1.0) - f, f, vec3(o));
      float w = wv.x * wv.y * wv.z;
      if (w == 0.0) continue;
      acc += w * gridFeature(vertexIndex(l, c0 + o));
    }
    enc[l * FEATURES] = acc.x;
    enc[l * FEATURES + 1] = acc.y;
  }
}

// Real spherical harmonics, same basis and ordering as sphericalHarmonics.ts.
void encodeDirection(vec3 d, out float sh[SH_DIM]) {
  sh[0] = 0.28209479177387814;
#if SH_DIM > 1
  sh[1] = -0.48860251190291987 * d.y;
  sh[2] = 0.48860251190291987 * d.z;
  sh[3] = -0.48860251190291987 * d.x;
#endif
#if SH_DIM > 4
  float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;
  float xy = d.x * d.y, yz = d.y * d.z, xz = d.x * d.z;
  sh[4] = 1.0925484305920792 * xy;
  sh[5] = -1.0925484305920792 * yz;
  sh[6] = 0.94617469575755997 * zz - 0.31539156525251999;
  sh[7] = -1.0925484305920792 * xz;
  sh[8] = 0.54627421529603959 * (xx - yy);
#endif
#if SH_DIM > 9
  sh[9] = 0.59004358992664352 * d.y * (-3.0 * xx + yy);
  sh[10] = 2.8906114426405538 * xy * d.z;
  sh[11] = 0.45704579946446572 * d.y * (1.0 - 5.0 * zz);
  sh[12] = 0.3731763325901154 * d.z * (5.0 * zz - 3.0);
  sh[13] = 0.45704579946446572 * d.x * (1.0 - 5.0 * zz);
  sh[14] = 1.4453057213202769 * d.z * (xx - yy);
  sh[15] = 0.59004358992664352 * d.x * (xx - 3.0 * yy);
#endif
}

/** Evaluates density and view-dependent colour for one sample. */
void evaluate(vec3 pos, vec3 dir, out float sigma, out vec3 rgb) {
  float enc[ENC_DIM];
  encodePosition(pos, enc);

  // densityHidden: ENC_DIM -> HIDDEN, ReLU.
  float h1[HIDDEN];
  for (int o = 0; o < HIDDEN; o++) {
    float acc = weightAt(DENSITY_HIDDEN_OFF + ENC_DIM * HIDDEN + o);
    int base = DENSITY_HIDDEN_OFF + o * ENC_DIM;
    for (int i = 0; i < ENC_DIM; i++) acc += weightAt(base + i) * enc[i];
    h1[o] = max(acc, 0.0);
  }

  // densityOut: HIDDEN -> 1 + GEO_DIM.
  float dout[DENSITY_OUT_DIM];
  for (int o = 0; o < DENSITY_OUT_DIM; o++) {
    float acc = weightAt(DENSITY_OUT_OFF + HIDDEN * DENSITY_OUT_DIM + o);
    int base = DENSITY_OUT_OFF + o * HIDDEN;
    for (int i = 0; i < HIDDEN; i++) acc += weightAt(base + i) * h1[i];
    dout[o] = acc;
  }
  sigma = exp(clamp(dout[0], -RAW_DENSITY_CLAMP, RAW_DENSITY_CLAMP));

  float sh[SH_DIM];
  encodeDirection(dir, sh);
  float colorIn[COLOR_IN_DIM];
  for (int k = 0; k < GEO_DIM; k++) colorIn[k] = dout[1 + k];
  for (int k = 0; k < SH_DIM; k++) colorIn[GEO_DIM + k] = sh[k];

  // colorHidden: COLOR_IN_DIM -> HIDDEN, ReLU.
  float c1[HIDDEN];
  for (int o = 0; o < HIDDEN; o++) {
    float acc = weightAt(COLOR_HIDDEN_OFF + COLOR_IN_DIM * HIDDEN + o);
    int base = COLOR_HIDDEN_OFF + o * COLOR_IN_DIM;
    for (int i = 0; i < COLOR_IN_DIM; i++) acc += weightAt(base + i) * colorIn[i];
    c1[o] = max(acc, 0.0);
  }

  // colorOut: HIDDEN -> 3, sigmoid.
  vec3 raw;
  for (int o = 0; o < 3; o++) {
    float acc = weightAt(COLOR_OUT_OFF + HIDDEN * 3 + o);
    int base = COLOR_OUT_OFF + o * HIDDEN;
    for (int i = 0; i < HIDDEN; i++) acc += weightAt(base + i) * c1[i];
    raw[o] = acc;
  }
  rgb = 1.0 / (1.0 + exp(-raw));
}

/** Slab test against the box [-uAabb, uAabb]^3. */
bool intersectBox(vec3 o, vec3 d, out float tNear, out float tFar) {
  vec3 inv = 1.0 / d;
  vec3 t0 = (vec3(-uAabb) - o) * inv;
  vec3 t1 = (vec3(uAabb) - o) * inv;
  vec3 lo = min(t0, t1);
  vec3 hi = max(t0, t1);
  tNear = max(max(lo.x, lo.y), lo.z);
  tFar = min(min(hi.x, hi.y), hi.z);
  tNear = max(tNear, 0.0);
  return tFar > tNear;
}

/** Near-is-warm ramp, matching the CPU depth image. */
vec3 depthRamp(float t) {
  return clamp(vec3(1.5 * t - 0.4, 1.4 * t - 0.1, 1.2 - 1.4 * t), 0.0, 1.0);
}

void main() {
  vec2 ndc = vUv * 2.0 - 1.0;
  vec3 dirCam = vec3(ndc.x * uTanHalfFovY * uAspect, ndc.y * uTanHalfFovY, -1.0);
  vec3 dir = normalize(uCameraBasis * dirCam);
  vec3 origin = uCameraPosition;

  float tNear, tFar;
  if (!intersectBox(origin, dir, tNear, tFar)) {
    fragColor = vec4(uMode == 1 ? vec3(0.047, 0.031, 0.039) : uBackground, 1.0);
    return;
  }

  float step = (tFar - tNear) / float(uSamples);
  float transmittance = 1.0;
  vec3 accum = vec3(0.0);
  float depth = 0.0;
  float invExtent = 1.0 / (2.0 * uAabb);

  for (int i = 0; i < uSamples; i++) {
    if (transmittance < 1e-4) break;
    // Mid-stratum sampling: deterministic, so frames do not shimmer.
    float t = tNear + (float(i) + 0.5) * step;
    vec3 p = (origin + t * dir + vec3(uAabb)) * invExtent;
    if (!occupiedAt(p)) continue;

    float sigma;
    vec3 rgb;
    evaluate(p, dir, sigma, rgb);

    float alpha = 1.0 - exp(-sigma * step);
    float w = transmittance * alpha;
    accum += w * rgb;
    depth += w * t;
    transmittance -= w;
  }

  if (uMode == 1) {
    float opacity = 1.0 - transmittance;
    if (opacity < 0.1) {
      fragColor = vec4(0.047, 0.055, 0.086, 1.0);
      return;
    }
    float d = depth / max(opacity, 1e-6);
    float norm = 1.0 - clamp((d - uDepthNear) / max(uDepthFar - uDepthNear, 1e-6), 0.0, 1.0);
    fragColor = vec4(depthRamp(norm), 1.0);
    return;
  }

  fragColor = vec4(accum + transmittance * uBackground, 1.0);
}
`
}
