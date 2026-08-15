/**
 * Real-time NeRF rendering on the GPU, via Three.js.
 *
 * The whole scene is one full-screen quad. All the work happens in the fragment
 * shader, which ray-marches the trained field per pixel. Three.js is here for
 * what it is genuinely good at: managing the WebGL context, uploading textures,
 * and giving us a proper orbit camera.
 *
 * The speed difference against the CPU path is not subtle — the CPU renders one
 * ray at a time in JavaScript, the GPU runs a few thousand in parallel.
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { GpuSnapshot } from '../nerf/trainer'
import { buildFragmentShader, computeShaderLayout, VERTEX_SHADER } from './nerfShader'

export type RenderMode = 'color' | 'depth'

export interface GpuViewerOptions {
  /** Horizontal field of view in degrees, matching the training cameras. */
  fovDegrees: number
  /** Initial camera distance. */
  radius: number
}

/** Names that identify a CPU implementation of GL pretending to be a GPU. */
const SOFTWARE_RENDERERS = ['swiftshader', 'llvmpipe', 'software', 'basic render']

/** True when WebGL is being emulated on the CPU, where this shader is far too heavy. */
export function isSoftwareRenderer(gl: WebGL2RenderingContext): boolean {
  const info = gl.getExtension('WEBGL_debug_renderer_info')
  const name = String(
    info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
  ).toLowerCase()
  return SOFTWARE_RENDERERS.some((needle) => name.includes(needle))
}

/** Reports why the GPU path is unavailable, so the UI can say something useful. */
export function describeWebglSupport(): { supported: boolean; reason?: string; software?: boolean } {
  try {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl2')
    if (!gl) return { supported: false, reason: 'Tarayıcınız WebGL2 desteklemiyor.' }
    // Sampling 32-bit float textures is core in WebGL2, but the driver still has
    // to give us enough texture units and a big enough texture.
    const maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number
    if (maxTexture < 2048) {
      return { supported: false, reason: 'Ekran kartı yeterince büyük doku desteklemiyor.' }
    }
    return { supported: true, software: isSoftwareRenderer(gl) }
  } catch (error) {
    return { supported: false, reason: error instanceof Error ? error.message : String(error) }
  }
}

export class GpuNerfRenderer {
  readonly renderer: THREE.WebGLRenderer
  readonly camera: THREE.PerspectiveCamera
  readonly controls: OrbitControls

  private readonly scene = new THREE.Scene()
  private readonly quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
  private readonly geometry = new THREE.PlaneGeometry(2, 2)
  private material: THREE.ShaderMaterial | null = null
  private mesh: THREE.Mesh | null = null

  private gridTexture: THREE.DataTexture | null = null
  private weightTexture: THREE.DataTexture | null = null
  private occupancyTexture: THREE.DataTexture | null = null

  /** Signature of the baked shader, so we only recompile when the shape changes. */
  private shaderKey = ''
  private aabb = 1
  private disposed = false
  private readonly horizontalFov: number

  constructor(canvas: HTMLCanvasElement, options: GpuViewerOptions) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: false,
      powerPreference: 'high-performance',
      alpha: false,
    })
    this.renderer.setClearColor(0x07060a, 1)

    // The app specifies a horizontal field of view (that is what the training
    // intrinsics use), while Three's camera takes a vertical one. They only
    // coincide at aspect 1, so the conversion is redone on every resize.
    this.horizontalFov = options.fovDegrees
    this.camera = new THREE.PerspectiveCamera(options.fovDegrees, 1, 0.01, 100)
    this.camera.position.set(
      options.radius * Math.sin(0.9) * Math.cos(0.35),
      options.radius * Math.sin(0.35),
      options.radius * Math.cos(0.9) * Math.cos(0.35),
    )

    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08
    this.controls.rotateSpeed = 0.8
    this.controls.zoomSpeed = 0.9
    this.controls.panSpeed = 0.6
    this.controls.minDistance = 0.6
    this.controls.maxDistance = 14
    this.controls.target.set(0, 0, 0)
    this.controls.update()
  }

  /** Uploads a trained model, rebuilding the shader if its shape changed. */
  setSnapshot(snapshot: GpuSnapshot): void {
    if (this.disposed) return
    const layout = computeShaderLayout(snapshot.modelConfig, snapshot.gridLayout)
    const key = JSON.stringify({
      grid: snapshot.gridLayout,
      hidden: snapshot.modelConfig.hiddenSize,
      geo: snapshot.modelConfig.geoFeatureSize,
      sh: snapshot.modelConfig.shDegree,
    })

    if (key !== this.shaderKey) {
      this.buildMaterial(snapshot, layout)
      this.shaderKey = key
    }

    this.uploadGrid(snapshot, layout)
    this.uploadWeights(snapshot, layout)
    this.uploadOccupancy(snapshot)

    this.aabb = snapshot.aabbSize
    const uniforms = this.material!.uniforms
    uniforms.uAabb.value = snapshot.aabbSize
    uniforms.uBackground.value.setRGB(...snapshot.background)
    uniforms.uUseOccupancy.value = snapshot.useOccupancy
    uniforms.uOccupancyRes.value = snapshot.occupancyResolution
  }

  private buildMaterial(snapshot: GpuSnapshot, layout: ReturnType<typeof computeShaderLayout>): void {
    this.material?.dispose()
    if (this.mesh) this.scene.remove(this.mesh)

    this.material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERTEX_SHADER,
      fragmentShader: buildFragmentShader(snapshot.modelConfig, snapshot.gridLayout, layout),
      depthTest: false,
      depthWrite: false,
      uniforms: {
        uGrid: { value: null },
        uWeights: { value: null },
        uOccupancy: { value: null },
        uOccupancyRes: { value: snapshot.occupancyResolution },
        uUseOccupancy: { value: snapshot.useOccupancy },
        uCameraPosition: { value: new THREE.Vector3() },
        uCameraBasis: { value: new THREE.Matrix3() },
        uTanHalfFovY: { value: 0.5 },
        uAspect: { value: 1 },
        uAabb: { value: snapshot.aabbSize },
        uBackground: { value: new THREE.Color(...snapshot.background) },
        uSamples: { value: 48 },
        uMode: { value: 0 },
        uDepthNear: { value: 0 },
        uDepthFar: { value: 1 },
      },
    })

    this.mesh = new THREE.Mesh(this.geometry, this.material)
    // The vertex shader writes clip space directly, so the bounding box Three
    // would cull against is meaningless here.
    this.mesh.frustumCulled = false
    this.scene.add(this.mesh)
  }

  private uploadGrid(snapshot: GpuSnapshot, layout: ReturnType<typeof computeShaderLayout>): void {
    const { gridTextureWidth: w, gridTextureHeight: h } = layout
    const needed = w * h * 2
    if (!this.gridTexture || this.gridTexture.image.width !== w || this.gridTexture.image.height !== h) {
      this.gridTexture?.dispose()
      this.gridTexture = new THREE.DataTexture(
        new Float32Array(needed), w, h, THREE.RGFormat, THREE.FloatType,
      )
      this.gridTexture.minFilter = THREE.NearestFilter
      this.gridTexture.magFilter = THREE.NearestFilter
      this.material!.uniforms.uGrid.value = this.gridTexture
    }
    const data = this.gridTexture.image.data as Float32Array
    data.set(snapshot.gridParams.subarray(0, Math.min(snapshot.gridParams.length, needed)))
    this.gridTexture.needsUpdate = true
  }

  private uploadWeights(snapshot: GpuSnapshot, layout: ReturnType<typeof computeShaderLayout>): void {
    const { weightTextureWidth: w, weightTextureHeight: h } = layout
    const needed = w * h
    if (!this.weightTexture || this.weightTexture.image.width !== w || this.weightTexture.image.height !== h) {
      this.weightTexture?.dispose()
      this.weightTexture = new THREE.DataTexture(
        new Float32Array(needed), w, h, THREE.RedFormat, THREE.FloatType,
      )
      this.weightTexture.minFilter = THREE.NearestFilter
      this.weightTexture.magFilter = THREE.NearestFilter
      this.material!.uniforms.uWeights.value = this.weightTexture
    }
    const data = this.weightTexture.image.data as Float32Array
    data.set(snapshot.layerWeights.subarray(0, Math.min(snapshot.layerWeights.length, needed)))
    this.weightTexture.needsUpdate = true
  }

  private uploadOccupancy(snapshot: GpuSnapshot): void {
    const res = snapshot.occupancyResolution
    const w = res * res
    const h = res
    if (!this.occupancyTexture || this.occupancyTexture.image.width !== w) {
      this.occupancyTexture?.dispose()
      this.occupancyTexture = new THREE.DataTexture(
        new Uint8Array(w * h), w, h, THREE.RedFormat, THREE.UnsignedByteType,
      )
      this.occupancyTexture.minFilter = THREE.NearestFilter
      this.occupancyTexture.magFilter = THREE.NearestFilter
      this.material!.uniforms.uOccupancy.value = this.occupancyTexture
    }
    const data = this.occupancyTexture.image.data as Uint8Array
    for (let i = 0; i < data.length; i++) data[i] = snapshot.occupancy[i] ? 255 : 0
    this.occupancyTexture.needsUpdate = true
  }

  get ready(): boolean {
    return this.material !== null
  }

  setMode(mode: RenderMode): void {
    if (this.material) this.material.uniforms.uMode.value = mode === 'depth' ? 1 : 0
  }

  setSamples(samples: number): void {
    if (this.material) this.material.uniforms.uSamples.value = Math.max(4, Math.round(samples))
  }

  /** Sets the drawing-buffer size; `scale` below 1 renders fewer pixels. */
  setSize(cssWidth: number, cssHeight: number, scale: number): void {
    this.renderer.setPixelRatio(scale)
    this.renderer.setSize(cssWidth, cssHeight, false)
    const aspect = cssWidth / Math.max(cssHeight, 1)
    this.camera.aspect = aspect
    const halfH = Math.tan((this.horizontalFov * Math.PI) / 360)
    this.camera.fov = (2 * Math.atan(halfH / aspect) * 180) / Math.PI
    this.camera.updateProjectionMatrix()
  }

  /** Renders one frame. Returns false when there is nothing loaded yet. */
  render(): boolean {
    if (this.disposed || !this.material) return false
    this.controls.update()
    this.camera.updateMatrixWorld()

    const uniforms = this.material.uniforms
    uniforms.uCameraPosition.value.copy(this.camera.position)
    ;(uniforms.uCameraBasis.value as THREE.Matrix3).setFromMatrix4(this.camera.matrixWorld)
    uniforms.uTanHalfFovY.value = Math.tan((this.camera.fov * Math.PI) / 360)
    uniforms.uAspect.value = this.camera.aspect

    // Normalising depth against the camera's own distance keeps the ramp stable
    // between frames, unlike a per-frame min/max over the image.
    const distance = this.camera.position.distanceTo(this.controls.target)
    uniforms.uDepthNear.value = Math.max(0, distance - this.aabb)
    uniforms.uDepthFar.value = distance + this.aabb

    this.renderer.render(this.scene, this.quadCamera)
    return true
  }

  /** Places the camera on the orbit used elsewhere in the app. */
  setOrbit(azimuth: number, elevation: number, radius: number): void {
    this.camera.position.set(
      radius * Math.cos(elevation) * Math.sin(azimuth),
      radius * Math.sin(elevation),
      radius * Math.cos(elevation) * Math.cos(azimuth),
    )
    this.controls.target.set(0, 0, 0)
    this.controls.update()
  }

  /** Reads the current frame back as RGBA bytes, for export and comparison. */
  readPixels(): { width: number; height: number; rgba: Uint8ClampedArray } {
    const size = new THREE.Vector2()
    this.renderer.getDrawingBufferSize(size)
    const width = size.x
    const height = size.y
    const buffer = new Uint8Array(width * height * 4)
    const gl = this.renderer.getContext()
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, buffer)

    // WebGL reads bottom-up; flip into image order.
    const rgba = new Uint8ClampedArray(width * height * 4)
    const rowBytes = width * 4
    for (let y = 0; y < height; y++) {
      const src = (height - 1 - y) * rowBytes
      rgba.set(buffer.subarray(src, src + rowBytes), y * rowBytes)
    }
    return { width, height, rgba }
  }

  dispose(): void {
    this.disposed = true
    this.controls.dispose()
    this.material?.dispose()
    this.geometry.dispose()
    this.gridTexture?.dispose()
    this.weightTexture?.dispose()
    this.occupancyTexture?.dispose()
    this.renderer.dispose()
  }
}
