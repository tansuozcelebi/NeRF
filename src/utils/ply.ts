/**
 * Writing a coloured triangle mesh as a PLY file.
 *
 * PLY rather than OBJ because per-vertex colour is part of the format rather
 * than a vendor extension, and the mesh coming out of a NeRF carries colour but
 * no texture coordinates — there is no UV unwrap to speak of. Blender, MeshLab,
 * CloudCompare and Houdini all read this directly.
 *
 * The body is binary: a couple of hundred thousand triangles in ASCII would be
 * tens of megabytes of text for no benefit.
 */

export interface PlyMesh {
  positions: Float32Array
  colors: Uint8Array
  indices: Uint32Array
  vertexCount: number
  triangleCount: number
}

export function encodePly(mesh: PlyMesh, comment: string): Blob {
  const header =
    'ply\n' +
    'format binary_little_endian 1.0\n' +
    `comment ${comment.replace(/\n/g, ' ')}\n` +
    `element vertex ${mesh.vertexCount}\n` +
    'property float x\nproperty float y\nproperty float z\n' +
    'property uchar red\nproperty uchar green\nproperty uchar blue\n' +
    `element face ${mesh.triangleCount}\n` +
    'property list uchar uint vertex_indices\n' +
    'end_header\n'

  // 12 bytes of position + 3 of colour per vertex; 1 count byte + 12 of indices
  // per face.
  const body = new ArrayBuffer(mesh.vertexCount * 15 + mesh.triangleCount * 13)
  const view = new DataView(body)
  let offset = 0
  for (let v = 0; v < mesh.vertexCount; v++) {
    view.setFloat32(offset, mesh.positions[v * 3], true)
    view.setFloat32(offset + 4, mesh.positions[v * 3 + 1], true)
    view.setFloat32(offset + 8, mesh.positions[v * 3 + 2], true)
    view.setUint8(offset + 12, mesh.colors[v * 3])
    view.setUint8(offset + 13, mesh.colors[v * 3 + 1])
    view.setUint8(offset + 14, mesh.colors[v * 3 + 2])
    offset += 15
  }
  for (let t = 0; t < mesh.triangleCount; t++) {
    view.setUint8(offset, 3)
    view.setUint32(offset + 1, mesh.indices[t * 3], true)
    view.setUint32(offset + 5, mesh.indices[t * 3 + 1], true)
    view.setUint32(offset + 9, mesh.indices[t * 3 + 2], true)
    offset += 13
  }

  return new Blob([new TextEncoder().encode(header), body], { type: 'application/octet-stream' })
}
