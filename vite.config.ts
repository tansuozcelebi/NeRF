import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './',
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2020',
    rollupOptions: {
      output: {
        // Three.js is big and changes rarely; keeping it in its own chunk means
        // app edits do not invalidate it in the browser cache.
        manualChunks: {
          three: ['three', 'three/examples/jsm/controls/OrbitControls.js'],
        },
      },
    },
  },
})
