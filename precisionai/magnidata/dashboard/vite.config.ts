import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

const apiProxyTarget = process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:5051'

export default defineConfig({
  plugins: [react()],
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
  server: {
    port: 5174,
    proxy: {
      '/api': { target: apiProxyTarget, changeOrigin: true },
    },
  },
})
