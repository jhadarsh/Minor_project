import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig(() => {
  const target = process.env.BACKEND_URL || 'http://127.0.0.1:8000'
  const proxy = { '/v1': target, '/health': target }
  return {
    plugins: [react()],
    server: { proxy },
    preview: { proxy },
  }
})
