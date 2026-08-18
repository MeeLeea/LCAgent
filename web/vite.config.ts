import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

/**
 * 读取后端 config/server_config.json，统一端口配置源。
 * 文件不存在或解析失败时回退到 127.0.0.1:8000。
 */
function loadServerConfig(): { host: string; port: number } {
  const defaults = { host: '127.0.0.1', port: 8000 }
  try {
    const cfgPath = resolve(__dirname, '..', 'config', 'server_config.json')
    const raw = readFileSync(cfgPath, 'utf-8')
    const data = JSON.parse(raw)
    if (typeof data.host === 'string') defaults.host = data.host
    if (typeof data.port === 'number') defaults.port = data.port
  } catch {
    /* 文件不存在或解析失败，使用默认值 */
  }
  return defaults
}

const serverConfig = loadServerConfig()
const apiTarget = `http://${serverConfig.host}:${serverConfig.port}`

// 开发时把 /api 代理到 FastAPI 后端（端口读自 config/server_config.json）
// Tauri 集成：固定端口、关闭清屏（Tauri CLI 需要稳定可读的终端输出）
export default defineConfig(({ mode }) => {
  // 允许命令行/环境变量覆盖（VITE_API_BASE 仍最高优先）
  loadEnv(mode, process.cwd(), '')

  return {
    plugins: [react()],
    // Tauri 期望相对资源路径，避免 http://tauri.localhost/assets 解析问题
    base: './',
    // 不清屏，便于 Tauri CLI 捕获终端输出
    clearScreen: false,
    // 把后端端口注入前端运行时，供 api.ts 的 Tauri 分支使用
    define: {
      __SERVER_HOST__: JSON.stringify(serverConfig.host),
      __SERVER_PORT__: JSON.stringify(serverConfig.port),
    },
    server: {
      port: 5173,
      strictPort: true,
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          // SSE 流式响应必须禁用缓冲，否则事件会被攒到连接关闭才一次性到达
          configure: (proxy) => {
            proxy.on('proxyRes', (proxyRes) => {
              // 禁用压缩，避免 gzip 缓冲导致 SSE 事件不实时到达
              proxyRes.headers['x-accel-buffering'] = 'no'
              proxyRes.headers['cache-control'] = 'no-cache'
            })
          },
        },
      },
    },
    build: {
      target: 'es2021',
      // Tauri 构建前会执行 beforeBuildCommand，输出到 dist
      outDir: 'dist',
      emptyOutDir: true,
    },
  }
})
