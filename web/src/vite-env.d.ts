/// <reference types="vite/client" />

// Vite 构建期从 config/server_config.json 注入的后端地址（Tauri 模式使用）
declare const __SERVER_HOST__: string
declare const __SERVER_PORT__: number
