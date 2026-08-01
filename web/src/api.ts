// API 客户端：REST 请求 + SSE 流式解析
import type { ProvidersInfo, ThreadSummary, RawMessage, StreamEvent } from './types'

// Vite 构建期从 config/server_config.json 注入的后端地址（Tauri 模式使用）
declare const __SERVER_HOST__: string
declare const __SERVER_PORT__: number

/**
 * API 基址：
 *  - 显式环境变量 VITE_API_BASE 优先（构建期注入）
 *  - 否则检测 Tauri 环境（加载本地文件时 fetch('/api') 无法到达后端，必须用绝对 URL）
 *  - 普通 web 模式用相对路径 '/api'（由 Vite 代理或 FastAPI 同源托管）
 */
function detectBase(): string {
  const env = import.meta.env.VITE_API_BASE
  if (env) return env.replace(/\/$/, '')
  // Tauri 2 运行时会在 window 上注入 __TAURI_INTERNALS__
  const g = globalThis as unknown as { __TAURI_INTERNALS__?: unknown }
  if (g.__TAURI_INTERNALS__) return `http://${__SERVER_HOST__}:${__SERVER_PORT__}/api`
  return '/api'
}

const BASE = detectBase()

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.json()).detail ?? ''
    } catch {
      /* ignore */
    }
    throw new Error(`${res.status} ${res.statusText} ${detail}`)
  }
  return res.json() as Promise<T>
}

/** 终止性事件：收到这些事件后流一定结束，前端据此复位 isStreaming */
const TERMINAL_TYPES = new Set(['done', 'error', 'cancelled', 'interrupt'])

/**
 * 读取 SSE 流并把每个 data: 事件回调出去。
 * 返回是否在流中收到过终止事件（done/error/cancelled/interrupt）。
 */
async function consumeSSE(
  res: Response,
  onEvent: (ev: StreamEvent) => void,
): Promise<boolean> {
  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let gotTerminal = false
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // SSE 以空行分隔事件
    let idx: number
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      const line = raw.split('\n').find((l) => l.startsWith('data:'))
      if (!line) continue
      const payload = line.slice(5).trim()
      if (!payload) continue
      try {
        const ev = JSON.parse(payload) as StreamEvent
        if (TERMINAL_TYPES.has(ev.type)) gotTerminal = true
        onEvent(ev)
      } catch {
        // 事件损坏：不能静默吞掉，否则流卡死时 isStreaming 无法复位
        gotTerminal = true
        onEvent({ type: 'error', content: '服务端事件解析失败，连接可能已损坏。' })
      }
    }
  }
  return gotTerminal
}

/**
 * 发起一次 SSE 流式请求。
 * 保证：无论流如何结束，onEvent 至少会收到一个终止事件（done/error）。
 * 返回 abort 函数（用户主动中止时调用）。
 */
function streamRequest(
  url: string,
  body: Record<string, unknown>,
  onEvent: (ev: StreamEvent) => void,
): () => void {
  const controller = new AbortController()
  // 用户主动 abort 标记：避免 abort 后再补发 error
  let abortedByUser = false

  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok || !res.body) {
        onEvent({ type: 'error', content: `请求失败: ${res.status} ${res.statusText}` })
        return
      }
      let gotTerminal = false
      try {
        gotTerminal = await consumeSSE(res, onEvent)
      } catch (err) {
        // 流读取中途异常（连接被中断等）
        if (!abortedByUser) {
          onEvent({ type: 'error', content: `连接中断: ${(err as Error).message}` })
        }
        return
      }
      // 流正常结束但没收到终止事件 → 补发 done，确保 isStreaming 复位
      if (!gotTerminal && !abortedByUser) {
        onEvent({ type: 'done' })
      }
    })
    .catch((err) => {
      if (abortedByUser || err.name === 'AbortError') return
      onEvent({ type: 'error', content: `网络错误: ${(err as Error).message}` })
    })

  return () => {
    abortedByUser = true
    controller.abort()
  }
}

export const api = {
  health: () => jsonFetch<{ status: string }>(`${BASE}/health`),

  getProviders: () => jsonFetch<ProvidersInfo>(`${BASE}/providers`),
  switchProvider: (provider: string) =>
    jsonFetch(`${BASE}/providers/switch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider }),
    }),
  switchModel: (model: string) =>
    jsonFetch(`${BASE}/models/switch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    }),

  getTools: () => jsonFetch<{ tools: string[] }>(`${BASE}/tools`),

  listThreads: () => jsonFetch<{ threads: ThreadSummary[]; current: string | null }>(`${BASE}/threads`),
  createThread: () =>
    jsonFetch<{ thread_id: string }>(`${BASE}/threads`, { method: 'POST' }),
  deleteThread: (thread_id: string) =>
    jsonFetch(`${BASE}/threads/${encodeURIComponent(thread_id)}`, { method: 'DELETE' }),
  getMessages: (thread_id: string) =>
    jsonFetch<{ thread_id: string; messages: RawMessage[] }>(
      `${BASE}/threads/${encodeURIComponent(thread_id)}/messages`,
    ),

  /**
   * 流式聊天：POST /api/chat，解析 SSE 事件。
   * 返回一个 abort 函数，用于中止请求。
   */
  streamChat: (
    body: { message: string; thread_id?: string | null },
    onEvent: (ev: StreamEvent) => void,
  ): (() => void) => streamRequest(`${BASE}/chat`, body, onEvent),

  streamResume: (
    body: { payload: Record<string, unknown>; thread_id?: string | null },
    onEvent: (ev: StreamEvent) => void,
  ): (() => void) => streamRequest(`${BASE}/chat/resume`, body, onEvent),

  executeCommand: (command: string, thread_id?: string | null) =>
    jsonFetch<{ success: boolean; outcome: string; output: string; thread_id: string }>(
      `${BASE}/command`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command, thread_id }),
      },
    ),
}
