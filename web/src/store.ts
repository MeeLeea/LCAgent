// 全局状态管理（Zustand）
import { create } from 'zustand'
import { api } from './api'
import type {
  ChatMessage,
  InterruptInfo,
  Provider,
  RawMessage,
  ThreadSummary,
  WorkflowInfo,
} from './types'

export type ThemeId = 'dark' | 'deepblue' | 'midnight' | 'forest' | 'light' | 'sepia'

export interface ThemeMeta {
  id: ThemeId
  name: string
  /** 预览色块（主背景色） */
  swatch: string
  /** 预览色块（次背景色，体现层次） */
  swatchAlt: string
  /** 强调色 */
  accent: string
  isDark: boolean
}

/** 全部可选主题（不同背景色） */
export const THEMES: ThemeMeta[] = [
  { id: 'dark', name: '极夜黑', swatch: '#0d0d0d', swatchAlt: '#171717', accent: '#00d96a', isDark: true },
  { id: 'deepblue', name: '深空蓝', swatch: '#0a1628', swatchAlt: '#122339', accent: '#3b82f6', isDark: true },
  { id: 'midnight', name: '午夜紫', swatch: '#1a1430', swatchAlt: '#251c44', accent: '#a78bfa', isDark: true },
  { id: 'forest', name: '森野绿', swatch: '#0d1a14', swatchAlt: '#15281f', accent: '#34d399', isDark: true },
  { id: 'sepia', name: '护眼米', swatch: '#f5ecd9', swatchAlt: '#ece1c8', accent: '#b8860b', isDark: false },
  { id: 'light', name: '皓月白', swatch: '#ffffff', swatchAlt: '#f7f7f8', accent: '#00b85a', isDark: false },
]

/** 判断会话是否属于工作流类型（兼容后端 type 缺失时按 thread_id 前缀兜底） */
export function isWorkflowThread(t: { thread_id: string; type?: 'chat' | 'workflow' }): boolean {
  return t.type === 'workflow' || t.thread_id.includes('workflow')
}

interface AppState {
  // 主题
  theme: ThemeId
  setTheme: (id: ThemeId) => void

  // 视图模式（对话 / 工作流）
  viewMode: 'chat' | 'workflow'
  setViewMode: (mode: 'chat' | 'workflow') => void
  /** 切换视图模式并同步选中对应模式的会话 */
  switchViewMode: (mode: 'chat' | 'workflow') => Promise<void>

  // 数据
  threads: ThreadSummary[]
  currentThreadId: string | null
  messages: ChatMessage[]
  providers: Provider[]
  currentProvider: string | null
  currentModel: string | null
  tools: string[]

  // 工作流
  workflows: string[]
  workflow: WorkflowInfo | null
  workflowLoading: boolean
  workflowError: string | null
  fetchWorkflows: () => Promise<void>
  fetchWorkflow: (name?: string, force?: boolean) => Promise<void>

  // 流式状态
  isStreaming: boolean
  pendingInterrupt: InterruptInfo | null

  // 连接状态
  connectionStatus: 'connected' | 'disconnected' | 'checking'
  checkConnection: () => Promise<void>

  // 初始化与元数据
  init: () => Promise<void>
  fetchThreads: () => Promise<void>
  refreshProviders: () => Promise<void>

  // 会话操作
  selectThread: (id: string) => Promise<void>
  newThread: () => Promise<void>
  newWorkflowThread: (workflowName: string) => Promise<void>
  deleteThread: (id: string) => Promise<void>

  // 聊天
  sendMessage: (text: string) => void
  /** 重新生成最后一条 assistant 回复：截断到上一条 user 消息后重发 */
  regenerate: () => void
  /** 编辑某条 user 消息后重发：截断到该消息（含）并重新发送 */
  editAndResend: (index: number, newText: string) => void
  resume: (payload: Record<string, unknown>) => void
  stopStreaming: () => void
  /** 清空当前会话的前端消息显示（不删后端历史） */
  clearMessages: () => void

  // 提供商/模型
  switchProvider: (key: string) => Promise<void>
  switchModel: (model: string) => Promise<void>
}

let abortFn: (() => void) | null = null
let msgSeq = 0
const nextId = () => `m${++msgSeq}`

/** 看门狗：N 秒内无任何事件则强制复位 isStreaming，防止流卡死 */
let watchdogTimer: ReturnType<typeof setTimeout> | null = null
const WATCHDOG_MS = 90_000

function armWatchdog(onTimeout: () => void) {
  clearWatchdog()
  watchdogTimer = setTimeout(onTimeout, WATCHDOG_MS)
}
function clearWatchdog() {
  if (watchdogTimer) {
    clearTimeout(watchdogTimer)
    watchdogTimer = null
  }
}

/**
 * 构造看门狗超时处理器：把最后一条仍在流式中的 assistant 消息标记为超时错误，
 * 并强制复位 isStreaming。所有事件循环中重挂看门狗时都必须复用同一个处理器，
 * 否则一旦换成空函数，流卡死时 isStreaming 将永久为 true，输入框被锁死。
 */
function makeWatchdogHandler(
  get: () => AppState,
  set: (partial: Partial<AppState>) => void,
  finish: () => void,
) {
  return () => {
    const msgs = [...get().messages]
    const lastIndex = msgs.length - 1
    const last = msgs[lastIndex]
    if (last && last.role === 'assistant' && last.streaming) {
      msgs[lastIndex] = {
        ...last,
        streaming: false,
        error: true,
        content: last.content + (last.content ? '\n\n' : '') + '> ⏱️ 响应超时，请重试。',
      }
      set({ messages: msgs })
    }
    finish()
  }
}

/** 把后端原始消息列表转换为前端回合结构 */
function rawToMessages(raw: RawMessage[]): ChatMessage[] {
  const turns: ChatMessage[] = []
  for (const m of raw) {
    if (m.role === 'user') {
      turns.push({ id: nextId(), role: 'user', content: m.content })
    } else if (m.role === 'assistant') {
      if (m.tool_calls && m.tool_calls.length) {
        turns.push({
          id: nextId(),
          role: 'assistant',
          content: m.content || '',
          toolCalls: m.tool_calls,
          toolResults: [],
        })
      } else {
        const last = turns[turns.length - 1]
        // 工具调用后的最终回答：合并到上一个带工具的 assistant 回合
        if (last && last.role === 'assistant' && last.toolCalls?.length && !last.content) {
          last.content = m.content
        } else {
          turns.push({ id: nextId(), role: 'assistant', content: m.content })
        }
      }
    } else if (m.role === 'tool') {
      const last = turns[turns.length - 1]
      if (last && last.role === 'assistant' && last.toolResults) {
        last.toolResults.push({ id: m.id ?? '', name: m.name ?? '', content: m.content })
      }
    }
  }
  return turns
}

function applyTheme(theme: ThemeId) {
  document.documentElement.setAttribute('data-theme', theme)
  localStorage.setItem('agent-theme', theme)
}

function isValidTheme(v: string | null): v is ThemeId {
  return !!v && THEMES.some((t) => t.id === v)
}

export const useStore = create<AppState>((set, get) => ({
  theme: isValidTheme(localStorage.getItem('agent-theme')) ? (localStorage.getItem('agent-theme') as ThemeId) : 'dark',
  setTheme: (id) => {
    applyTheme(id)
    set({ theme: id })
  },

  threads: [],
  currentThreadId: null,
  messages: [],
  providers: [],
  currentProvider: null,
  currentModel: null,
  tools: [],
  viewMode: 'chat',
  workflows: [],
  workflow: null,
  workflowLoading: false,
  workflowError: null,
  isStreaming: false,
  pendingInterrupt: null,
  connectionStatus: 'checking',

  checkConnection: async () => {
    set({ connectionStatus: 'checking' })
    try {
      await api.health()
      set({ connectionStatus: 'connected' })
    } catch {
      set({ connectionStatus: 'disconnected' })
    }
  },

  setViewMode: (mode) => set({ viewMode: mode }),

  switchViewMode: async (mode) => {
    if (get().isStreaming) return
    set({ viewMode: mode })
    if (mode === 'workflow') void get().fetchWorkflow()
    await get().fetchThreads()
    // 切换到目标模式的第一个会话；若无则清空，避免主区域显示跨模式会话
    const wantWorkflow = mode === 'workflow'
    const first = get().threads.find((t) => isWorkflowThread(t) === wantWorkflow)
    if (first) {
      if (get().currentThreadId !== first.thread_id) {
        await get().selectThread(first.thread_id)
      }
    } else {
      set({ currentThreadId: null, messages: [], pendingInterrupt: null })
    }
  },

  fetchWorkflows: async () => {
    try {
      const r = await api.getWorkflows()
      set({ workflows: r.workflows })
    } catch {
      /* ignore */
    }
  },

  fetchWorkflow: async (name = 'simple', force = false) => {
    // 已加载同名称工作流且非强制刷新时复用缓存，避免重复请求
    if (!force && get().workflow?.name === name) return
    set({ workflowLoading: true, workflowError: null })
    try {
      const info = await api.getWorkflow(name)
      set({ workflow: info, workflowLoading: false })
    } catch (e) {
      set({
        workflowLoading: false,
        workflowError: `获取工作流失败: ${(e as Error).message}`,
      })
    }
  },

  init: async () => {
    applyTheme(get().theme)
    void get().checkConnection()
    await Promise.all([get().refreshProviders(), get().fetchThreads(), get().fetchWorkflows()])
    api.getTools().then((r) => set({ tools: r.tools })).catch(() => {})
  },

  refreshProviders: async () => {
    try {
      const info = await api.getProviders()
      set({
        providers: info.providers,
        currentProvider: info.current_provider,
        currentModel: info.current_model,
      })
    } catch {
      /* ignore */
    }
  },

  fetchThreads: async () => {
    try {
      const r = await api.listThreads()
      set({ threads: r.threads })
      // 没有选中会话时，自动选当前模式的第一个会话（无匹配则退回首个）
      if (!get().currentThreadId && r.threads.length) {
        const wantWorkflow = get().viewMode === 'workflow'
        const first =
          r.threads.find((t) => isWorkflowThread(t) === wantWorkflow) ?? r.threads[0]
        await get().selectThread(first.thread_id)
      }
    } catch {
      /* ignore */
    }
  },

  selectThread: async (id) => {
    if (get().isStreaming) return
    set({ currentThreadId: id, messages: [], pendingInterrupt: null })
    try {
      const r = await api.getMessages(id)
      if (get().currentThreadId === id) {
        set({ messages: rawToMessages(r.messages) })
      }
    } catch {
      /* ignore */
    }
  },

  newThread: async () => {
    if (get().isStreaming) return
    try {
      // 工作流模式下创建专属工作流会话，其余为普通对话
      const type = get().viewMode
      const r = type === 'workflow' ? await api.createThread('workflow', 'simple') : await api.createThread('chat')
      set({ currentThreadId: r.thread_id, messages: [], pendingInterrupt: null })
      await get().fetchThreads()
    } catch (e) {
      console.error(e)
    }
  },

  newWorkflowThread: async (workflowName) => {
    if (get().isStreaming) return
    try {
      const r = await api.createThread('workflow', workflowName)
      set({ currentThreadId: r.thread_id, messages: [], pendingInterrupt: null })
      await get().fetchThreads()
    } catch (e) {
      console.error(e)
    }
  },

  deleteThread: async (id) => {
    try {
      await api.deleteThread(id)
      if (get().currentThreadId === id) {
        set({ currentThreadId: null, messages: [] })
      }
      await get().fetchThreads()
    } catch (e) {
      console.error(e)
    }
  },

  sendMessage: async (text) => {
    const trimmed = text.trim()
    if (!trimmed || get().isStreaming) return
    const threadId = get().currentThreadId

    // 所有消息（包括 / 命令）统一走流式通道
    console.log('[前端] 发送消息:', trimmed.slice(0, 50) + (trimmed.length > 50 ? '...' : ''))

    const now = Date.now()
    const userMsg: ChatMessage = { id: nextId(), role: 'user', content: trimmed, timestamp: now }
    const assistantMsg: ChatMessage = {
      id: nextId(),
      role: 'assistant',
      content: '',
      toolCalls: [],
      toolResults: [],
      streaming: true,
      timestamp: now,
    }
    set((s) => ({
      messages: [...s.messages, userMsg, assistantMsg],
      isStreaming: true,
      pendingInterrupt: null,
    }))

    // 终止标记：避免重复处理终止事件（如 error 后流关闭又补发 done）
    let terminated = false
    const finish = () => {
      if (terminated) return
      terminated = true
      clearWatchdog()
      abortFn = null
      set({ isStreaming: false })
      console.log('[前端] 消息接收完成')
    }

    const watchdogHandler = makeWatchdogHandler(get, set, finish)
    armWatchdog(watchdogHandler)

    abortFn = api.streamChat({ message: trimmed, thread_id: threadId }, (ev) => {
      // 每收到任意事件，重置看门狗（复用同一超时处理器，保持兜底有效）
      if (!terminated) armWatchdog(watchdogHandler)
      const msgs = [...get().messages]
      const lastIndex = msgs.length - 1
      const last = msgs[lastIndex]
      if (!last || last.role !== 'assistant') return

      switch (ev.type) {
        case 'thread_created':
          set({ currentThreadId: ev.thread_id })
          get().fetchThreads()
          break
        case 'token':
          // 创建新对象，触发不可变更新
          msgs[lastIndex] = { ...last, content: last.content + ev.content }
          set({ messages: msgs })
          break
        case 'tool_call':
          console.log('[前端] 工具调用:', ev.name)
          msgs[lastIndex] = { ...last, toolCalls: [...(last.toolCalls ?? []), { id: ev.id, name: ev.name, args: ev.args }] }
          set({ messages: msgs })
          break
        case 'tool_result':
          // 接收 tool_result 标记完成状态，但不保存内容
          msgs[lastIndex] = {
            ...last,
            toolResults: [
              ...(last.toolResults ?? []),
              { id: ev.id, name: ev.name, content: '' }, // 内容置空
            ],
          }
          set({ messages: msgs })
          break
        case 'interrupt':
          msgs[lastIndex] = { ...last, streaming: false, interrupted: true }
          set({ messages: msgs, pendingInterrupt: { prompt: ev.prompt, choices: ev.choices } })
          finish()
          break
        case 'cancelled':
          msgs[lastIndex] = {
            ...last,
            streaming: false,
            content: last.content + (last.content ? '\n\n' : '') + `> ⚠️ ${ev.content}`,
          }
          set({ messages: msgs })
          finish()
          get().fetchThreads()
          break
        case 'error':
          // 避免重复追加多条错误
          if (!last.error) {
            msgs[lastIndex] = {
              ...last,
              streaming: false,
              error: true,
              content: last.content + (last.content ? '\n\n' : '') + `> ❌ ${ev.content}`,
            }
            set({ messages: msgs })
          }
          finish()
          break
        case 'workflow_node': {
          const wf = get().workflow
          if (wf) {
            set({
              workflow: {
                ...wf,
                nodes: wf.nodes.map((n) => (n.id === ev.node ? { ...n, status: ev.status } : n)),
              },
            })
          }
          break
        }
        case 'workflow_status': {
          const wf = get().workflow
          if (wf) set({ workflow: { ...wf, workflow_status: ev.status } })
          break
        }
        case 'done':
          msgs[lastIndex] = { ...last, streaming: false }
          set({ messages: msgs })
          finish()
          get().fetchThreads()
          break
      }
    })
  },

  regenerate: () => {
    if (get().isStreaming) return
    const msgs = [...get().messages]
    if (msgs.length === 0) return
    // 找到最后一条 user 消息
    let lastUserIdx = -1
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === 'user') {
        lastUserIdx = i
        break
      }
    }
    if (lastUserIdx === -1) return
    const lastUserText = msgs[lastUserIdx].content
    // 截断到最后一条 user 消息（不含），然后重新发送
    set({ messages: msgs.slice(0, lastUserIdx) })
    void get().sendMessage(lastUserText)
  },

  editAndResend: (index, newText) => {
    if (get().isStreaming) return
    const msgs = [...get().messages]
    if (index < 0 || index >= msgs.length) return
    if (msgs[index].role !== 'user') return
    // 截断到该 user 消息（不含），重新发送新文本
    set({ messages: msgs.slice(0, index) })
    void get().sendMessage(newText)
  },

  resume: (payload) => {
    if (get().isStreaming) return
    const threadId = get().currentThreadId
    console.log('[前端] 恢复会话')
    const assistantMsg: ChatMessage = {
      id: nextId(),
      role: 'assistant',
      content: '',
      toolCalls: [],
      toolResults: [],
      streaming: true,
      timestamp: Date.now(),
    }
    set((s) => ({ messages: [...s.messages, assistantMsg], isStreaming: true, pendingInterrupt: null }))

    let terminated = false
    const finish = () => {
      if (terminated) return
      terminated = true
      clearWatchdog()
      abortFn = null
      set({ isStreaming: false })
      console.log('[前端] 会话恢复完成')
    }

    const watchdogHandler = makeWatchdogHandler(get, set, finish)
    armWatchdog(watchdogHandler)

    abortFn = api.streamResume({ payload, thread_id: threadId }, (ev) => {
      if (!terminated) armWatchdog(watchdogHandler)
      const msgs = [...get().messages]
      const lastIndex = msgs.length - 1
      const last = msgs[lastIndex]
      if (!last || last.role !== 'assistant') return
      switch (ev.type) {
        case 'token':
          msgs[lastIndex] = { ...last, content: last.content + ev.content }
          set({ messages: msgs })
          break
        case 'tool_call':
          console.log('[前端] 工具调用:', ev.name)
          msgs[lastIndex] = { ...last, toolCalls: [...(last.toolCalls ?? []), { id: ev.id, name: ev.name, args: ev.args }] }
          set({ messages: msgs })
          break
        case 'tool_result':
          // 接收 tool_result 标记完成状态，但不保存内容
          msgs[lastIndex] = {
            ...last,
            toolResults: [
              ...(last.toolResults ?? []),
              { id: ev.id, name: ev.name, content: '' }, // 内容置空
            ],
          }
          set({ messages: msgs })
          break
        case 'interrupt':
          msgs[lastIndex] = { ...last, streaming: false, interrupted: true }
          set({ messages: msgs, pendingInterrupt: { prompt: ev.prompt, choices: ev.choices } })
          finish()
          break
        case 'cancelled':
          msgs[lastIndex] = {
            ...last,
            streaming: false,
            content: last.content + (last.content ? '\n\n' : '') + `> ⚠️ ${ev.content}`,
          }
          set({ messages: msgs })
          finish()
          break
        case 'error':
          if (!last.error) {
            msgs[lastIndex] = {
              ...last,
              streaming: false,
              error: true,
              content: last.content + (last.content ? '\n\n' : '') + `> ❌ ${ev.content}`,
            }
            set({ messages: msgs })
          }
          finish()
          break
        case 'workflow_node': {
          const wf = get().workflow
          if (wf) {
            set({
              workflow: {
                ...wf,
                nodes: wf.nodes.map((n) => (n.id === ev.node ? { ...n, status: ev.status } : n)),
              },
            })
          }
          break
        }
        case 'workflow_status': {
          const wf = get().workflow
          if (wf) set({ workflow: { ...wf, workflow_status: ev.status } })
          break
        }
        case 'done':
          msgs[lastIndex] = { ...last, streaming: false }
          set({ messages: msgs })
          finish()
          get().fetchThreads()
          break
      }
    })
  },

  stopStreaming: () => {
    clearWatchdog()
    if (abortFn) abortFn()
    abortFn = null
    set((s) => {
      const msgs = [...s.messages]
      const lastIndex = msgs.length - 1
      const last = msgs[lastIndex]
      if (last && last.role === 'assistant' && last.streaming) {
        msgs[lastIndex] = {
          ...last,
          streaming: false,
          content: last.content || '_(已中止)_',
        }
      }
      return { messages: msgs, isStreaming: false }
    })
  },

  clearMessages: () => {
    if (get().isStreaming) return
    set({ messages: [], pendingInterrupt: null })
  },

  switchProvider: async (key) => {
    console.log('[前端] 切换提供商:', key)
    try {
      await api.switchProvider(key)
      await get().refreshProviders()
      console.log('[前端] 提供商切换完成')
    } catch (e) {
      console.error('[前端] 切换提供商失败:', e)
    }
  },

  switchModel: async (model) => {
    console.log('[前端] 切换模型:', model)
    try {
      await api.switchModel(model)
      await get().refreshProviders()
      console.log('[前端] 模型切换完成')
    } catch (e) {
      console.error('[前端] 切换模型失败:', e)
    }
  },
}))
