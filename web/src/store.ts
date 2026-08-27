// 全局状态管理（Zustand）
import { create } from 'zustand'
import { api } from './api'
import type {
  ChatMessage,
  InterruptInfo,
  Provider,
  RawMessage,
  StreamEvent,
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

/** 从 thread_id 中反解工作流名称（与后端 registry.workflow_name_of 逻辑一致） */
export function workflowNameOf(threadId: string): string | null {
  let body: string
  if (threadId.startsWith('workflow-')) {
    body = threadId.slice('workflow-'.length)
  } else if (threadId.includes('-workflow-')) {
    body = threadId.split('-workflow-', 2)[1]
  } else {
    return null
  }
  const idx = body.lastIndexOf('-thread-')
  return idx >= 0 ? body.slice(0, idx) : body
}

/**
 * 剥离后端写回记忆的 "workflow:" 命令前缀（仅显示层使用，不动原始消息数据，
 * 以保证重新生成/编辑重发时仍携带完整命令）。
 * 例：`workflow:pipline 帮我分析` -> `pipline 帮我分析`
 */
export function stripWorkflowPrefix(text: string): string {
  return text.replace(/^workflow:/, '')
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

  // 团队角色
  roles: string[]
  currentRole: string | null

  // LLM 累计 token 用量（输入栏右下角展示）
  totalTokens: number

  // 工作空间（当前会话绑定的工作目录，输入栏左下角展示）
  workspace: string | null

  // 工作流
  workflows: string[]
  workflow: WorkflowInfo | null
  workflowLoading: boolean
  workflowError: string | null
  fetchWorkflows: () => Promise<void>
  fetchWorkflow: (name?: string, force?: boolean) => Promise<void>

  // 流式状态（per-thread：支持多会话并发流）
  /** 各线程的消息缓冲（含进行中流的半成品 assistant 消息），跨会话切换保留 */
  messagesByThread: Record<string, ChatMessage[]>
  /** 正在流式输出的线程集合 */
  streamingThreads: Record<string, boolean>
  /** 各线程的挂起 HITL 中断 */
  pendingInterrupts: Record<string, InterruptInfo>
  /** 当前线程的挂起中断（展示用，随 selectThread 同步） */
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

  // 团队角色
  fetchRoles: () => Promise<void>
  switchRole: (role: string) => Promise<void>

  // 工作空间
  fetchWorkspace: (thread_id: string) => Promise<void>
  setWorkspace: (path: string) => Promise<boolean>
  clearWorkspace: () => Promise<void>
}

/** 当前线程是否正在流式输出（派生选择器，供组件订阅） */
export const selectIsStreaming = (s: AppState): boolean => {
  return !!s.streamingThreads[normKey(s.currentThreadId)]
}

type StoreSet = (partial: Partial<AppState> | ((s: AppState) => Partial<AppState>)) => void
type StoreGet = () => AppState

/** 统一线程 key：null（尚无会话，等待后端 thread_created）归一化为 '' */
function normKey(threadId: string | null): string {
  return threadId ?? ''
}

// per-thread 流式基础设施：abort / 终止标记 / 看门狗 均按线程隔离
let abortFns: Record<string, (() => void) | null> = {}
let terminatedThreads: Set<string> = new Set()
let msgSeq = 0
const nextId = () => `m${++msgSeq}`

/** 看门狗：N 秒内无任何事件则强制复位该线程的 streaming，防止流卡死 */
let watchdogTimers: Record<string, ReturnType<typeof setTimeout> | null> = {}
const WATCHDOG_MS = 90_000

function armWatchdog(threadId: string, onTimeout: () => void) {
  clearWatchdog(threadId)
  watchdogTimers[threadId] = setTimeout(onTimeout, WATCHDOG_MS)
}
function clearWatchdog(threadId: string) {
  if (watchdogTimers[threadId]) {
    clearTimeout(watchdogTimers[threadId])
    watchdogTimers[threadId] = null
  }
}

/**
 * 构造看门狗超时处理器：把指定线程最后一条仍在流式中的 assistant 消息标记为
 * 超时错误，并强制复位该线程 streaming。所有事件循环中重挂看门狗时都必须复用
 * 同一个处理器，否则一旦换成空函数，流卡死时线程将永久处于 streaming 状态。
 */
function makeWatchdogHandler(
  get: StoreGet,
  set: StoreSet,
  threadId: string,
  finish: () => void,
) {
  return () => {
    const msgs = [...(get().messagesByThread[threadId] ?? [])]
    const lastIndex = msgs.length - 1
    const last = msgs[lastIndex]
    if (last && last.role === 'assistant' && last.streaming) {
      msgs[lastIndex] = {
        ...last,
        streaming: false,
        error: true,
        content: last.content + (last.content ? '\n\n' : '') + '> ⏱️ 响应超时，请重试。',
      }
      commitThreadMessages(get, set, threadId, msgs)
    }
    finish()
  }
}

/** 把某线程的消息数组写入 messagesByThread；若该线程恰为当前线程，同步 messages */
function commitThreadMessages(
  get: StoreGet,
  set: StoreSet,
  threadId: string,
  arr: ChatMessage[],
) {
  set((s) => {
    const messagesByThread = { ...s.messagesByThread, [threadId]: arr }
    const patch: Partial<AppState> = { messagesByThread }
    if (normKey(s.currentThreadId) === threadId) patch.messages = arr
    return patch
  })
}

/** 标记某线程是否处于流式输出状态 */
function markStreaming(get: StoreGet, set: StoreSet, threadId: string, streaming: boolean) {
  set((s) => {
    const streamingThreads = { ...s.streamingThreads }
    if (streaming) streamingThreads[threadId] = true
    else delete streamingThreads[threadId]
    return { streamingThreads }
  })
}

/**
 * 流结束时补齐未匹配到结果的工具调用：若后端 tool_result 事件的 id 缺失/不匹配
 * （tool_call 与 tool_result 无法关联），为剩余 toolCall 添加占位 result，
 * 避免 ToolCallCard 永久停留在"执行中"。
 */
function fillMissingToolResults(msg: ChatMessage): ChatMessage {
  if (!msg.toolCalls || msg.toolCalls.length === 0) return msg
  const results = msg.toolResults ?? []
  const have = new Set(results.map((r) => r.id))
  const missing = msg.toolCalls.filter((c) => c.id && !have.has(c.id))
  if (missing.length === 0) return msg
  return {
    ...msg,
    toolResults: [
      ...results,
      ...missing.map((c) => ({ id: c.id, name: c.name, content: '' })),
    ],
  }
}

/** 构造某线程流的终止处理器（幂等：只执行一次） */
function makeFinish(get: StoreGet, set: StoreSet, threadId: string) {
  return () => {
    if (terminatedThreads.has(threadId)) return
    terminatedThreads.add(threadId)
    clearWatchdog(threadId)
    delete abortFns[threadId]
    markStreaming(get, set, threadId, false)
  }
}

/**
 * 线程重绑定：首条消息无 thread_id 时后端创建新线程并返回 thread_created，
 * 此时把 sentinel key（''）下的全部流式状态迁移到真实 thread_id。
 */
function rebindThread(get: StoreGet, set: StoreSet, oldKey: string, newKey: string) {
  if (oldKey === newKey) return
  const s = get()
  const messagesByThread = { ...s.messagesByThread }
  if (messagesByThread[oldKey] !== undefined) {
    messagesByThread[newKey] = messagesByThread[oldKey]
    delete messagesByThread[oldKey]
  }
  const streamingThreads = { ...s.streamingThreads }
  if (streamingThreads[oldKey] !== undefined) {
    streamingThreads[newKey] = streamingThreads[oldKey]
    delete streamingThreads[oldKey]
  }
  const pendingInterrupts = { ...s.pendingInterrupts }
  if (pendingInterrupts[oldKey] !== undefined) {
    pendingInterrupts[newKey] = pendingInterrupts[oldKey]
    delete pendingInterrupts[oldKey]
  }
  if (abortFns[oldKey]) {
    abortFns[newKey] = abortFns[oldKey]
    delete abortFns[oldKey]
  }
  if (watchdogTimers[oldKey]) {
    watchdogTimers[newKey] = watchdogTimers[oldKey]
    watchdogTimers[oldKey] = null
  }
  if (terminatedThreads.has(oldKey)) {
    terminatedThreads.delete(oldKey)
    terminatedThreads.add(newKey)
  }
  set({
    messagesByThread,
    streamingThreads,
    pendingInterrupts,
    currentThreadId: newKey,
    messages: messagesByThread[newKey] ?? [],
    pendingInterrupt: pendingInterrupts[newKey] ?? null,
  })
}

/**
 * 统一处理某线程的流式事件（sendMessage / resume 共用）。
 * 事件始终写入该线程的消息缓冲；仅当该线程是当前线程时同步到 messages 展示。
 * 返回当前有效的线程 key（thread_created 事件后更新为后端分配的真实 ID）。
 */
function handleStreamEvent(
  get: StoreGet,
  set: StoreSet,
  threadId: string,
  ev: StreamEvent,
  ctx: { finish: () => void },
): string {
  // terminatedThreads 仅阻止终止事件重复处理（done/error/cancelled/interrupt），
  // 不阻止 tool_result 等迟到事件——代理缓冲可能导致 done 先于 tool_result 到达。
  const isTerminalEv = ev.type === 'done' || ev.type === 'error' || ev.type === 'cancelled' || ev.type === 'interrupt'
  if (isTerminalEv && terminatedThreads.has(threadId)) return threadId
  const arr = [...(get().messagesByThread[threadId] ?? [])]
  const lastIndex = arr.length - 1
  const last = arr[lastIndex]
  // 流式操作目标：最后一条 streaming 的 assistant 消息。
  // 流已结束（done 已处理）时回退到最后一条带 toolCalls 的 assistant 消息，
  // 使迟到的 tool_result 仍能配对到正确的回合。
  let streamIdx = lastIndex
  for (let i = lastIndex; i >= 0; i--) {
    if (arr[i].role === 'assistant' && arr[i].streaming) {
      streamIdx = i
      break
    }
  }
  if (!arr[streamIdx]?.streaming) {
    for (let i = lastIndex; i >= 0; i--) {
      if (arr[i].role === 'assistant' && (arr[i].toolCalls?.length || arr[i].content)) {
        streamIdx = i
        break
      }
    }
  }
  const streamLast = arr[streamIdx]

  switch (ev.type) {
    case 'thread_created':
      rebindThread(get, set, threadId, ev.thread_id)
      get().fetchThreads()
      return ev.thread_id
    case 'token':
      if (!streamLast || streamLast.role !== 'assistant') return threadId
      arr[streamIdx] = { ...streamLast, content: streamLast.content + ev.content }
      commitThreadMessages(get, set, threadId, arr)
      break
    case 'tool_call':
      if (!streamLast || streamLast.role !== 'assistant') return threadId
      // ask_human 的 UI 由 INTERRUPT 事件渲染，不添加到 toolCalls 避免重复显示
      if (ev.name === 'ask_human') return threadId
      arr[streamIdx] = {
        ...streamLast,
        toolCalls: [...(streamLast.toolCalls ?? []), { id: ev.id, name: ev.name, args: ev.args }],
      }
      commitThreadMessages(get, set, threadId, arr)
      break
    case 'tool_result':
      if (!streamLast || streamLast.role !== 'assistant') return threadId
      // ask_human 的结果不渲染为 ToolCallCard（避免孤儿 toolResult）
      if (ev.name === 'ask_human') return threadId
      {
        // ID 桥接兜底：后端 tool_result.id 可能因 LangChain 事件
        // 不含 tool_call_id 而回退为工具名，导致与 tool_call.id 不匹配。
        // 按 ID 精确匹配 → 按 name+顺序兜底，确保结果能配对到正确的工具卡片。
        const calls = streamLast.toolCalls ?? []
        const results = streamLast.toolResults ?? []
        const hasResult = (cid: string) => results.some((r) => r.id === cid)
        let matchId = ev.id
        if (!calls.some((c) => c.id === ev.id)) {
          const fallback = calls.find(
            (c) => c.name === ev.name && !hasResult(c.id),
          )
          if (fallback) matchId = fallback.id
        }
        // 已存在同 ID 结果则替换（迟到的正确结果覆盖空占位）
        const filtered = results.filter((r) => r.id !== matchId)
        arr[streamIdx] = {
          ...streamLast,
          toolResults: [
            ...filtered,
            { id: matchId, name: ev.name, content: ev.content },
          ],
        }
      }
      commitThreadMessages(get, set, threadId, arr)
      break
    case 'interrupt': {
      if (!streamLast || streamLast.role !== 'assistant') return threadId
      arr[streamIdx] = { ...streamLast, streaming: false, interrupted: true }
      const info: InterruptInfo = { prompt: ev.prompt, choices: ev.choices, items: ev.items }
      set((s) => {
        const pendingInterrupts = { ...s.pendingInterrupts, [threadId]: info }
        const patch: Partial<AppState> = { pendingInterrupts }
        if (normKey(s.currentThreadId) === threadId) patch.pendingInterrupt = info
        return patch
      })
      commitThreadMessages(get, set, threadId, arr)
      ctx.finish()
      break
    }
    case 'cancelled':
      if (!streamLast || streamLast.role !== 'assistant') return threadId
      arr[streamIdx] = {
        ...fillMissingToolResults(streamLast),
        streaming: false,
        content: streamLast.content + (streamLast.content ? '\n\n' : '') + `> ⚠️ ${ev.content}`,
      }
      commitThreadMessages(get, set, threadId, arr)
      ctx.finish()
      get().fetchThreads()
      break
    case 'error':
      if (!streamLast || streamLast.role !== 'assistant') return threadId
      // 避免重复追加多条错误
      if (!streamLast.error) {
        arr[streamIdx] = {
          ...fillMissingToolResults(streamLast),
          streaming: false,
          error: true,
          content: streamLast.content + (streamLast.content ? '\n\n' : '') + `> ❌ ${ev.content}`,
        }
        commitThreadMessages(get, set, threadId, arr)
      }
      ctx.finish()
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
      // 节点正常结束且携带产出：在消息区追加节点结果块（与 token 实时流并存）
      if (ev.status === 'done' && ev.content) {
        const nodeMsg: ChatMessage = {
          id: nextId(),
          role: 'assistant',
          content: ev.content,
          nodeName: ev.node,
          toolCalls: [],
          toolResults: [],
          timestamp: Date.now(),
        }
        commitThreadMessages(get, set, threadId, [...arr, nodeMsg])
      }
      break
    }
    case 'workflow_status': {
      const wf = get().workflow
      if (wf) set({ workflow: { ...wf, workflow_status: ev.status } })
      break
    }
    case 'done': {
      if (!streamLast || streamLast.role !== 'assistant') return threadId
      arr[streamIdx] = { ...fillMissingToolResults(streamLast), streaming: false }
      commitThreadMessages(get, set, threadId, arr)
      set((s) => {
        const pendingInterrupts = { ...s.pendingInterrupts }
        delete pendingInterrupts[threadId]
        const patch: Partial<AppState> = {
          pendingInterrupts,
          totalTokens: ev.total_tokens ?? s.totalTokens,
        }
        if (normKey(s.currentThreadId) === threadId) patch.pendingInterrupt = null
        return patch
      })
      ctx.finish()
      get().fetchThreads()
      break
    }
  }
  return threadId
}

/** 把后端原始消息列表转换为前端回合结构 */
function rawToMessages(raw: RawMessage[]): ChatMessage[] {
  const turns: ChatMessage[] = []
  for (const m of raw) {
    if (m.role === 'user') {
      turns.push({ id: nextId(), role: 'user', content: m.content })
    } else if (m.role === 'assistant') {
      // ask_human 的 UI 由 InterruptCard 渲染，从 tool_calls 中过滤掉
      const toolCalls = (m.tool_calls ?? []).filter((tc) => tc.name !== 'ask_human')
      if (toolCalls.length) {
        turns.push({
          id: nextId(),
          role: 'assistant',
          content: m.content || '',
          toolCalls,
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
      // 跳过 ask_human 的工具结果
      if (m.name === 'ask_human') continue
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
  roles: [],
  currentRole: null,
  totalTokens: 0,
  workspace: null,
  viewMode: 'chat',
  workflows: [],
  workflow: null,
  workflowLoading: false,
  workflowError: null,
  messagesByThread: {},
  streamingThreads: {},
  pendingInterrupts: {},
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
    // 拉取初始 token 用量
    api.getMetrics().then((m) => set({ totalTokens: m.llm.total_tokens })).catch(() => {})
    // 拉取团队角色列表与当前角色
    void get().fetchRoles()
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
    // 支持并发：切换会话不再被全局 streaming 阻塞；
    // 目标线程自身正在流式时也能切回，继续展示其进行中的消息。
    set((s) => ({
      currentThreadId: id,
      pendingInterrupt: s.pendingInterrupts[id] ?? null,
    }))
    // 同步工作流显示：从 thread_id 反解工作流名称并加载对应图结构
    const wfName = workflowNameOf(id)
    if (wfName) void get().fetchWorkflow(wfName, true)
    // 加载该会话绑定的工作空间（输入栏左下角展示）
    void get().fetchWorkspace(id)
    const cached = get().messagesByThread[id]
    if (cached) {
      set({ messages: cached })
      return
    }
    set({ messages: [] })
    try {
      const r = await api.getMessages(id)
      if (get().currentThreadId === id) {
        const arr = rawToMessages(r.messages)
        commitThreadMessages(get, set, id, arr)
      }
    } catch {
      /* ignore */
    }
  },

  newThread: async () => {
    try {
      // 工作流模式下创建专属工作流会话，其余为普通对话
      const type = get().viewMode
      const r = type === 'workflow' ? await api.createThread('workflow', 'simple') : await api.createThread('chat')
      set((s) => ({
        currentThreadId: r.thread_id,
        messages: [],
        pendingInterrupt: null,
        workspace: null,
        messagesByThread: { ...s.messagesByThread, [r.thread_id]: [] },
      }))
      await get().fetchThreads()
    } catch (e) {
      console.error(e)
    }
  },

  newWorkflowThread: async (workflowName) => {
    try {
      const r = await api.createThread('workflow', workflowName)
      set((s) => ({
        currentThreadId: r.thread_id,
        messages: [],
        pendingInterrupt: null,
        workspace: null,
        messagesByThread: { ...s.messagesByThread, [r.thread_id]: [] },
      }))
      await get().fetchThreads()
    } catch (e) {
      console.error(e)
    }
  },

  deleteThread: async (id) => {
    try {
      await api.deleteThread(id)
      // 清理该线程的内存状态（含进行中流的 abort/看门狗）
      clearWatchdog(id)
      const fn = abortFns[id]
      if (fn) fn()
      delete abortFns[id]
      terminatedThreads.add(id)
      set((s) => {
        const messagesByThread = { ...s.messagesByThread }
        delete messagesByThread[id]
        const streamingThreads = { ...s.streamingThreads }
        delete streamingThreads[id]
        const pendingInterrupts = { ...s.pendingInterrupts }
        delete pendingInterrupts[id]
        const patch: Partial<AppState> = {
          messagesByThread,
          streamingThreads,
          pendingInterrupts,
        }
        if (s.currentThreadId === id) {
          patch.currentThreadId = null
          patch.messages = []
          patch.pendingInterrupt = null
        }
        return patch
      })
      await get().fetchThreads()
    } catch (e) {
      console.error(e)
    }
  },

  sendMessage: async (text) => {
    const trimmed = text.trim()
    const rawThreadId = get().currentThreadId
    if (!trimmed) return
    // 尚无会话时用哨兵 key（''），后端创建线程后经 thread_created 重绑定
    const key = normKey(rawThreadId)
    if (get().streamingThreads[key]) return

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
    // 以当前线程已展示的消息为基底（regenerate / editAndResend 截断后也保持一致）
    const arr = [...get().messages, userMsg, assistantMsg]
    commitThreadMessages(get, set, key, arr)
    markStreaming(get, set, key, true)

    // 清除上一轮的终止标记：finish() 会将 key 加入 terminatedThreads 且永不删除，
    // 若不清理，后续 handleStreamEvent 的 early return 会丢弃所有事件（需刷新才恢复）
    terminatedThreads.delete(key)

    // 可变线程 key：thread_created 事件后由 handleStreamEvent 返回新值，
    // 确保后续事件（token/done 等）写入正确的线程缓冲。
    // 修复：此前 key 为闭包常量，rebindThread 迁移状态后旧 key 的缓冲被删除，
    // 后续事件因找不到消息而被忽略，streaming 永不复位（需刷新才更新）。
    let activeKey = key

    const finish = () => {
      if (terminatedThreads.has(activeKey)) return
      terminatedThreads.add(activeKey)
      clearWatchdog(activeKey)
      delete abortFns[activeKey]
      markStreaming(get, set, activeKey, false)
    }

    const watchdogHandler = () => {
      const msgs = [...(get().messagesByThread[activeKey] ?? [])]
      const lastIndex = msgs.length - 1
      const last = msgs[lastIndex]
      if (last && last.role === 'assistant' && last.streaming) {
        msgs[lastIndex] = {
          ...last,
          streaming: false,
          error: true,
          content: last.content + (last.content ? '\n\n' : '') + '> ⏱️ 响应超时，请重试。',
        }
        commitThreadMessages(get, set, activeKey, msgs)
      }
      finish()
    }

    armWatchdog(key, watchdogHandler)

    abortFns[key] = api.streamChat({ message: trimmed, thread_id: rawThreadId }, (ev) => {
      // 每收到任意事件，重置看门狗（复用同一超时处理器，保持兜底有效）
      activeKey = handleStreamEvent(get, set, activeKey, ev, { finish })
      if (!terminatedThreads.has(activeKey)) armWatchdog(activeKey, watchdogHandler)
    })
  },

  regenerate: () => {
    const threadId = get().currentThreadId
    if (threadId === null) return
    if (get().streamingThreads[threadId]) return
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
    const truncated = msgs.slice(0, lastUserIdx)
    set({ messages: truncated })
    commitThreadMessages(get, set, threadId, truncated)
    void get().sendMessage(lastUserText)
  },

  editAndResend: (index, newText) => {
    const threadId = get().currentThreadId
    if (threadId === null) return
    if (get().streamingThreads[threadId]) return
    const msgs = [...get().messages]
    if (index < 0 || index >= msgs.length) return
    if (msgs[index].role !== 'user') return
    // 截断到该 user 消息（不含），重新发送新文本
    const truncated = msgs.slice(0, index)
    set({ messages: truncated })
    commitThreadMessages(get, set, threadId, truncated)
    void get().sendMessage(newText)
  },

  resume: (payload) => {
    const threadId = get().currentThreadId
    if (threadId === null) return
    if (get().streamingThreads[threadId]) return
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
    const arr = [...get().messages, assistantMsg]
    commitThreadMessages(get, set, threadId, arr)
    markStreaming(get, set, threadId, true)

    // 清除上一轮的终止标记，确保恢复流的事件不被阻塞
    terminatedThreads.delete(threadId)

    const finish = makeFinish(get, set, threadId)
    const watchdogHandler = makeWatchdogHandler(get, set, threadId, finish)
    armWatchdog(threadId, watchdogHandler)

    abortFns[threadId] = api.streamResume({ payload, thread_id: threadId }, (ev) => {
      if (!terminatedThreads.has(threadId)) armWatchdog(threadId, watchdogHandler)
      handleStreamEvent(get, set, threadId, ev, { finish })
    })
  },

  stopStreaming: () => {
    const key = normKey(get().currentThreadId)
    // 向后端发送显式停止信号：后端取消工作流后会回送 `cancelled` SSE 事件，
    // 由 handleStreamEvent 的 case 'cancelled' 负责最终回退 UI（清 streaming、补提示）。
    // 此处不再立即 abort 连接或本地改写消息——按钮保持"停止生成"直到后端确认取消，
    // 以避免"本地显示已中止、后端仍在执行"的假反馈。
    const threadId = get().currentThreadId
    if (threadId) void api.stop(threadId).catch(() => {})
    clearWatchdog(key)

    // 安全兜底：若后端在限时内未回送 cancelled（如断网、后端未实现取消），
    // 在此执行原始拆卸（abort 连接 + 标记终止 + 本地回写"_(已中止)_"），
    // 确保按钮不会永久卡在"停止生成"。
    const FALLBACK_MS = 1500
    setTimeout(() => {
      // 已由 cancelled 路径正常回退则无需兜底
      if (!get().streamingThreads[key]) return
      abortFns[key]?.()
      delete abortFns[key]
      terminatedThreads.add(key)
      set((s) => {
        const arr = [...(s.messagesByThread[key] ?? s.messages)]
        const lastIndex = arr.length - 1
        const last = arr[lastIndex]
        if (last && last.role === 'assistant' && last.streaming) {
          arr[lastIndex] = {
            ...last,
            streaming: false,
            content: last.content || '_(已中止)_',
          }
        }
        const streamingThreads = { ...s.streamingThreads }
        delete streamingThreads[key]
        const patch: Partial<AppState> = {
          streamingThreads,
          messagesByThread: { ...s.messagesByThread, [key]: arr },
        }
        if (normKey(s.currentThreadId) === key) patch.messages = arr
        return patch
      })
    }, FALLBACK_MS)
  },

  clearMessages: () => {
    const threadId = get().currentThreadId
    if (threadId === null) return
    if (get().streamingThreads[threadId]) return
    set((s) => {
      const pendingInterrupts = { ...s.pendingInterrupts }
      delete pendingInterrupts[threadId]
      const patch: Partial<AppState> = { messages: [], pendingInterrupt: null, pendingInterrupts }
      const messagesByThread = { ...s.messagesByThread, [threadId]: [] }
      patch.messagesByThread = messagesByThread
      return patch
    })
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

  fetchRoles: async () => {
    try {
      const r = await api.getRoles()
      set({ roles: r.roles, currentRole: r.current })
    } catch {
      /* ignore */
    }
  },

  switchRole: async (role) => {
    console.log('[前端] 切换角色:', role)
    try {
      await api.switchRole(role)
      await get().fetchRoles()
      console.log('[前端] 角色切换完成')
    } catch (e) {
      console.error('[前端] 切换角色失败:', e)
    }
  },

  fetchWorkspace: async (thread_id) => {
    try {
      const r = await api.getWorkspace(thread_id)
      // 防止异步竞态：仅当仍是当前会话时才更新
      if (get().currentThreadId === thread_id) {
        set({ workspace: r.workspace })
      }
    } catch {
      if (get().currentThreadId === thread_id) set({ workspace: null })
    }
  },

  setWorkspace: async (path) => {
    const tid = get().currentThreadId
    if (!tid) return false
    try {
      const r = await api.setWorkspace(tid, path)
      set({ workspace: r.workspace })
      console.log('[前端] 工作空间已绑定:', r.workspace)
      return true
    } catch (e) {
      console.error('[前端] 设置工作空间失败:', e)
      return false
    }
  },

  clearWorkspace: async () => {
    const tid = get().currentThreadId
    if (!tid) return
    try {
      await api.clearWorkspace(tid)
      set({ workspace: null })
      console.log('[前端] 工作空间已清除')
    } catch (e) {
      console.error('[前端] 清除工作空间失败:', e)
    }
  },
}))
