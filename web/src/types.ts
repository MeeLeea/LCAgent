// 前后端共享的数据类型定义

export interface Provider {
  key: string
  name: string
  base_url: string
  models: string[]
  has_key: boolean
}

export interface ProvidersInfo {
  providers: Provider[]
  current_provider: string | null
  current_provider_name: string | null
  current_model: string | null
}

export interface ThreadSummary {
  thread_id: string
  message_count: number
  preview: string
  /** 会话类型：chat=普通对话，workflow=专属工作流会话 */
  type?: 'chat' | 'workflow'
  /** 工作流会话绑定的工作流名称（仅 type=workflow 时存在） */
  workflow_name?: string
}

export interface WorkflowNode {
  id: string
  label: string
  status: 'pending' | 'running' | 'done' | 'error'
}

export interface WorkflowEdge {
  source: string
  target: string
}

export interface WorkflowInfo {
  name: string
  workflow_status: 'idle' | 'running' | 'done'
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
}

export interface ToolCall {
  id: string
  name: string
  args: unknown
}

export interface ToolResult {
  id: string
  name: string
  content: string
}

/** 后端返回的原始消息（历史接口） */
export interface RawMessage {
  role: 'user' | 'assistant' | 'tool'
  content: string
  id?: string
  name?: string
  tool_calls?: ToolCall[]
}

/** 前端渲染用的消息回合 */
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  toolCalls?: ToolCall[]
  toolResults?: ToolResult[]
  streaming?: boolean
  error?: boolean
  interrupted?: boolean
  /** 消息创建时间戳（ms） */
  timestamp?: number
  /** 节点结果块标记：workflow 会话中某节点的产出消息（非空时按节点块渲染） */
  nodeName?: string
}

export interface InterruptItemChoice {
  id: string
  label: string
}

export interface InterruptItem {
  id: string
  question: string
  choices: InterruptItemChoice[]
}

export interface InterruptInfo {
  prompt: string
  choices: { id: string; label: string }[]
  /** 分组单选项：存在且非空时进入分组模式（user_confirmation），否则走扁平单选 */
  items?: InterruptItem[]
}

/** 运行时指标 */
export interface MetricsSummary {
  session: {
    duration_seconds: number
    turn_count: number
  }
  llm: {
    total_calls: number
    total_prompt_tokens: number
    total_completion_tokens: number
    total_tokens: number
    total_duration_ms: number
    by_provider: Record<string, {
      count: number
      prompt_tokens: number
      completion_tokens: number
      total_tokens: number
      avg_tokens: number
      total_ms: number
    }>
  }
  tools: {
    total_calls: number
    total_duration_ms: number
    by_name: Record<string, {
      count: number
      total_ms: number
      min_ms: number
      max_ms: number
      avg_ms: number
      failures: number
      timeouts: number
    }>
  }
  compaction: {
    total_count: number
    total_messages_before: number
    total_messages_after: number
    total_duration_ms: number
    messages_saved: number
  }
}

/** 上下文压缩结果 */
export interface CompactResult {
  compacted: boolean
  thread_id?: string
  message?: string
  summary?: string
  messages_before?: number
  messages_after?: number
}

/** 记忆摘要 */
export interface MemorySummary {
  thread_id: string
  checkpoint_messages: number
  checkpoint_backend: string
  checkpoint_file: string
  long_term_count: number
  total_threads: number
}

/** 长期记忆压缩结果 */
export interface CompressResult {
  success: boolean
  original_count?: number
  original_chars?: number
  compressed_chars?: number
  summary?: string
  error?: string
}

/** 安全策略配置 */
export interface SafetyConfig {
  mode: 'blacklist' | 'whitelist'
  confirm_dangerous: boolean
  blacklist?: string[]
  whitelist?: string[]
  path_protection?: Record<string, unknown>
}

/** 技能信息 */
export interface SkillInfo {
  name: string
  description: string
}

/** 会话导出结果 */
export interface ExportResult {
  thread_id: string
  format: string
  content: string
}

/** 工作空间绑定信息（GET/POST /api/threads/{id}/workspace） */
export interface WorkspaceInfo {
  thread_id: string
  workspace: string | null
}

/** 文件检索器：单个目录条目 */
export interface BrowseEntry {
  name: string
  path: string
  has_children: boolean
}

/** 文件检索器：目录浏览结果（GET /api/workspace/browse） */
export interface BrowseResult {
  path: string
  entries: BrowseEntry[]
  is_root: boolean
}

/** SSE 事件联合类型 */
export type StreamEvent =
  | { type: 'thread_created'; thread_id: string }
  | { type: 'token'; content: string }
  | { type: 'tool_call'; id: string; name: string; args: unknown }
  | { type: 'tool_result'; id: string; name: string; content: string }
  | { type: 'interrupt'; prompt: string; choices: { id: string; label: string }[]; items?: InterruptItem[] }
  | { type: 'cancelled'; content: string }
  | { type: 'error'; content: string }
  | { type: 'workflow_node'; node: string; status: 'running' | 'done' | 'error'; content?: string }
  | { type: 'workflow_status'; status: 'idle' | 'running' | 'done' }
  | { type: 'done'; content?: string; total_tokens?: number }
