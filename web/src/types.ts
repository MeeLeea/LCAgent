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
}

export interface InterruptInfo {
  prompt: string
  choices: { id: string; label: string }[]
}

/** SSE 事件联合类型 */
export type StreamEvent =
  | { type: 'thread_created'; thread_id: string }
  | { type: 'token'; content: string }
  | { type: 'tool_call'; id: string; name: string; args: unknown }
  | { type: 'tool_result'; id: string; name: string; content: string }
  | { type: 'interrupt'; prompt: string; choices: { id: string; label: string }[] }
  | { type: 'cancelled'; content: string }
  | { type: 'error'; content: string }
  | { type: 'done' }
