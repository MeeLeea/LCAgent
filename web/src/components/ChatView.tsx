// 聊天主视图：空状态 + 消息列表 + 中断选择 + 导出 + 统计 + 滚动控制
import { useLayoutEffect, useRef, useCallback, useState, useEffect } from 'react'
import {
  Sparkles,
  Search,
  FileText,
  Calculator,
  CalendarClock,
  Download,
  Trash2,
  ArrowDown,
  MessageSquare,
  Wrench,
} from 'lucide-react'
import { useStore, stripWorkflowPrefix } from '../store'
import { Message } from './Message'
import type { ChatMessage } from '../types'

const SUGGESTIONS = [
  { icon: Search, text: '帮我搜索一下 LangGraph 的最新特性' },
  { icon: FileText, text: '读取项目里的 README.md 并总结要点' },
  { icon: Calculator, text: '计算 (1234 * 567 + 890) / 42 等于多少' },
  { icon: CalendarClock, text: '每天早上 9 点提醒我查看任务进度' },
]

function InterruptCard() {
  const pendingInterrupt = useStore((s) => s.pendingInterrupt)
  const resume = useStore((s) => s.resume)
  if (!pendingInterrupt) return null

  return (
    <div className="msg assistant">
      <div className="avatar assistant">
        <Sparkles size={16} />
      </div>
      <div className="msg-body">
        <div className="interrupt-card">
          <div className="interrupt-prompt">{pendingInterrupt.prompt}</div>
          <div className="interrupt-choices">
            {pendingInterrupt.choices.length > 0 ? (
              pendingInterrupt.choices.map((c) => (
                <button
                  key={c.id}
                  className="choice-btn"
                  onClick={() => resume({ choice_id: c.id })}
                >
                  {c.label}
                </button>
              ))
            ) : (
              <button className="choice-btn" onClick={() => resume({ text: '继续' })}>
                继续
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

/** 把当前会话消息导出为 Markdown 文件并下载 */
function exportMarkdown(messages: ChatMessage[], preview: string) {
  const lines: string[] = []
  lines.push(`# ${preview || '对话记录'}`)
  lines.push('')
  lines.push(`> 导出时间：${new Date().toLocaleString('zh-CN')}`)
  lines.push('')
  for (const m of messages) {
    if (m.role === 'user') {
      lines.push(`## 🧑 用户`)
      lines.push('')
      lines.push(stripWorkflowPrefix(m.content))
      lines.push('')
    } else {
      if (m.toolCalls && m.toolCalls.length) {
        for (const tc of m.toolCalls) {
          lines.push(`### 🔧 工具调用：${tc.name}`)
          lines.push('```json')
          try {
            lines.push(JSON.stringify(tc.args, null, 2))
          } catch {
            lines.push(String(tc.args))
          }
          lines.push('```')
          lines.push('')
        }
      }
      if (m.content) {
        lines.push(`## 🤖 助手`)
        lines.push('')
        lines.push(m.content)
        lines.push('')
      }
    }
  }
  const blob = new Blob([lines.join('\n')], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  const safeName = (preview || '对话').replace(/[\\/:*?"<>|]/g, '_').slice(0, 40)
  a.download = `${safeName}.md`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export function ChatView() {
  const messages = useStore((s) => s.messages)
  const isStreaming = useStore((s) => s.isStreaming)
  const sendMessage = useStore((s) => s.sendMessage)
  const regenerate = useStore((s) => s.regenerate)
  const editAndResend = useStore((s) => s.editAndResend)
  const clearMessages = useStore((s) => s.clearMessages)
  const threads = useStore((s) => s.threads)
  const currentThreadId = useStore((s) => s.currentThreadId)
  const scrollRef = useRef<HTMLDivElement>(null)

  // 滚动控制：用户上滑时暂停自动滚动，显示"回到底部"按钮
  const [showScrollBtn, setShowScrollBtn] = useState(false)
  const autoScrollRef = useRef(true)

  // 监听滚动位置，判断是否在底部附近
  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const distFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
    const atBottom = distFromBottom < 80
    autoScrollRef.current = atBottom
    setShowScrollBtn(!atBottom && messages.length > 0)
  }, [messages.length])

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    el.addEventListener('scroll', handleScroll, { passive: true })
    return () => el.removeEventListener('scroll', handleScroll)
  }, [handleScroll])

  // 有新消息或流式更新时，仅在用户未上滑时自动滚到底部
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (el && autoScrollRef.current) {
      el.scrollTop = el.scrollHeight
    }
  }, [messages])

  const scrollToBottom = () => {
    const el = scrollRef.current
    if (el) {
      el.scrollTop = el.scrollHeight
      autoScrollRef.current = true
      setShowScrollBtn(false)
    }
  }

  const isEmpty = messages.length === 0 && !isStreaming

  const currentThread = threads.find((t) => t.thread_id === currentThreadId)

  // 只在最后一条 assistant 消息上显示重新生成
  const lastAssistantIdx = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant') return i
    }
    return -1
  })()

  const handleEdit = useCallback(
    (index: number) => (newText: string) => editAndResend(index, newText),
    [editAndResend],
  )

  // 统计：消息数、工具调用数
  const toolCallCount = messages.reduce(
    (sum, m) => sum + (m.toolCalls?.length ?? 0),
    0,
  )

  return (
    <div className="chat-scroll" ref={scrollRef}>
      {isEmpty ? (
        <div className="empty-state">
          <div className="empty-logo">
            <Sparkles size={28} />
          </div>
          <div className="empty-title">有什么可以帮你？</div>
          <div className="empty-desc">
            我是基于 LangChain + LangGraph 的智能 Agent，配备搜索、文件、终端、计算、定时任务等工具。试试下面这些：
          </div>
          <div className="suggestion-grid">
            {SUGGESTIONS.map((s) => (
              <button
                key={s.text}
                className="suggestion-card"
                onClick={() => sendMessage(s.text)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <s.icon size={14} style={{ color: 'var(--accent)' }} />
                  <span style={{ fontWeight: 600 }}>{s.text.split('，')[0]}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      ) : (
        <div className="chat-inner">
          {/* 工具栏：统计 + 导出 + 清空 */}
          {messages.length > 0 && (
            <div className="chat-toolbar">
              <div className="chat-stats">
                <span className="chat-stat-item" title="消息总数">
                  <MessageSquare size={11} />
                  {messages.length} 条
                </span>
                {toolCallCount > 0 && (
                  <span className="chat-stat-item" title="工具调用次数">
                    <Wrench size={11} />
                    {toolCallCount} 次工具
                  </span>
                )}
              </div>
              <div className="chat-toolbar-actions">
                <button
                  className="msg-action-btn"
                  onClick={() => exportMarkdown(messages, currentThread?.preview ?? '')}
                  title="导出当前会话为 Markdown"
                >
                  <Download size={12} />
                  导出
                </button>
                <button
                  className="msg-action-btn danger"
                  onClick={clearMessages}
                  disabled={isStreaming}
                  title="清空当前显示（不删后端历史）"
                >
                  <Trash2 size={12} />
                  清空
                </button>
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <Message
              key={m.id}
              message={m}
              onRegenerate={i === lastAssistantIdx && !isStreaming ? regenerate : undefined}
              onEdit={m.role === 'user' && !isStreaming ? handleEdit(i) : undefined}
            />
          ))}
          <InterruptCard />
        </div>
      )}

      {/* 滚动到底部浮动按钮 */}
      {showScrollBtn && (
        <button className="scroll-bottom-btn" onClick={scrollToBottom} title="回到底部">
          <ArrowDown size={18} />
          {isStreaming && <span className="scroll-bottom-pulse" />}
        </button>
      )}
    </div>
  )
}
