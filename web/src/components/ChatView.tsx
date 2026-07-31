// 聊天主视图：空状态 + 消息列表 + 中断选择
import { useLayoutEffect, useRef } from 'react'
import { Sparkles, Search, FileText, Calculator, CalendarClock } from 'lucide-react'
import { useStore } from '../store'
import { Message } from './Message'

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

export function ChatView() {
  const messages = useStore((s) => s.messages)
  const isStreaming = useStore((s) => s.isStreaming)
  const sendMessage = useStore((s) => s.sendMessage)
  const scrollRef = useRef<HTMLDivElement>(null)

  // 有新消息或流式更新时自动滚到底部
  // 用 useLayoutEffect 在绘制前同步滚动，避免切换会话时看到从顶部滑到底部的动画
  useLayoutEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  const isEmpty = messages.length === 0 && !isStreaming

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
          {messages.map((m) => (
            <Message key={m.id} message={m} />
          ))}
          <InterruptCard />
        </div>
      )}
    </div>
  )
}
