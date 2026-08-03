// 消息气泡组件
import { memo, useState, useRef, useEffect } from 'react'
import { Bot, User, Copy, Check, RotateCcw, Pencil, Send } from 'lucide-react'
import { Markdown } from './Markdown'
import { ToolCallCard } from './ToolCallCard'
import type { ChatMessage } from '../types'

interface Props {
  message: ChatMessage
  onRegenerate?: () => void
  onEdit?: (newText: string) => void
}

/** 格式化时间戳为 HH:MM */
function formatTime(ts?: number): string {
  if (!ts) return ''
  const d = new Date(ts)
  const h = String(d.getHours()).padStart(2, '0')
  const m = String(d.getMinutes()).padStart(2, '0')
  return `${h}:${m}`
}

export const Message = memo(function Message({ message, onRegenerate, onEdit }: Props) {
  const [copied, setCopied] = useState(false)
  const [editing, setEditing] = useState(false)
  const [editText, setEditText] = useState(message.content)
  const editRef = useRef<HTMLTextAreaElement>(null)
  const isUser = message.role === 'user'

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  const startEdit = () => {
    setEditText(message.content)
    setEditing(true)
  }

  // 进入编辑态时自动聚焦并自适应高度
  useEffect(() => {
    if (editing && editRef.current) {
      const el = editRef.current
      el.focus()
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 200) + 'px'
    }
  }, [editing])

  const handleEditKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submitEdit()
    }
    if (e.key === 'Escape') {
      e.preventDefault()
      setEditing(false)
    }
  }

  const submitEdit = () => {
    const trimmed = editText.trim()
    if (!trimmed || !onEdit) {
      setEditing(false)
      return
    }
    onEdit(trimmed)
    setEditing(false)
  }

  return (
    <div className={`msg ${isUser ? 'user' : 'assistant'}`}>
      <div className={`avatar ${isUser ? 'user' : 'assistant'}`}>
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>
      <div className="msg-body">
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="tool-group">
            {message.toolCalls.map((call) => (
              <ToolCallCard
                key={call.id}
                call={call}
                result={message.toolResults?.find((r) => r.id === call.id)}
              />
            ))}
          </div>
        )}

        {(message.content || message.streaming) && !editing && (
          <div className={`bubble ${message.error ? 'error' : ''}`}>
            {message.content ? (
              <Markdown content={message.content} />
            ) : message.streaming ? (
              <span className="streaming-cursor" />
            ) : null}
            {message.streaming && message.content && <span className="streaming-cursor" />}
          </div>
        )}

        {/* 用户消息编辑框 */}
        {editing && (
          <div className="msg-edit">
            <textarea
              ref={editRef}
              className="msg-edit-textarea"
              value={editText}
              rows={1}
              onChange={(e) => {
                setEditText(e.target.value)
                const el = e.currentTarget
                el.style.height = 'auto'
                el.style.height = Math.min(el.scrollHeight, 200) + 'px'
              }}
              onKeyDown={handleEditKey}
            />
            <div className="msg-edit-actions">
              <button className="msg-action-btn" onClick={submitEdit} title="发送（Enter）">
                <Send size={12} />
                发送
              </button>
              <button className="msg-action-btn" onClick={() => setEditing(false)} title="取消（Esc）">
                取消
              </button>
            </div>
          </div>
        )}

        {/* 操作按钮：非流式、非编辑态时显示 */}
        {!message.streaming && !editing && (
          <div className="msg-actions">
            {message.timestamp && (
              <span className="msg-time">{formatTime(message.timestamp)}</span>
            )}
            {isUser ? (
              <>
                <button className="msg-action-btn" onClick={handleCopy}>
                  {copied ? <Check size={12} /> : <Copy size={12} />}
                  {copied ? '已复制' : '复制'}
                </button>
                {onEdit && (
                  <button className="msg-action-btn" onClick={startEdit}>
                    <Pencil size={12} />
                    编辑
                  </button>
                )}
              </>
            ) : (
              <>
                {message.content && (
                  <button className="msg-action-btn" onClick={handleCopy}>
                    {copied ? <Check size={12} /> : <Copy size={12} />}
                    {copied ? '已复制' : '复制'}
                  </button>
                )}
                {onRegenerate && (
                  <button className="msg-action-btn" onClick={onRegenerate}>
                    <RotateCcw size={12} />
                    重新生成
                  </button>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
})
