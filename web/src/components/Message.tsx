// 消息气泡组件
import { memo, useState } from 'react'
import { Bot, User, Copy, Check, RotateCcw } from 'lucide-react'
import { Markdown } from './Markdown'
import { ToolCallCard } from './ToolCallCard'
import type { ChatMessage } from '../types'

interface Props {
  message: ChatMessage
  onRegenerate?: () => void
}

export const Message = memo(function Message({ message, onRegenerate }: Props) {
  const [copied, setCopied] = useState(false)
  const isUser = message.role === 'user'

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
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

        {(message.content || message.streaming) && (
          <div className={`bubble ${message.error ? 'error' : ''}`}>
            {message.content ? (
              <Markdown content={message.content} />
            ) : message.streaming ? (
              <span className="streaming-cursor" />
            ) : null}
            {message.streaming && message.content && <span className="streaming-cursor" />}
          </div>
        )}

        {!isUser && !message.streaming && message.content && (
          <div className="msg-actions">
            <button className="msg-action-btn" onClick={handleCopy}>
              {copied ? <Check size={12} /> : <Copy size={12} />}
              {copied ? '已复制' : '复制'}
            </button>
            {onRegenerate && (
              <button className="msg-action-btn" onClick={onRegenerate}>
                <RotateCcw size={12} />
                重新生成
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
})
