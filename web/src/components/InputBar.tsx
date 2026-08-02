// 输入栏：自适应文本框 + 发送/停止按钮
import { useRef, useState, useEffect } from 'react'
import { ArrowUp, Square, Wrench } from 'lucide-react'
import { useStore } from '../store'

export function InputBar() {
  const [text, setText] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const isStreaming = useStore((s) => s.isStreaming)
  const sendMessage = useStore((s) => s.sendMessage)
  const stopStreaming = useStore((s) => s.stopStreaming)
  const tools = useStore((s) => s.tools)
  const threads = useStore((s) => s.threads)
  const currentThreadId = useStore((s) => s.currentThreadId)

  const currentThread = threads.find((t) => t.thread_id === currentThreadId)
  const isWorkflowThread = currentThread?.type === 'workflow' || currentThreadId?.includes('workflow')

  // 自适应高度
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }, [text])

  const handleSend = () => {
    if (isStreaming) return
    if (!text.trim()) return
    sendMessage(text)
    setText('')
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="input-wrap">
      <div className="input-card">
        <textarea
          ref={textareaRef}
          id="chat-message"
          name="message"
          className="input-textarea"
          placeholder={
            isWorkflowThread
              ? '输入任务，自动以工作流方式执行…'
              : currentThreadId
                ? '输入消息，Enter 发送，Shift+Enter 换行…'
                : '输入消息开始对话…'
          }
          value={text}
          rows={1}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <div className="input-toolbar">
          <div className="input-tools">
            <span className="input-tool-btn" title={`可用工具 ${tools.length} 个`}>
              <Wrench size={13} />
              {tools.length} 工具
            </span>
          </div>
          {isStreaming ? (
            <button className="send-btn stop" onClick={stopStreaming} title="停止生成">
              <Square size={14} fill="currentColor" />
            </button>
          ) : (
            <button
              className="send-btn"
              onClick={handleSend}
              disabled={!text.trim()}
              title="发送"
            >
              <ArrowUp size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
