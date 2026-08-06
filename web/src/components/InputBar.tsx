// 输入栏：自适应文本框 + 发送/停止按钮 + 工具列表 + 字符计数 + 上下文 Token 估算 + 快捷键
import { useRef, useState, useEffect, useMemo } from 'react'
import { ArrowUp, Square, Wrench, Keyboard, Users, Check, Loader2 } from 'lucide-react'
import { useStore, isWorkflowThread } from '../store'

export function InputBar() {
  const [text, setText] = useState('')
  const [toolPopoverOpen, setToolPopoverOpen] = useState(false)
  const [shortcutOpen, setShortcutOpen] = useState(false)
  const [rolePopoverOpen, setRolePopoverOpen] = useState(false)
  const [switchingRole, setSwitchingRole] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const toolWrapRef = useRef<HTMLDivElement>(null)
  const shortcutWrapRef = useRef<HTMLDivElement>(null)
  const roleWrapRef = useRef<HTMLDivElement>(null)
  const isStreaming = useStore((s) => s.isStreaming)
  const sendMessage = useStore((s) => s.sendMessage)
  const stopStreaming = useStore((s) => s.stopStreaming)
  const tools = useStore((s) => s.tools)
  const messages = useStore((s) => s.messages)
  const threads = useStore((s) => s.threads)
  const currentThreadId = useStore((s) => s.currentThreadId)
  const roles = useStore((s) => s.roles)
  const currentRole = useStore((s) => s.currentRole)
  const switchRole = useStore((s) => s.switchRole)

  const currentThread = threads.find((t) => t.thread_id === currentThreadId)
  const isWorkflow = currentThread
    ? isWorkflowThread(currentThread)
    : !!currentThreadId?.includes('workflow')

  // 自适应高度
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 200) + 'px'
  }, [text])

  // 全局快捷键：按 / 聚焦输入框
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '/' && document.activeElement?.tagName !== 'TEXTAREA' && document.activeElement?.tagName !== 'INPUT') {
        e.preventDefault()
        textareaRef.current?.focus()
      }
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [])

  // 点击外部关闭弹窗
  useEffect(() => {
    if (!toolPopoverOpen && !shortcutOpen && !rolePopoverOpen) return
    const handler = (e: MouseEvent) => {
      if (toolPopoverOpen && toolWrapRef.current && !toolWrapRef.current.contains(e.target as Node)) {
        setToolPopoverOpen(false)
      }
      if (shortcutOpen && shortcutWrapRef.current && !shortcutWrapRef.current.contains(e.target as Node)) {
        setShortcutOpen(false)
      }
      if (rolePopoverOpen && roleWrapRef.current && !roleWrapRef.current.contains(e.target as Node)) {
        setRolePopoverOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [toolPopoverOpen, shortcutOpen, rolePopoverOpen])

  const handleSwitchRole = async (role: string) => {
    if (switchingRole) return
    if (role === currentRole) {
      setRolePopoverOpen(false)
      return
    }
    setRolePopoverOpen(false)
    setSwitchingRole(true)
    await switchRole(role)
    setSwitchingRole(false)
  }

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

  const charCount = text.length
  const hasText = text.trim().length > 0

  // 估算当前会话上下文 token：所有已有消息内容 + 工具调用/结果 + 当前输入框文本
  const contextTokens = useMemo(() => {
    let totalChars = 0
    for (const m of messages) {
      totalChars += (m.content || '').length
      if (m.toolCalls) {
        for (const tc of m.toolCalls) {
          totalChars += JSON.stringify(tc.args ?? '').length
        }
      }
      if (m.toolResults) {
        for (const tr of m.toolResults) {
          totalChars += (tr.content || '').length
        }
      }
    }
    // 加上当前输入框文本
    totalChars += text.length
    return Math.max(1, Math.ceil(totalChars / 4))
  }, [messages, text])

  // 按 k 单位显示：0 → 0k, 500 → 0.5k, 12300 → 12.3k
  const formatTokens = (n: number) => (n / 1000).toFixed(1) + 'k'

  return (
    <div className="input-wrap">
      <div className="input-card">
        <textarea
          ref={textareaRef}
          id="chat-message"
          name="message"
          className="input-textarea"
          placeholder={
            isWorkflow
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
            {/* 角色选择 */}
            <div ref={roleWrapRef} style={{ position: 'relative' }}>
              <button
                className="input-tool-btn"
                onClick={() => setRolePopoverOpen((v) => !v)}
                disabled={switchingRole}
                title={`当前角色：${currentRole ?? '未设置'}（点击切换）`}
              >
                {switchingRole ? (
                  <Loader2 size={13} className="spin" />
                ) : (
                  <Users size={13} />
                )}
                角色：{switchingRole ? '切换中…' : (currentRole ?? '—')}
              </button>
              {rolePopoverOpen && (
                <div className="tool-popover role-popover">
                  <div className="tool-popover-title">团队角色（{roles.length}）</div>
                  <div className="tool-popover-list">
                    {roles.length === 0 ? (
                      <div className="tool-popover-empty">暂无可用角色</div>
                    ) : (
                      roles.map((r) => (
                        <div
                          key={r}
                          className={`role-popover-item${r === currentRole ? ' active' : ''}`}
                          onClick={() => handleSwitchRole(r)}
                        >
                          <Users size={11} />
                          <span className="role-popover-name">{r}</span>
                          {r === currentRole && <Check size={12} className="role-check" />}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* 工具列表 */}
            <div ref={toolWrapRef} style={{ position: 'relative' }}>
              <button
                className="input-tool-btn"
                onClick={() => setToolPopoverOpen((v) => !v)}
                title={`可用工具 ${tools.length} 个（点击查看）`}
              >
                <Wrench size={13} />
                {tools.length} 工具
              </button>
              {toolPopoverOpen && (
                <div className="tool-popover">
                  <div className="tool-popover-title">可用工具（{tools.length}）</div>
                  <div className="tool-popover-list">
                    {tools.length === 0 ? (
                      <div className="tool-popover-empty">暂无工具</div>
                    ) : (
                      tools.map((t) => (
                        <div key={t} className="tool-popover-item">
                          <Wrench size={11} />
                          {t}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* 快捷键提示 */}
            <div ref={shortcutWrapRef} style={{ position: 'relative' }}>
              <button
                className="input-tool-btn"
                onClick={() => setShortcutOpen((v) => !v)}
                title="键盘快捷键"
              >
                <Keyboard size={13} />
                快捷键
              </button>
              {shortcutOpen && (
                <div className="tool-popover">
                  <div className="tool-popover-title">键盘快捷键</div>
                  <div className="tool-popover-list">
                    <div className="shortcut-item">
                      <span className="shortcut-desc">发送消息</span>
                      <kbd className="shortcut-key">Enter</kbd>
                    </div>
                    <div className="shortcut-item">
                      <span className="shortcut-desc">换行</span>
                      <kbd className="shortcut-key">Shift + Enter</kbd>
                    </div>
                    <div className="shortcut-item">
                      <span className="shortcut-desc">取消编辑</span>
                      <kbd className="shortcut-key">Esc</kbd>
                    </div>
                    <div className="shortcut-item">
                      <span className="shortcut-desc">聚焦输入框</span>
                      <kbd className="shortcut-key">/</kbd>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* 字符计数 */}
            {charCount > 0 && (
              <span className={`char-count${charCount > 2000 ? ' warn' : ''}`}>
                {charCount}
              </span>
            )}
          </div>
          <div className="input-right">
            {/* 当前会话上下文估算 token */}
            <span className="token-count" title={`当前会话上下文估算 Token: ${contextTokens.toLocaleString()}`}>
              {formatTokens(contextTokens)}
            </span>
            {isStreaming ? (
              <button className="send-btn stop" onClick={stopStreaming} title="停止生成">
                <Square size={14} fill="currentColor" />
              </button>
            ) : (
              <button
                className="send-btn"
                onClick={handleSend}
                disabled={!hasText}
                title="发送"
              >
                <ArrowUp size={16} />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
