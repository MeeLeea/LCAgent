// 侧边栏：会话列表管理（内容渲染到外层 aside 容器中）
import { MessageSquarePlus, MessagesSquare, Trash2, PanelLeftClose } from 'lucide-react'
import { useStore, THEMES } from '../store'
import { ThemePicker } from './ThemePicker'

export function Sidebar({ onCollapse }: { onCollapse: () => void }) {
  const threads = useStore((s) => s.threads)
  const currentThreadId = useStore((s) => s.currentThreadId)
  const isStreaming = useStore((s) => s.isStreaming)
  const theme = useStore((s) => s.theme)
  const newThread = useStore((s) => s.newThread)
  const selectThread = useStore((s) => s.selectThread)
  const deleteThread = useStore((s) => s.deleteThread)

  const currentTheme = THEMES.find((t) => t.id === theme)

  return (
    <>
      <div className="sidebar-header">
        <div className="logo">
          <MessagesSquare size={18} />
        </div>
        <div>
          <div className="logo-title">LC Agent</div>
          <div className="logo-sub">Work 模式</div>
        </div>
      </div>

      <button className="new-thread-btn" onClick={() => newThread()} disabled={isStreaming}>
        <MessageSquarePlus size={16} />
        新建会话
      </button>

      <div className="thread-list">
        {threads.length === 0 && (
          <div style={{ padding: '20px 12px', color: 'var(--text-faint)', fontSize: 12 }}>
            暂无会话，点击上方按钮开始
          </div>
        )}
        {threads.map((t) => (
          <div
            key={t.thread_id}
            className={`thread-item ${t.thread_id === currentThreadId ? 'active' : ''}`}
            onClick={() => selectThread(t.thread_id)}
          >
            <MessagesSquare size={14} className="thread-icon" />
            <div className="thread-meta">
              <div className="thread-preview">{t.preview || '新会话'}</div>
              <div className="thread-count">{t.message_count} 条消息</div>
            </div>
            <button
              className="thread-delete"
              onClick={(e) => {
                e.stopPropagation()
                deleteThread(t.thread_id)
              }}
              title="删除会话"
            >
              <Trash2 size={13} />
            </button>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <span style={{ fontSize: 11 }}>{currentTheme?.name ?? '主题'}</span>
        <div style={{ display: 'flex', gap: 4 }}>
          <ThemePicker />
          <button className="icon-btn" onClick={onCollapse} title="收起侧栏">
            <PanelLeftClose size={15} />
          </button>
        </div>
      </div>
    </>
  )
}
