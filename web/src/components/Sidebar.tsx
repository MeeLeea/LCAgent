// 侧边栏：会话列表管理（内容渲染到外层 aside 容器中）
import { useState } from 'react'
import { MessageSquarePlus, MessagesSquare, Trash2, PanelLeftClose, GitBranch, Search } from 'lucide-react'
import { useStore, THEMES, isWorkflowThread, stripWorkflowPrefix } from '../store'
import { ThemePicker } from './ThemePicker'

export function Sidebar({ onCollapse }: { onCollapse: () => void }) {
  const threads = useStore((s) => s.threads)
  const currentThreadId = useStore((s) => s.currentThreadId)
  const isStreaming = useStore((s) => s.isStreaming)
  const theme = useStore((s) => s.theme)
  const viewMode = useStore((s) => s.viewMode)
  const switchViewMode = useStore((s) => s.switchViewMode)
  const workflow = useStore((s) => s.workflow)
  const newThread = useStore((s) => s.newThread)
  const newWorkflowThread = useStore((s) => s.newWorkflowThread)
  const selectThread = useStore((s) => s.selectThread)
  const deleteThread = useStore((s) => s.deleteThread)
  const [search, setSearch] = useState('')

  const currentTheme = THEMES.find((t) => t.id === theme)

  const visibleThreads = threads.filter((t) =>
    viewMode === 'workflow' ? isWorkflowThread(t) : !isWorkflowThread(t),
  )
  // 搜索过滤：匹配 preview 或 thread_id
  const searchLower = search.trim().toLowerCase()
  const filteredThreads = searchLower
    ? visibleThreads.filter(
        (t) =>
          (t.preview || '').toLowerCase().includes(searchLower) ||
          t.thread_id.toLowerCase().includes(searchLower),
      )
    : visibleThreads

  const switchMode = (mode: 'chat' | 'workflow') => {
    void switchViewMode(mode)
  }

  // 工作流模式下「新建会话」创建绑定当前工作流的专属工作流会话
  const handleNew = () => {
    if (viewMode === 'workflow') newWorkflowThread(workflow?.name ?? 'simple')
    else newThread()
  }

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

      <div className="mode-switch">
        <button
          className={`mode-switch-btn${viewMode === 'chat' ? ' active' : ''}`}
          onClick={() => switchMode('chat')}
        >
          <MessageSquarePlus size={14} />
          对话
        </button>
        <button
          className={`mode-switch-btn${viewMode === 'workflow' ? ' active' : ''}`}
          onClick={() => switchMode('workflow')}
        >
          <GitBranch size={14} />
          工作流
        </button>
      </div>

      <button className="new-thread-btn" onClick={handleNew} disabled={isStreaming}>
        <GitBranch size={16} />
        {viewMode === 'workflow' ? '新建工作流会话' : '新建会话'}
      </button>

      <div className="thread-search">
        <Search size={13} />
        <input
          type="text"
          className="thread-search-input"
          placeholder="搜索会话…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="thread-list">
        {filteredThreads.length === 0 && (
          <div style={{ padding: '20px 12px', color: 'var(--text-faint)', fontSize: 12 }}>
            {searchLower
              ? '没有匹配的会话'
              : viewMode === 'workflow'
                ? '暂无工作流会话，请先运行工作流'
                : '暂无会话，点击上方按钮开始'}
          </div>
        )}
        {filteredThreads.map((t) => (
          <div
            key={t.thread_id}
            className={`thread-item ${t.thread_id === currentThreadId ? 'active' : ''}`}
            onClick={() => selectThread(t.thread_id)}
          >
            {isWorkflowThread(t) ? (
              <GitBranch size={14} className="thread-icon" />
            ) : (
              <MessagesSquare size={14} className="thread-icon" />
            )}
            <div className="thread-meta">
              <div className="thread-preview">
                {t.preview ? stripWorkflowPrefix(t.preview) : '新会话'}
                {isWorkflowThread(t) && (
                  <span className="thread-badge" title={`工作流: ${t.workflow_name ?? ''}`}>
                    工作流
                  </span>
                )}
              </div>
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
