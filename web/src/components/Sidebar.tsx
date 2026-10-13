// 侧边栏：会话列表管理（内容渲染到外层 aside 容器中）
import { useEffect, useState } from 'react'
import {
  MessageSquarePlus,
  MessagesSquare,
  Trash2,
  GitBranch,
  GitFork,
  Search,
  MoreVertical,
  Pencil,
} from 'lucide-react'
import { useStore, THEMES, isWorkflowThread, stripWorkflowPrefix } from '../store'
import type { ThreadSummary } from '../types'
import { ThemePicker } from './ThemePicker'

/** 会话列表展示名：自定义名优先，回落消息预览 */
function displayName(t: ThreadSummary): string {
  if (t.title) return t.title
  return t.preview ? stripWorkflowPrefix(t.preview) : '新会话'
}

export function Sidebar() {
  const threads = useStore((s) => s.threads)
  const currentThreadId = useStore((s) => s.currentThreadId)
  const theme = useStore((s) => s.theme)
  const viewMode = useStore((s) => s.viewMode)
  const switchViewMode = useStore((s) => s.switchViewMode)
  const workflow = useStore((s) => s.workflow)
  const newThread = useStore((s) => s.newThread)
  const newWorkflowThread = useStore((s) => s.newWorkflowThread)
  const selectThread = useStore((s) => s.selectThread)
  const deleteThread = useStore((s) => s.deleteThread)
  const renameThread = useStore((s) => s.renameThread)
  const forkThread = useStore((s) => s.forkThread)
  const [search, setSearch] = useState('')
  /** 当前展开操作菜单的会话 ID（null=全部收起） */
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null)
  /** 正在内联重命名的会话 ID */
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editingValue, setEditingValue] = useState('')

  // 点击菜单/按钮外部时收起菜单
  useEffect(() => {
    if (!menuOpenId) return
    const onDocMouseDown = (e: MouseEvent) => {
      const target = e.target as HTMLElement
      if (!target.closest('.thread-menu') && !target.closest('.thread-menu-btn')) {
        setMenuOpenId(null)
      }
    }
    document.addEventListener('mousedown', onDocMouseDown)
    return () => document.removeEventListener('mousedown', onDocMouseDown)
  }, [menuOpenId])

  const currentTheme = THEMES.find((t) => t.id === theme)

  const visibleThreads = threads.filter((t) =>
    viewMode === 'workflow' ? isWorkflowThread(t) : !isWorkflowThread(t),
  )
  // 搜索过滤：匹配展示名（自定义名或 preview）或 thread_id
  const searchLower = search.trim().toLowerCase()
  const filteredThreads = searchLower
    ? visibleThreads.filter(
        (t) =>
          displayName(t).toLowerCase().includes(searchLower) ||
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

  // 开始内联重命名：预填当前展示名并全选
  const startRename = (t: ThreadSummary) => {
    setMenuOpenId(null)
    setEditingId(t.thread_id)
    setEditingValue(displayName(t))
  }

  // 提交重命名：值有变化才请求（含清空=清除自定义名）；空输入视为取消
  const commitRename = (id: string) => {
    if (editingId !== id) return
    const value = editingValue.trim()
    setEditingId(null)
    if (!value) return
    const thread = threads.find((t) => t.thread_id === id)
    if (thread && value !== displayName(thread)) void renameThread(id, value)
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

      {/* 并发多会话：当前线程流式时仍可新建/切换会话，其他会话可继续聊天 */}
      <button className="new-thread-btn" onClick={handleNew}>
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
        {filteredThreads.map((t) => {
          const editing = editingId === t.thread_id
          return (
            <div
              key={t.thread_id}
              className={`thread-item ${t.thread_id === currentThreadId ? 'active' : ''}`}
              onClick={() => {
                if (!editing) selectThread(t.thread_id)
              }}
            >
              {isWorkflowThread(t) ? (
                <GitBranch size={14} className="thread-icon" />
              ) : (
                <MessagesSquare size={14} className="thread-icon" />
              )}
              <div className="thread-meta">
                {editing ? (
                  <input
                    className="thread-rename-input"
                    value={editingValue}
                    autoFocus
                    maxLength={200}
                    onFocus={(e) => e.target.select()}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => setEditingValue(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') commitRename(t.thread_id)
                      if (e.key === 'Escape') setEditingId(null)
                    }}
                    onBlur={() => commitRename(t.thread_id)}
                  />
                ) : (
                  <div className="thread-preview">
                    {displayName(t)}
                    {isWorkflowThread(t) && (
                      <span className="thread-badge" title={`工作流: ${t.workflow_name ?? ''}`}>
                        工作流
                      </span>
                    )}
                  </div>
                )}
                <div className="thread-count">{t.message_count} 条消息</div>
              </div>
              <button
                className="thread-menu-btn"
                onClick={(e) => {
                  e.stopPropagation()
                  setMenuOpenId(menuOpenId === t.thread_id ? null : t.thread_id)
                }}
                title="会话操作"
              >
                <MoreVertical size={13} />
              </button>
              {menuOpenId === t.thread_id && (
                <div className="thread-menu" onClick={(e) => e.stopPropagation()}>
                  <button className="thread-menu-item" onClick={() => startRename(t)}>
                    <Pencil size={13} />
                    重命名
                  </button>
                  <button
                    className="thread-menu-item"
                    onClick={() => {
                      setMenuOpenId(null)
                      forkThread(t.thread_id)
                    }}
                  >
                    <GitFork size={13} />
                    分叉
                  </button>
                  <button
                    className="thread-menu-item danger"
                    onClick={() => {
                      setMenuOpenId(null)
                      deleteThread(t.thread_id)
                    }}
                  >
                    <Trash2 size={13} />
                    删除
                  </button>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <div className="sidebar-footer">
        <span style={{ fontSize: 11 }}>{currentTheme?.name ?? '主题'}</span>
        <ThemePicker />
      </div>
    </>
  )
}
