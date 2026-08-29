// 文件检索器：浏览目录树选择工作空间，选定后经 POST workspace 绑定并记录
import { useState, useEffect, useCallback, useMemo } from 'react'
import {
  X,
  Folder,
  FolderOpen,
  ChevronRight,
  ArrowLeft,
  Loader2,
  Check,
  HardDrive,
} from 'lucide-react'
import { api } from '../api'
import { useStore } from '../store'
import type { BrowseEntry, BrowseResult } from '../types'

/** 把绝对路径拆成可点击的面包屑段（兼容 Windows \ 与 Unix /） */
function splitPath(path: string): { name: string; path: string }[] {
  if (!path) return []
  const sep = path.includes('\\') ? '\\' : '/'
  const parts = path.split(/[\\/]/).filter(Boolean)
  const segs: { name: string; path: string }[] = []
  let acc = ''
  for (const p of parts) {
    if (acc === '' && /^[a-zA-Z]:$/.test(p)) {
      acc = p + sep
    } else {
      acc = acc ? acc + sep + p : p
    }
    segs.push({ name: p, path: acc })
  }
  return segs
}

export function WorkspacePicker({ onClose }: { onClose: () => void }) {
  const setWorkspace = useStore((s) => s.setWorkspace)
  const workspace = useStore((s) => s.workspace)

  const [currentPath, setCurrentPath] = useState<string>(workspace || '')
  const [entries, setEntries] = useState<BrowseEntry[]>([])
  const [isRoot, setIsRoot] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [selectedPath, setSelectedPath] = useState<string>(workspace || '')

  const loadPath = useCallback(async (path?: string) => {
    setLoading(true)
    setError(null)
    try {
      const r: BrowseResult = await api.browseFolders(path)
      setEntries(r.entries)
      setIsRoot(r.is_root)
      setCurrentPath(r.path)
      // 进入新目录时默认选中该目录本身
      if (r.path) setSelectedPath(r.path)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  // 初始加载：从当前 workspace 起步，无则从浏览起点（盘符/根目录）开始
  useEffect(() => {
    void loadPath(workspace || undefined)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ESC 关闭
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', h)
    return () => document.removeEventListener('keydown', h)
  }, [onClose])

  const breadcrumbs = useMemo(() => splitPath(currentPath), [currentPath])

  const handleEnter = (entry: BrowseEntry) => {
    void loadPath(entry.path)
  }

  const handleGoUp = () => {
    if (isRoot) return
    // 已经在盘符根（仅 1 段）→ 返回盘符列表
    if (breadcrumbs.length <= 1) {
      void loadPath(undefined)
      return
    }
    void loadPath(breadcrumbs[breadcrumbs.length - 2].path)
  }

  const handleConfirm = async () => {
    if (!selectedPath) return
    setSubmitting(true)
    const ok = await setWorkspace(selectedPath)
    setSubmitting(false)
    if (ok) onClose()
  }

  return (
    <div className="ws-picker-overlay" onClick={onClose}>
      <div className="ws-picker" onClick={(e) => e.stopPropagation()}>
        {/* 标题栏 */}
        <div className="ws-picker-header">
          <div className="ws-picker-title">
            <FolderOpen size={15} />
            <span>选择工作目录</span>
          </div>
          <button className="ws-picker-close" onClick={onClose} title="关闭">
            <X size={16} />
          </button>
        </div>

        {/* 面包屑导航 */}
        <div className="ws-picker-breadcrumb">
          <button
            className="ws-picker-up"
            onClick={handleGoUp}
            disabled={isRoot}
            title="返回上级"
          >
            <ArrowLeft size={14} />
          </button>
          <div className="ws-picker-crumbs">
            {isRoot || breadcrumbs.length === 0 ? (
              <span className="ws-crumb-root">
                <HardDrive size={12} />
                此电脑
              </span>
            ) : (
              breadcrumbs.map((seg, i) => (
                <span key={seg.path} className="ws-crumb-wrap">
                  {i > 0 && <ChevronRight size={11} className="ws-crumb-sep" />}
                  <span
                    className={`ws-crumb${i === breadcrumbs.length - 1 ? ' active' : ''}`}
                    onClick={() => void loadPath(seg.path)}
                  >
                    {seg.name}
                  </span>
                </span>
              ))
            )}
          </div>
        </div>

        {/* 目录列表 */}
        <div className="ws-picker-list">
          {loading ? (
            <div className="ws-picker-empty">
              <Loader2 size={18} className="spin" />
              <span>加载中…</span>
            </div>
          ) : error ? (
            <div className="ws-picker-empty ws-picker-error">
              <span>⚠️ {error}</span>
            </div>
          ) : entries.length === 0 ? (
            <div className="ws-picker-empty">
              <Folder size={18} />
              <span>没有子文件夹</span>
            </div>
          ) : (
            entries.map((entry) => {
              const isSelected = entry.path === selectedPath
              return (
                <div
                  key={entry.path}
                  className={`ws-picker-item${isSelected ? ' selected' : ''}`}
                  onClick={() => setSelectedPath(entry.path)}
                  onDoubleClick={() => handleEnter(entry)}
                  title={entry.path}
                >
                  <Folder size={14} className="ws-item-icon" />
                  <span className="ws-item-name">{entry.name}</span>
                  {entry.has_children && (
                    <ChevronRight
                      size={24}
                      className="ws-item-enter"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleEnter(entry)
                      }}
                    />
                  )}
                  {isSelected && <Check size={13} className="ws-item-check" />}
                </div>
              )
            })
          )}
        </div>

        {/* 底部操作栏 */}
        <div className="ws-picker-footer">
          <div className="ws-picker-selected" title={selectedPath}>
            {selectedPath ? (
              <>
                <FolderOpen size={12} />
                <span>{selectedPath}</span>
              </>
            ) : (
              <span className="ws-picker-hint">请选择一个文件夹</span>
            )}
          </div>
          <div className="ws-picker-actions">
            <button className="ws-btn ws-btn-ghost" onClick={onClose}>
              取消
            </button>
            <button
              className="ws-btn ws-btn-primary"
              onClick={handleConfirm}
              disabled={!selectedPath || submitting}
            >
              {submitting ? <Loader2 size={13} className="spin" /> : <Check size={13} />}
              选择此目录
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
