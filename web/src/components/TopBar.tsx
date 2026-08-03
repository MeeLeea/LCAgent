// 顶栏：会话标题 + 连接状态 + 提供商/模型选择器
import { useEffect } from 'react'
import { PanelLeft, Cpu } from 'lucide-react'
import { useStore } from '../store'

const STATUS_META = {
  connected: { color: 'var(--accent)', label: '已连接' },
  disconnected: { color: 'var(--danger)', label: '已断开' },
  checking: { color: 'var(--text-faint)', label: '检测中…' },
} as const

export function TopBar({ onExpand }: { onExpand: () => void }) {
  const providers = useStore((s) => s.providers)
  const currentProvider = useStore((s) => s.currentProvider)
  const currentModel = useStore((s) => s.currentModel)
  const threads = useStore((s) => s.threads)
  const currentThreadId = useStore((s) => s.currentThreadId)
  const switchProvider = useStore((s) => s.switchProvider)
  const switchModel = useStore((s) => s.switchModel)
  const connectionStatus = useStore((s) => s.connectionStatus)
  const checkConnection = useStore((s) => s.checkConnection)

  // 每 30 秒定时检测连接状态
  useEffect(() => {
    const timer = setInterval(() => void checkConnection(), 30_000)
    return () => clearInterval(timer)
  }, [checkConnection])

  const currentThread = threads.find((t) => t.thread_id === currentThreadId)
  const activeProvider = providers.find((p) => p.key === currentProvider)
  const models = activeProvider?.models ?? []
  const meta = STATUS_META[connectionStatus]

  return (
    <div className="topbar">
      <button className="icon-btn" onClick={onExpand} title="展开侧栏">
        <PanelLeft size={16} />
      </button>
      <div className="topbar-title">{currentThread?.preview || '新会话'}</div>

      {/* 连接状态圆点 */}
      <div
        className="connection-dot-wrap"
        onClick={() => void checkConnection()}
        title={`${meta.label}（点击重新检测）`}
      >
        <span
          className={`connection-dot ${connectionStatus}`}
          style={{ background: meta.color }}
        />
      </div>

      <div className="model-selector">
        <Cpu size={13} />
        <select
          value={currentProvider ?? ''}
          onChange={(e) => switchProvider(e.target.value)}
          title="切换提供商"
        >
          {providers.map((p) => (
            <option key={p.key} value={p.key} disabled={!p.has_key}>
              {p.name}
              {!p.has_key ? '（未配置）' : ''}
            </option>
          ))}
        </select>
        <span style={{ color: 'var(--text-faint)' }}>/</span>
        <select
          value={currentModel ?? ''}
          onChange={(e) => switchModel(e.target.value)}
          title="切换模型"
        >
          {models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
