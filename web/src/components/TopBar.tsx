// 顶栏：会话标题 + 提供商/模型选择器
import { PanelLeft, Cpu } from 'lucide-react'
import { useStore } from '../store'

export function TopBar({ onExpand }: { onExpand: () => void }) {
  const providers = useStore((s) => s.providers)
  const currentProvider = useStore((s) => s.currentProvider)
  const currentModel = useStore((s) => s.currentModel)
  const threads = useStore((s) => s.threads)
  const currentThreadId = useStore((s) => s.currentThreadId)
  const switchProvider = useStore((s) => s.switchProvider)
  const switchModel = useStore((s) => s.switchModel)

  const currentThread = threads.find((t) => t.thread_id === currentThreadId)
  const activeProvider = providers.find((p) => p.key === currentProvider)
  const models = activeProvider?.models ?? []

  return (
    <div className="topbar">
      <button className="icon-btn" onClick={onExpand} title="展开侧栏">
        <PanelLeft size={16} />
      </button>
      <div className="topbar-title">{currentThread?.preview || '新会话'}</div>

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
