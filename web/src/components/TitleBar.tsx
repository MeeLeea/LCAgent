// 自定义窗口标题栏：替代 Windows 原生标题栏，与应用深色主题融合
import { useEffect, useState } from 'react'
import { Minus, Square, X, Copy } from 'lucide-react'
import { getCurrentWindow } from '@tauri-apps/api/window'

// 是否运行在 Tauri webview 中（Tauri 2 运行时会在 globalThis 注入 __TAURI_INTERNALS__）
function isTauri(): boolean {
  return typeof globalThis !== 'undefined' && '__TAURI_INTERNALS__' in globalThis
}

export function TitleBar() {
  const [maximized, setMaximized] = useState(false)
  // 懒加载获取窗口实例：仅在 Tauri 环境下调用，避免在普通浏览器或运行时注入未完成时抛错
  const [win] = useState<ReturnType<typeof getCurrentWindow> | null>(() => {
    if (!isTauri()) return null
    try {
      return getCurrentWindow()
    } catch {
      return null
    }
  })

  useEffect(() => {
    if (!win) return
    win.isMaximized().then(setMaximized).catch(() => {})
    const unlisten = win.onResized(() => {
      win.isMaximized().then(setMaximized).catch(() => {})
    })
    return () => void unlisten.then((fn) => fn())
  }, [win])

  const handleMinimize = () => win?.minimize().catch(() => {})
  const handleMaximize = () => win?.toggleMaximize().catch(() => {})
  const handleClose = () => win?.close().catch(() => {})

  return (
    <div className="titlebar" data-tauri-drag-region>
      <div className="titlebar-left" data-tauri-drag-region>
        <div className="titlebar-logo" data-tauri-drag-region>
          <span className="titlebar-logo-text">LC</span>
        </div>
        <span className="titlebar-title" data-tauri-drag-region>
          LC Agent · Work
        </span>
      </div>

      {win && (
        <div className="titlebar-controls">
          <button className="titlebar-btn" onClick={handleMinimize} title="最小化">
            <Minus size={15} strokeWidth={2} />
          </button>
          <button className="titlebar-btn" onClick={handleMaximize} title={maximized ? '还原' : '最大化'}>
            {maximized ? <Copy size={13} strokeWidth={2} /> : <Square size={12} strokeWidth={2} />}
          </button>
          <button className="titlebar-btn titlebar-close" onClick={handleClose} title="关闭">
            <X size={15} strokeWidth={2} />
          </button>
        </div>
      )}
    </div>
  )
}
