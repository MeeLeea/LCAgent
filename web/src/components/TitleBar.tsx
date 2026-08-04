// 自定义窗口标题栏：替代 Windows 原生标题栏，与应用深色主题融合
import { useEffect, useState } from 'react'
import { Minus, Square, X, Copy } from 'lucide-react'
import { getCurrentWindow } from '@tauri-apps/api/window'

const appWindow = getCurrentWindow()

export function TitleBar() {
  const [maximized, setMaximized] = useState(false)

  useEffect(() => {
    appWindow.isMaximized().then(setMaximized).catch(() => {})
    const unlisten = appWindow.onResized(() => {
      appWindow.isMaximized().then(setMaximized).catch(() => {})
    })
    return () => void unlisten.then((fn) => fn())
  }, [])

  const handleMinimize = () => appWindow.minimize().catch(() => {})
  const handleMaximize = () => appWindow.toggleMaximize().catch(() => {})
  const handleClose = () => appWindow.close().catch(() => {})

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
    </div>
  )
}
