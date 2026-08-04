// 应用根组件
import { useEffect, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import { useStore } from './store'
import { Sidebar } from './components/Sidebar'
import { TopBar } from './components/TopBar'
import { ChatView } from './components/ChatView'
import { WorkflowView } from './components/WorkflowView'
import { InputBar } from './components/InputBar'
import { TitleBar } from './components/TitleBar'

const DEFAULT_SIDEBAR_WIDTH = 264
const MIN_SIDEBAR_WIDTH = 180
const SIDEBAR_WIDTH_KEY = 'sidebar-width'

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max)
}

function loadSidebarWidth(): number {
  const num = Number(localStorage.getItem(SIDEBAR_WIDTH_KEY))
  if (!Number.isFinite(num)) return DEFAULT_SIDEBAR_WIDTH
  return clamp(num, MIN_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, window.innerWidth * 0.4))
}

export default function App() {
  const init = useStore((s) => s.init)
  const viewMode = useStore((s) => s.viewMode)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(loadSidebarWidth)
  const [isResizing, setIsResizing] = useState(false)
  const dragRef = useRef({ startX: 0, startWidth: 0 })

  useEffect(() => {
    init()
  }, [init])

  useEffect(() => {
    localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth))
  }, [sidebarWidth])

  const onResizeStart = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    dragRef.current = { startX: e.clientX, startWidth: sidebarWidth }
    setIsResizing(true)

    const onMove = (ev: PointerEvent) => {
      const maxWidth = Math.max(MIN_SIDEBAR_WIDTH, window.innerWidth * 0.4)
      const next = clamp(
        dragRef.current.startWidth + (ev.clientX - dragRef.current.startX),
        MIN_SIDEBAR_WIDTH,
        maxWidth,
      )
      setSidebarWidth(next)
    }
    const onUp = () => {
      setIsResizing(false)
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  return (
    <div className="app">
      <TitleBar />
      <div className="app-body">
        <aside
          className={`sidebar${sidebarCollapsed ? ' collapsed' : ''}${isResizing ? ' resizing' : ''}`}
          style={{ width: sidebarCollapsed ? 0 : sidebarWidth }}
        >
          <Sidebar />
          <div className="sidebar-resizer" onPointerDown={onResizeStart} />
        </aside>
        <div className="main">
          <TopBar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed((v) => !v)} />
          <div className="main-content">
            {viewMode === 'workflow' && <WorkflowView />}
            <ChatView />
          </div>
          <InputBar />
        </div>
      </div>
    </div>
  )
}
