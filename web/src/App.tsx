// 应用根组件
import { useEffect, useState } from 'react'
import { useStore } from './store'
import { Sidebar } from './components/Sidebar'
import { TopBar } from './components/TopBar'
import { ChatView } from './components/ChatView'
import { InputBar } from './components/InputBar'

export default function App() {
  const init = useStore((s) => s.init)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  useEffect(() => {
    init()
  }, [init])

  return (
    <div className="app">
      <aside className={sidebarCollapsed ? 'sidebar collapsed' : 'sidebar'}>
        <Sidebar onCollapse={() => setSidebarCollapsed(true)} />
      </aside>
      <div className="main">
        <TopBar onExpand={() => setSidebarCollapsed(false)} />
        <ChatView />
        <InputBar />
      </div>
    </div>
  )
}
