// 工作流视图：作为可折叠顶部面板展示工作流节点/边结构图与节点状态
import { useEffect, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import {
  GitBranch,
  Loader2,
  RefreshCw,
  AlertTriangle,
  MessageSquarePlus,
  ChevronUp,
  ChevronDown,
} from 'lucide-react'
import { useStore } from '../store'
import { WorkflowGraph } from './WorkflowGraph'
import type { WorkflowInfo, WorkflowNode } from '../types'

// 工作流面板高度区间与持久化 key
const DEFAULT_PANEL_HEIGHT = 280
const MIN_PANEL_HEIGHT = 52 // 拖拽允许的下限：约等于头部高度，可收缩为只剩标题栏
const MIN_VISIBLE_HEIGHT = 140 // 初始加载/展开时保证内容可见的下限
const PANEL_HEIGHT_KEY = 'workflow-panel-height'

function clampHeight(value: number, max: number) {
  return Math.min(Math.max(value, MIN_PANEL_HEIGHT), max)
}

function loadPanelHeight(): number {
  const num = Number(localStorage.getItem(PANEL_HEIGHT_KEY))
  if (!Number.isFinite(num)) return DEFAULT_PANEL_HEIGHT
  // 持久化值过小（曾被拖至接近收起）时回退默认值，避免刷新后面板内容不可见
  if (num < MIN_VISIBLE_HEIGHT) return DEFAULT_PANEL_HEIGHT
  return clampHeight(num, Math.max(MIN_PANEL_HEIGHT, window.innerHeight * 0.8))
}

const STATUS_TEXT: Record<WorkflowNode['status'], string> = {
  pending: '待执行',
  running: '执行中',
  done: '已完成',
}

const STATUS_DESC: Record<WorkflowInfo['workflow_status'], string> = {
  idle: '空闲',
  running: '正在运行',
  done: '执行完成',
}

function NodeCard({ node }: { node: WorkflowNode }) {
  return (
    <div className={`wf-node ${node.status}`}>
      <div className="wf-node-label">{node.label}</div>
      <div className="wf-node-status">
        <span className={`wf-node-dot ${node.status}`} />
        {STATUS_TEXT[node.status]}
      </div>
    </div>
  )
}

export function WorkflowView() {
  const workflow = useStore((s) => s.workflow)
  const workflows = useStore((s) => s.workflows)
  const loading = useStore((s) => s.workflowLoading)
  const error = useStore((s) => s.workflowError)
  const fetchWorkflow = useStore((s) => s.fetchWorkflow)
  const fetchWorkflows = useStore((s) => s.fetchWorkflows)
  const newWorkflowThread = useStore((s) => s.newWorkflowThread)
  const isStreaming = useStore((s) => s.isStreaming)
  const [collapsed, setCollapsed] = useState(false)
  const [panelHeight, setPanelHeight] = useState(loadPanelHeight)
  const [isResizing, setIsResizing] = useState(false)
  const dragRef = useRef({ startY: 0, startHeight: 0 })

  useEffect(() => {
    fetchWorkflows()
    if (!workflow) fetchWorkflow()
  }, [workflow, fetchWorkflow, fetchWorkflows])

  useEffect(() => {
    localStorage.setItem(PANEL_HEIGHT_KEY, String(panelHeight))
  }, [panelHeight])

  // 拖拽手柄：向上拖收缩面板，向下拖放大面板
  const onResizeStart = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    dragRef.current = { startY: e.clientY, startHeight: panelHeight }
    setIsResizing(true)

    const onMove = (ev: PointerEvent) => {
      const maxHeight = Math.max(MIN_PANEL_HEIGHT, window.innerHeight * 0.8)
      // 向上拖（clientY 减小）→ delta 正 → 高度减小
      const delta = dragRef.current.startY - ev.clientY
      setPanelHeight(clampHeight(dragRef.current.startHeight - delta, maxHeight))
    }
    const onUp = () => {
      setIsResizing(false)
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
  }

  const handleNewThread = () => {
    newWorkflowThread(workflow?.name ?? 'simple')
  }

  const status = workflow?.workflow_status ?? 'idle'
  const selectValue = workflow?.name ?? workflows[0] ?? ''

  // 折叠时不设固定高度，由内容（仅头部）撑开；拖拽时禁用过渡避免卡顿
  const panelStyle = collapsed
    ? undefined
    : { height: panelHeight, transition: isResizing ? 'none' : undefined }

  return (
    <div
      className={`workflow-view workflow-panel${collapsed ? ' collapsed' : ''}${isResizing ? ' resizing' : ''}`}
      style={panelStyle}
    >
      <div className="workflow-head">
        <div className="workflow-title">
          <GitBranch size={16} />
          <span>工作流：</span>
          <select
            className="workflow-select"
            value={selectValue}
            onChange={(e) => fetchWorkflow(e.target.value, true)}
            title="切换工作流"
          >
            {workflows.map((w) => (
              <option key={w} value={w}>
                {w}
              </option>
            ))}
          </select>
        </div>
        <div className="workflow-status">
          <span className={`wf-node-dot ${status}`} />
          {STATUS_DESC[status]}
        </div>
        <button
          className="icon-btn"
          onClick={handleNewThread}
          disabled={isStreaming}
          title="新建工作流会话"
        >
          <MessageSquarePlus size={15} />
        </button>
        <button className="icon-btn" onClick={() => fetchWorkflow(undefined, true)} title="刷新">
          <RefreshCw size={15} />
        </button>
        <button
          className="icon-btn"
          onClick={() => {
            // 展开：若记录的高度过小（曾被拖至接近收起），恢复默认高度确保内容可见
            if (collapsed && panelHeight < 120) setPanelHeight(DEFAULT_PANEL_HEIGHT)
            setCollapsed((c) => !c)
          }}
          title={collapsed ? '展开面板' : '收起面板'}
        >
          {collapsed ? <ChevronDown size={15} /> : <ChevronUp size={15} />}
        </button>
      </div>

      {!collapsed && (
        <div className="workflow-body">
          {loading ? (
            <div className="workflow-loading">
              <Loader2 size={20} className="spin" />
              正在加载工作流…
            </div>
          ) : error ? (
            <div className="workflow-error">
              <AlertTriangle size={20} />
              <span>{error}</span>
              <button className="choice-btn" onClick={() => fetchWorkflow()}>
                重试
              </button>
            </div>
          ) : !workflow ? (
            <div className="workflow-empty">
              <GitBranch size={22} />
              暂无工作流数据
            </div>
          ) : (
            <>
              <div className="workflow-section">
                <div className="workflow-section-title">节点</div>
                <div className="wf-node-list">
                  {workflow.nodes.map((n) => (
                    <NodeCard key={n.id} node={n} />
                  ))}
                </div>
              </div>

              <div className="workflow-section">
                <div className="workflow-section-title">执行链路</div>
                <div className="wf-graph-wrap">
                  <WorkflowGraph workflow={workflow} />
                </div>
              </div>
            </>
          )}
        </div>
      )}
      {/* 拖拽收缩手柄：位于面板底边，上下拖动调整面板高度 */}
      <div className="workflow-resizer" onPointerDown={onResizeStart} />
    </div>
  )
}
