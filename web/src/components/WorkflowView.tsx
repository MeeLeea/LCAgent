// 工作流视图：作为可折叠顶部面板展示工作流节点/边结构图与节点状态
import { useEffect, useMemo, useRef, useState } from 'react'
import type { PointerEvent as ReactPointerEvent } from 'react'
import {
  GitBranch,
  Loader2,
  RefreshCw,
  AlertTriangle,
  MessageSquarePlus,
  ChevronUp,
  ChevronDown,
  RotateCcw,
} from 'lucide-react'
import { useStore, selectIsStreaming } from '../store'
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
  error: '出错',
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
  const failedNode = useStore((s) => s.failedNode)
  const retryCurrentThread = useStore((s) => s.retryCurrentThread)
  const isStreaming = useStore(selectIsStreaming)
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

  // 刷新还原失败红态：以后端 Store 失败记录驱动，把仍处于 pending 的失败节点显示为 error。
  // 仅覆盖 pending（刷新后复位的状态）；实时态由 SSE workflow_node 事件驱动，互不冲突。
  const displayWorkflow = useMemo(() => {
    if (!workflow || !failedNode?.node) return workflow
    if (!workflow.nodes.some((n) => n.id === failedNode.node)) return workflow
    return {
      ...workflow,
      nodes: workflow.nodes.map((n) =>
        n.id === failedNode.node && n.status === 'pending' ? { ...n, status: 'error' as const } : n,
      ),
    }
  }, [workflow, failedNode])

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
            disabled={isStreaming}
            onChange={(e) => {
              const name = e.target.value
              fetchWorkflow(name, true)
              newWorkflowThread(name)
            }}
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
          {failedNode && (
            <div className="workflow-failure-banner">
              <AlertTriangle size={15} className="workflow-failure-icon" />
              <div className="workflow-failure-info">
                <span className="workflow-failure-title">
                  {failedNode.node ? `节点「${failedNode.node}」执行失败` : '工作流执行失败'}
                </span>
                {failedNode.error && (
                  <span className="workflow-failure-detail" title={failedNode.error}>
                    {failedNode.error}
                  </span>
                )}
              </div>
              <button
                className="choice-btn workflow-retry-btn"
                onClick={retryCurrentThread}
                disabled={isStreaming}
                title={
                  isStreaming
                    ? '正在执行中，结束后可重试'
                    : '从 checkpoint 断点仅重跑失败节点，已完成节点不重跑'
                }
              >
                <RotateCcw size={13} />
                从失败节点重试
              </button>
            </div>
          )}
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
                  {displayWorkflow!.nodes.map((n) => (
                    <NodeCard key={n.id} node={n} />
                  ))}
                </div>
              </div>

              <div className="workflow-section">
                <div className="workflow-section-title">执行链路</div>
                <div className="wf-graph-wrap">
                  <WorkflowGraph workflow={displayWorkflow!} />
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
