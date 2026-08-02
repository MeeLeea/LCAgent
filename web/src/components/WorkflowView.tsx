// 工作流视图：展示工作流节点/边结构图与节点状态
import { useEffect } from 'react'
import { GitBranch, Loader2, RefreshCw, AlertTriangle } from 'lucide-react'
import { useStore } from '../store'
import { WorkflowGraph } from './WorkflowGraph'
import type { WorkflowInfo, WorkflowNode } from '../types'

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

  useEffect(() => {
    fetchWorkflows()
    if (!workflow) fetchWorkflow()
  }, [workflow, fetchWorkflow, fetchWorkflows])

  if (loading) {
    return (
      <div className="workflow-view">
        <div className="workflow-loading">
          <Loader2 size={20} className="spin" />
          正在加载工作流…
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="workflow-view">
        <div className="workflow-error">
          <AlertTriangle size={20} />
          <span>{error}</span>
          <button className="choice-btn" onClick={() => fetchWorkflow()}>
            重试
          </button>
        </div>
      </div>
    )
  }

  if (!workflow) {
    return (
      <div className="workflow-view">
        <div className="workflow-empty">
          <GitBranch size={22} />
          暂无工作流数据
        </div>
      </div>
    )
  }

  return (
    <div className="workflow-view">
      <div className="workflow-head">
        <div className="workflow-title">
          <GitBranch size={16} />
          <span>工作流：</span>
          <select
            className="workflow-select"
            value={workflow.name}
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
          <span className={`wf-node-dot ${workflow.workflow_status}`} />
          {STATUS_DESC[workflow.workflow_status]}
        </div>
        <button className="icon-btn" onClick={() => fetchWorkflow(undefined, true)} title="刷新">
          <RefreshCw size={15} />
        </button>
      </div>

      <div className="workflow-body">
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
      </div>
    </div>
  )
}
