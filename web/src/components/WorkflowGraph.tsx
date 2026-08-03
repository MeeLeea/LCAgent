// 工作流图形化渲染：把 WorkflowInfo 转为 mermaid flowchart 并渲染为 SVG
import { useCallback, useEffect, useRef, useState } from 'react'
import mermaid from 'mermaid'
import { useStore } from '../store'
import type { WorkflowInfo } from '../types'

// mermaid 渲染 id 需全局唯一，用计数器避免多次渲染时 DOM id 冲突
let renderSeq = 0

// 深浅主题的节点/连线配色（mermaid themeVariables 需合法颜色值，不能传 CSS var()）
const THEME_COLORS = {
  dark: {
    primary: '#16291f',
    text: '#ececec',
    border: '#262626',
    line: '#8a8a8a',
  },
  default: {
    primary: '#e8f7ee',
    text: '#1a1a1a',
    border: '#e5e5e7',
    line: '#6b6b6b',
  },
}

/** 根据当前主题选择 mermaid 主题与配色 */
function mermaidOptions(theme: string): { theme: 'dark' | 'default'; colors: typeof THEME_COLORS.dark } {
  const isDark = theme !== 'light' && theme !== 'sepia'
  return {
    theme: isDark ? 'dark' : 'default',
    colors: isDark ? THEME_COLORS.dark : THEME_COLORS.default,
  }
}

// 每个子图容纳的业务节点数，节点多时折行，避免单行拉得过长
const GROUP_SIZE = 2

function toMermaidCode(workflow: WorkflowInfo): string {
  const lines: string[] = ['flowchart TB']

  const nodeDefs = (id: string, indent = 4): string => {
    const n = workflow.nodes.find((x) => x.id === id)
    const label = n?.label ?? id
    // 括号类字符需转义，避免破坏 mermaid 语法
    const safe = label.replace(/[()]/g, '\\(')
    return `${' '.repeat(indent)}${id}["${safe}"]`
  }

  // 按业务节点顺序分组成子图，形成折行布局
  const bizIds = workflow.nodes.map((n) => n.id)
  for (let i = 0; i < bizIds.length; i += GROUP_SIZE) {
    const group = bizIds.slice(i, i + GROUP_SIZE)
    const groupName = `group_${i / GROUP_SIZE}`
    lines.push(`  subgraph ${groupName}`)
    for (const id of group) {
      lines.push(nodeDefs(id))
    }
    lines.push('  end')
  }

  // 哨兵节点 START/END 定义在子图外
  const sentinelNodes = new Set<string>()
  for (const e of workflow.edges) {
    for (const id of [e.source, e.target]) {
      if (id.startsWith('__')) continue
      if (!bizIds.includes(id) && !sentinelNodes.has(id)) {
        sentinelNodes.add(id)
        lines.push(nodeDefs(id, 2))
      }
    }
  }

  for (const e of workflow.edges) {
    lines.push(`  ${e.source} --> ${e.target}`)
  }
  return lines.join('\n')
}

export function WorkflowGraph({ workflow }: { workflow: WorkflowInfo }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const theme = useStore((s) => s.theme)
  const [error, setError] = useState<string | null>(null)

  const renderFallback = useCallback(() => {
    const chain = workflow.edges.map((e) => `${e.source} → ${e.target}`).join('，')
    return chain || '无可视化数据'
  }, [workflow.edges])

  useEffect(() => {
    let cancelled = false
    const { theme: mTheme, colors } = mermaidOptions(theme)

    const render = async () => {
      try {
        mermaid.initialize({
          startOnLoad: false,
          theme: mTheme,
          securityLevel: 'loose',
          fontFamily: 'inherit',
          themeVariables: {
            primaryColor: colors.primary,
            primaryTextColor: colors.text,
            primaryBorderColor: colors.border,
            lineColor: colors.line,
            fontSize: '14px',
            fontFamily: 'inherit',
          },
          flowchart: {
            htmlLabels: true,
            padding: 8,
            nodeSpacing: 32,
            rankSpacing: 28,
            useMaxWidth: true,
          },
        })
        const code = toMermaidCode(workflow)
        const renderId = `wf-graph-${++renderSeq}`
        const { svg } = await mermaid.render(renderId, code)
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg
          setError(null)
        }
      } catch (e) {
        // 渲染失败时回退为文本描述，保证工作流信息不丢失
        if (!cancelled) {
          setError(`流程图渲染失败: ${(e as Error).message}`)
        }
      }
    }

    void render()

    return () => {
      cancelled = true
    }
  }, [workflow, theme])

  return (
    <div className="wf-graph">
      {error && (
        <div className="wf-graph-fallback">
          <div style={{ color: 'var(--danger)', marginBottom: 8 }}>{error}</div>
          {renderFallback()}
        </div>
      )}
      <div ref={containerRef} />
    </div>
  )
}
