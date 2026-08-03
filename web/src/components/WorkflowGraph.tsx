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

function toMermaidCode(workflow: WorkflowInfo): string {
  const lines: string[] = ['flowchart TB']

  const nodeDefs = (id: string, indent = 2): string => {
    const n = workflow.nodes.find((x) => x.id === id)
    const label = n?.label ?? id
    // 括号类字符需转义，避免破坏 mermaid 语法
    const safe = label.replace(/[()]/g, '\\(')
    return `${' '.repeat(indent)}${id}["${safe}"]`
  }

  // 收集所有节点 ID（业务节点 + 哨兵节点）
  const bizIds = new Set(workflow.nodes.map((n) => n.id))
  const allNodeIds = new Set<string>()

  // 先定义所有业务节点
  for (const n of workflow.nodes) {
    lines.push(nodeDefs(n.id))
    allNodeIds.add(n.id)
  }

  // 哨兵节点 START/END 等
  for (const e of workflow.edges) {
    for (const id of [e.source, e.target]) {
      if (id.startsWith('__')) continue
      if (!bizIds.has(id) && !allNodeIds.has(id)) {
        allNodeIds.add(id)
        lines.push(nodeDefs(id))
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
          themeVariables: {
            primaryColor: colors.primary,
            primaryTextColor: colors.text,
            primaryBorderColor: colors.border,
            lineColor: colors.line,
            fontSize: '14px',
            // 指定具体字体族，避免 'inherit' 导致测量时字体不确定
            fontFamily: "'Segoe UI', 'PingFang SC', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif",
          },
          flowchart: {
            htmlLabels: false,
            padding: 16,
            nodeSpacing: 40,
            rankSpacing: 36,
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
