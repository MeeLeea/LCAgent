// 工具调用卡片：展示工具名、参数、执行结果（默认收起，点击展开）
import { useState } from 'react'
import { ChevronDown, ChevronRight, Terminal, Check, Loader2, AlertTriangle, Clock } from 'lucide-react'
import type { ToolCall, ToolResult } from '../types'

function formatArgs(args: unknown): string {
  try {
    return JSON.stringify(args, null, 2)
  } catch {
    return String(args)
  }
}

/** 启发式判断工具结果是否为异常（后端将错误信息写入 content） */
function isErrorContent(content: string): boolean {
  return /Traceback|Error[:：]|错误|失败|Exception|❌|执行出错/i.test(content)
}

/** 启发式判断工具结果是否为超时
 * 与后端 agent/turn_runners.py、team/base.py 的识别逻辑保持一致：
 * - terminal_tools.py 超时返回含 "error_type": "timeout"
 * - tool_wrapper.py 超时返回含 "error": "tool_timeout"
 * - terminal_tools.py 旧文案含 "执行超时" / "命令超时"
 */
function isTimeoutContent(content: string): boolean {
  if (content.includes('"error_type": "timeout"')) return true
  if (content.includes('"error": "tool_timeout"')) return true
  return /执行超时|命令超时|操作超时/.test(content)
}

/**
 * 从完整输出中提取异常摘要：
 * - Python Traceback：取最后几行（实际的异常类型 + 消息）
 * - 其他错误：取包含错误关键词的行
 * - 兜底：取最后 3 行
 */
function extractErrorSummary(content: string): string {
  const lines = content.split('\n').filter((l) => l.trim())
  if (lines.length === 0) return content

  // Python Traceback：最后的异常行（跳过 "During handling..." 等嵌套信息）
  if (/Traceback/i.test(content)) {
    const tail: string[] = []
    for (let i = lines.length - 1; i >= 0 && tail.length < 3; i--) {
      tail.unshift(lines[i])
      // 遇到 Traceback 行就停（它本身不是错误信息）
      if (/Traceback/i.test(lines[i])) break
    }
    return tail.join('\n')
  }

  // 非 Traceback：取包含错误关键词的行
  const errorLines = lines.filter((l) =>
    /Error[:：]|错误|失败|Exception|❌|执行出错/i.test(l),
  )
  if (errorLines.length > 0) {
    return errorLines.slice(0, 3).join('\n')
  }

  // 兜底：最后 3 行
  return lines.slice(-3).join('\n')
}

/**
 * 截断极长的结果内容，防止 DOM 膨胀影响性能。
 * 滑动窗口负责视觉上的滚动，这里只做硬上限保护。
 */
function truncate(text: string, max = 20000): string {
  if (text.length <= max) return text
  return text.slice(0, max) + `\n... (已截断，共 ${text.length} 字符)`
}

export function ToolCallCard({ call, result }: { call: ToolCall; result?: ToolResult }) {
  // 卡片整体默认收起
  const [open, setOpen] = useState(false)

  const done = !!result
  const isTimeout = !!(result && result.content && isTimeoutContent(result.content))
  const isError = !isTimeout && !!(result && result.content && isErrorContent(result.content))

  // 异常时只取错误摘要；正常时保留完整内容（由滑动窗口滚动查看）
  const displayContent = result?.content
    ? isError
      ? extractErrorSummary(result.content)
      : truncate(result.content)
    : ''

  return (
    <div className={`tool-card${isError ? ' tool-card-error' : ''}${isTimeout ? ' tool-card-timeout' : ''}`}>
      <div className="tool-card-head" onClick={() => setOpen((v) => !v)}>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        {isError ? <AlertTriangle size={13} /> : isTimeout ? <Clock size={13} /> : <Terminal size={13} />}
        <span className="tool-card-name">{call.name}</span>
        <span className="tool-card-status">
          {done ? (
            isTimeout ? (
              <>
                <Clock size={12} /> 超时
              </>
            ) : isError ? (
              <>
                <AlertTriangle size={12} /> 异常
              </>
            ) : (
              <>
                <Check size={12} /> 已完成
              </>
            )
          ) : (
            <>
              <Loader2 size={12} className="spin" /> 执行中
            </>
          )}
        </span>
      </div>
      {open && (
        <div className="tool-card-body">
          <div>参数：</div>
          <pre>{formatArgs(call.args)}</pre>
          {displayContent && (
            <>
              <div className="tool-card-result-label">
                {isTimeout ? '超时信息：' : isError ? '异常信息：' : '执行结果：'}
              </div>
              <pre className="tool-card-result">{displayContent}</pre>
            </>
          )}
        </div>
      )}
    </div>
  )
}
