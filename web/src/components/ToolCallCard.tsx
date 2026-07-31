// 工具调用卡片：展示工具名、参数、执行结果（可折叠）
import { useEffect, useState } from 'react'
import { ChevronDown, ChevronRight, Terminal, Check, Loader2, AlertTriangle } from 'lucide-react'
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

export function ToolCallCard({ call, result }: { call: ToolCall; result?: ToolResult }) {
  // 默认折叠：正常情况下只显示一行摘要，突出 AI 文本回复
  const [open, setOpen] = useState(false)
  const done = !!result
  const isError = !!(result && isErrorContent(result.content))

  // 出现异常时自动展开，便于排查
  useEffect(() => {
    if (isError) setOpen(true)
  }, [isError])

  return (
    <div className={`tool-card${isError ? ' tool-card-error' : ''}`}>
      <div className="tool-card-head" onClick={() => setOpen((v) => !v)}>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        {isError ? <AlertTriangle size={13} /> : <Terminal size={13} />}
        <span className="tool-card-name">{call.name}</span>
        <span className="tool-card-status">
          {done ? (
            isError ? (
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
        </div>
      )}
    </div>
  )
}
