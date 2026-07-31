// 主题选择器：弹出面板，展示各主题背景色预览，点击切换
import { useEffect, useRef, useState } from 'react'
import { Check, Palette } from 'lucide-react'
import { THEMES, useStore } from '../store'

export function ThemePicker() {
  const [open, setOpen] = useState(false)
  const theme = useStore((s) => s.theme)
  const setTheme = useStore((s) => s.setTheme)
  const wrapRef = useRef<HTMLDivElement>(null)

  // 当前主题名（用于按钮提示）
  const current = THEMES.find((t) => t.id === theme) ?? THEMES[0]

  // 点击外部关闭
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  return (
    <div ref={wrapRef} style={{ position: 'relative' }}>
      <button
        className="icon-btn"
        onClick={() => setOpen((v) => !v)}
        title={`主题：${current.name}（点击切换背景色）`}
      >
        <Palette size={15} />
      </button>

      {open && (
        <div className="theme-popover">
          <div className="theme-popover-title">选择背景主题</div>
          <div className="theme-grid">
            {THEMES.map((t) => (
              <button
                key={t.id}
                className={`theme-option ${t.id === theme ? 'active' : ''}`}
                onClick={() => {
                  setTheme(t.id)
                  setOpen(false)
                }}
                title={t.name}
              >
                <div className="theme-swatch" style={{ background: t.swatch }}>
                  <div
                    className="theme-swatch-bar"
                    style={{ background: t.swatchAlt, height: '60%' }}
                  />
                  <div
                    className="theme-swatch-bar"
                    style={{ background: t.swatchAlt, height: '40%', opacity: 0.7 }}
                  />
                  <div className="theme-swatch-dot" style={{ background: t.accent }} />
                </div>
                <span className="theme-name">{t.name}</span>
                {t.id === theme && (
                  <span className="theme-check">
                    <Check size={11} strokeWidth={3} />
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
