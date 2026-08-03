# LangChainAgent Web 前端

基于 React + TypeScript + Vite 构建的智能 Agent 对话界面，支持 SSE 流式输出、工具调用展示、多会话管理、多主题切换，并可打包为 Tauri 桌面应用。

## 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| React | 18 | UI 框架 |
| TypeScript | 5.7 | 类型安全 |
| Vite | 6 | 构建工具 + 开发服务器 |
| Zustand | 5 | 全局状态管理 |
| react-markdown | 9 | Markdown 渲染（含 GFM、代码高亮） |
| lucide-react | 0.471 | 图标库 |
| Tauri | 2 | 桌面应用打包（可选） |

## 目录结构

```
web/
├── src/
│   ├── components/          # UI 组件
│   │   ├── ChatView.tsx     # 聊天主视图（消息列表 + 空状态 + 中断卡片）
│   │   ├── InputBar.tsx     # 输入框（自适应高度 + 快捷发送）
│   │   ├── Message.tsx      # 消息气泡（用户/AI + 复制/重新生成）
│   │   ├── Markdown.tsx     # Markdown 渲染器（代码高亮）
│   │   ├── ToolCallCard.tsx # 工具调用卡片（默认折叠，异常自动展开）
│   │   ├── Sidebar.tsx      # 侧边栏（会话列表 + 新建/删除）
│   │   ├── TopBar.tsx       # 顶栏（会话标题 + 提供商/模型切换）
│   │   └── ThemePicker.tsx  # 主题选择器
│   ├── App.tsx              # 应用根组件（布局编排 + 侧边栏宽度拖拽）
│   ├── main.tsx             # 入口文件
│   ├── api.ts               # API 客户端（REST + SSE 流式解析）
│   ├── store.ts             # Zustand 全局状态（会话、消息、流式、主题）
│   ├── types.ts             # 前后端共享类型定义
│   ├── styles.css           # 全局样式 + 主题变量
│   └── vite-env.d.ts        # Vite 环境类型声明
├── index.html               # HTML 模板
├── vite.config.ts           # Vite 配置（代理 + 端口注入）
├── tsconfig.json            # TypeScript 配置
└── package.json
```

## 快速开始

### 环境要求

- Node.js 18+
- npm（或 pnpm）

### 安装依赖

```bash
cd web
npm install
```

### 开发模式

```bash
npm run dev
```

启动后访问 `http://localhost:5173`。Vite 开发服务器会自动将 `/api` 请求代理到后端（默认 `http://127.0.0.1:8000`，端口读自 `config/server_config.json`）。代码改动自动热更新。

### 生产构建

```bash
npm run build
```

输出到 `web/dist/`。后端 `api/server.py` 启动时会自动托管该目录，直接访问 `http://127.0.0.1:8000` 即可使用。

### 预览构建产物

```bash
npm run preview
```

## 后端连接

前端通过 `api.ts` 的 `detectBase()` 自动检测后端地址，优先级：

1. 环境变量 `VITE_API_BASE`（构建期注入，最高优先级）
2. Tauri 环境 → `http://127.0.0.1:<port>/api`（端口由 Vite 从 `config/server_config.json` 注入）
3. 普通 Web 模式 → 相对路径 `/api`（由 Vite 代理或 FastAPI 同源托管）

端口配置统一在项目根目录的 `config/server_config.json`：

```json
{
    "host": "127.0.0.1",
    "port": 8000
}
```

改端口后：开发模式重启 `npm run dev` 即可；生产模式需重新 `npm run build`。

## 核心功能

### 流式对话

- SSE（Server-Sent Events）实时接收 token 增量
- 工具调用过程默认折叠，异常时自动展开高亮
- 看门狗机制：90 秒无响应自动复位，防止流卡死

### 会话管理

- 多会话切换，历史消息持久化（后端 SQLite checkpoint）
- 切换会话时自动滚到底部（`useLayoutEffect` 同步滚动，无滑动动画）

### 提供商/模型切换

- 顶栏下拉框切换 LLM 提供商和模型
- 切换请求实时生效，后端终端会打印切换日志

### 侧边栏宽度调整

- 拖拽侧边栏右缘手柄即可调整宽度，范围限制在 180px ~ 窗口宽度 40%
- 折叠/展开后保留调整后的宽度
- 宽度持久化到 localStorage（key: `sidebar-width`），刷新后保持

### 主题系统

内置 6 套主题（极夜黑、深空蓝、午夜紫、森野绿、护眼米、皓月白），通过 CSS 变量实现，选择持久化到 localStorage。

### 工作流视图

侧边栏左上角提供「对话 | 工作流」切换开关：

- 切到「工作流」时，主内容区从聊天视图切换为工作流视图。
- 视图头部提供工作流名称下拉框（`simple` / `pipline` 等，来源 `GET /api/workflows`），可随时切换并刷新展示对应工作流的节点与链路。
- 工作流模式下侧边栏显示专属工作流会话列表，「新建会话」创建绑定当前工作流的专属会话（会话 ID 为 `{process_type}-workflow-{name}-{uuid}`，以工作流图标 + 徽标区分）。
- 在专属工作流会话中直接输入消息即可，后端自动包装为 `/workflow:{name} <任务>` 命令执行。
- 节点/边结构与状态通过 `GET /api/workflow?name=<name>` 获取，渲染为节点卡片 + mermaid 流程图；工作流结构带进程级缓存，仅点击刷新按钮强制重新拉取。
- 哨兵节点（`__start__`/`__end__`）映射为 `START`/`END` 标签；节点状态目前为静态 `pending`，暂无实时进度跟踪。

#### 相关 API

- `GET /api/workflows`：列出可用工作流名称。
- `POST /api/threads`：`{"type": "workflow", "workflow_name": "simple"}` 创建专属工作流会话；缺省时创建普通会话。
- `GET /api/threads`：会话摘要带 `type`（`chat`/`workflow`），工作流会话额外带 `workflow_name`。
- `POST /api/chat`：在 `type=workflow` 的会话中发送不以 `/` 开头的消息时，自动包装为 `/workflow:{workflow_name} <消息>` 执行。

## Tauri 桌面应用

本项目前端代码同时作为 Tauri 桌面应用的 UI，配置见项目根目录 `src-tauri/`。

### 开发模式

```bash
npm run tauri:dev
```

### 打包

```bash
npm run tauri:build
```

产物为 `src-tauri/target/release/` 下的 exe 和 NSIS 安装包。Tauri 的 CSP 安全策略已放行 `127.0.0.1:*`，改端口无需重新编译 Rust。

## 跨平台兼容性

前端是纯 Web 应用，本身兼容 Windows / macOS / Linux，无需额外修改。

### Linux 开发与构建

- **开发模式**：直接 `npm install && npm run dev`，npm 会自动下载 Linux 版依赖
- **生产构建**：`npm run build` 产物可由后端 `api/server.py` 托管，或用 `npm run preview` 预览
- **后端**：Python 跨平台，按项目根目录 README 的步骤安装依赖即可

### Linux / macOS 打包 Tauri 桌面端

默认 `tauri.conf.json` 的 `bundle.targets` 仅包含 `nsis`（Windows）。其他平台需修改该字段：

- **Linux**：改为 `["deb", "appimage"]`，并安装系统依赖 `libwebkit2gtk-4.0-dev`、`libgtk-3-dev`、`libayatana-appindicator3-dev`、`librsvg2-dev`
- **macOS**：改为 `["dmg", "pkg"]`

具体配置见 `src-tauri/README.md` 的「跨平台构建」章节。

## 常见问题

**切换模型/提供商后没生效？**
强制刷新浏览器（`Ctrl + Shift + R`）清除缓存，或确认后端终端是否打印了 `[切换]` 日志。

**前端发消息无响应？**
检查后端是否正常运行（`http://127.0.0.1:8000/api/health`），以及 LLM 是否返回 429 限流错误。

**端口被占用？**
修改 `config/server_config.json` 的 `port` 字段，重启后端和开发服务器。
