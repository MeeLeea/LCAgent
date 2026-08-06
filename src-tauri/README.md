# LCAgent Tauri 桌面端

基于 Tauri 2 的桌面应用壳层，复用 `web/` 目录的 React + TypeScript 前端，通过 HTTP 调用独立运行的 FastAPI 后端。

## 架构说明

```
src-tauri/          ← 本目录：Rust 壳层 + Tauri 配置
├── src/
│   ├── main.rs     # 入口（Windows 隐藏控制台窗口）
│   └── lib.rs      # 应用构建（加载前端资源，仅作壳层）
├── tauri.conf.json # Tauri 配置（窗口、CSP、构建命令）
├── Cargo.toml      # Rust 依赖
└── build.rs        # Tauri 构建脚本
```

桌面端本身**不包含业务逻辑**，仅提供窗口容器。所有功能由前端 UI + 独立后端进程实现：

- 前端代码在 `web/`，构建产物 `web/dist/` 由 Tauri 打包进应用
- 后端是独立的 Python 进程（`api/server.py`），桌面端通过 HTTP 调用

## 环境要求

- Rust 工具链（rustup + cargo，MSVC Build Tools 2022）
- Node.js 18+（用于构建前端）
- Windows SDK 10.0.26100.0+
- Tauri CLI 2（已包含在 `web/package.json` 的 devDependencies）

### Rust 镜像配置（首次编译加速）

如需配置清华镜像，参考项目根目录的环境变量设置：

```powershell
# CARGO_HOME / RUSTUP_HOME 指向 AppData\Local（避开 OneDrive 硬链接问题）
$env:CARGO_HOME = "$env:LOCALAPPDATA\cargo"
$env:RUSTUP_HOME = "$env:LOCALAPPDATA\rustup"
```

Cargo 镜像在 `config.toml`（清华源），rustup 镜像通过环境变量设置。

## 构建命令

> **重要**：所有命令必须在**项目根目录**（`src-tauri/` 和 `web/` 的父目录）运行，不要在 `web/` 或 `src-tauri/` 目录运行。`tauri.conf.json` 中 `frontendDist` 指向 `../web/dist`、`beforeDevCommand` 使用 `--prefix web`，CLI 需要从根目录定位 `src-tauri/`。

### 开发模式

```bash
# 在项目根目录下，通过 web 子目录的 node_modules 调用
npx --prefix web tauri dev
```

会先启动 Vite 开发服务器，再打开 Tauri 窗口。

### 生产打包

```bash
# 在项目根目录下
npx --prefix web tauri build
```

构建流程：

1. `beforeBuildCommand`（`npm run build --prefix web`）构建前端到 `web/dist/`
2. Rust 编译（首次约 5-6 分钟）
3. 生成 NSIS 安装包

### 产物

- 可执行文件：`src-tauri/target/release/langchainagent.exe`
- 安装包：`src-tauri/target/release/bundle/nsis/` 下的 `.exe`

### 跨平台构建

默认配置仅打包 Windows（NSIS 安装包）。其他平台需修改 `tauri.conf.json` 的 `bundle.targets`：

**Linux（deb / AppImage）：**

```json
"bundle": {
  "targets": ["deb", "appimage"]
}
```

需安装系统依赖：

```bash
sudo apt-get install libwebkit2gtk-4.0-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev
```

**macOS（dmg / pkg）：**

```json
"bundle": {
  "targets": ["dmg", "pkg"]
}
```

> 前端代码（`web/`）本身跨平台，无需修改；仅 Rust 壳层的打包目标与系统依赖不同。构建命令仍是 `npx --prefix web tauri build`（从项目根目录运行）。

## 配置说明

### tauri.conf.json 关键字段

| 字段                         | 说明                                                            |
| ---------------------------- | --------------------------------------------------------------- |
| `build.frontendDist`       | 前端产物路径（`../web/dist`）                                 |
| `build.beforeBuildCommand` | 构建前执行的命令（字符串格式，非对象）                          |
| `app.security.csp`         | 内容安全策略，已放行`127.0.0.1:*` 和 `localhost:*` 任意端口 |
| `bundle.targets`           | 打包目标（`nsis` 即 Windows 安装包）                          |

> **注意**：`beforeBuildCommand` / `beforeDevCommand` 必须用**字符串格式**（如 `"npm run build --prefix web"`），Tauri 不支持对象格式。

### 端口配置

桌面端连接的后端端口由 `config/server_config.json` 控制。Vite 构建时读取该文件并注入前端运行时，CSP 已放行任意端口，因此**改端口无需重新编译 Rust**，只需重新 `npm run build` 前端即可。

### 权限配置

`capabilities/default.json` 定义了桌面端的默认权限集合，当前仅启用 `core:default`。

## 运行时行为

- 启动后自动加载 `web/dist` 的前端资源
- 前端通过 `api.ts` 的 `detectBase()` 检测 Tauri 环境（`__TAURI_INTERNALS__`），自动使用 `http://127.0.0.1:<port>/api`
- 桌面端**不启动后端**，需单独运行 `python -m api.server`
- Release 模式下隐藏控制台窗口（`windows_subsystem = "windows"`）

## 常见问题

**构建报错 `beforeBuildCommand` 格式不对？**
确保用字符串格式，不要用对象格式。

**首次编译很慢？**
Rust 首次编译 Tauri 项目约需 5-6 分钟，属正常现象。配置清华镜像可加速依赖下载。

**应用启动后连不上后端？**

1. 确认后端已运行：`python -m api.server`
2. 确认 `config/server_config.json` 端口与后端一致
3. 重新 `npm run build` 让前端注入正确端口

**端口改了应用连不上？**
CSP 已放行 `127.0.0.1:*`，重新 `npm run build` 即可，无需重新编译 Rust。
