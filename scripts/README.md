# LangChain Agent 启动脚本

## 📁 目录结构

```
LangChainAgent/
├── LCA.ps1                    # 开发版入口（转发到 scripts/LCA.ps1）
├── LCA-Release.ps1            # 生产版入口（转发到 scripts/LCA-Release.ps1）
└── scripts/
    ├── LCA.ps1                # 开发版核心脚本（Debug 模式，增量编译）
    └── LCA-Release.ps1        # 生产版核心脚本（Release 模式，完全优化）
```

## 🚀 快速开始

### 开发模式（推荐日常使用）

```powershell
# 构建并启动（默认）
.\LCA.ps1

# 等价于
.\LCA.ps1 --all
```

**特点**：
- ⚡ 快速增量编译（10-30 秒）
- 📦 Debug 模式（体积大但构建快）
- 🔄 支持快速迭代开发

### 生产模式（发布打包）

```powershell
# 构建并启动
.\LCA-Release.ps1

# 等价于
.\LCA-Release.ps1 --all
```

**特点**：
- 🐌 完全优化编译（2-5 分钟）
- 📦 Release 模式（体积小，完全优化）
- 🚀 适合最终发布

## 📋 命令参数

### 三种模式

| 参数 | 说明 | 示例 |
|------|------|------|
| `--all` 或无参数 | 构建并运行（默认） | `.\LCA.ps1` |
| `--build` | 仅构建前端和 Tauri | `.\LCA.ps1 --build` |
| `--run` | 仅启动后端和桌面应用 | `.\LCA.ps1 --run` |

### 使用场景

#### 1. 完整启动（默认）
```powershell
.\LCA.ps1           # 构建前端 → 构建 Tauri → 启动后端 → 启动桌面应用
```

#### 2. 只构建，不启动
```powershell
.\LCA.ps1 --build   # 构建前端 + Tauri，但不启动服务
```

适用于：
- CI/CD 构建流程
- 仅需要更新构建产物
- 准备离线部署包

#### 3. 只启动，不构建
```powershell
.\LCA.ps1 --run     # 直接启动后端和桌面应用（使用已有构建）
```

适用于：
- 已经构建过，快速重启
- 调试后端代码（无需重新构建前端）
- 节省构建时间

## 🎯 典型工作流

### 日常开发
```powershell
# 首次运行（完整构建）
.\LCA.ps1

# 修改前端代码后
.\LCA.ps1 --build    # 仅重新构建
.\LCA.ps1 --run      # 重启应用

# 或者一步到位
.\LCA.ps1            # 重新构建并启动
```

### 发布打包
```powershell
# 生成 Release 版本
.\LCA-Release.ps1 --build

# 测试 Release 版本
.\LCA-Release.ps1 --run

# 或者一步到位
.\LCA-Release.ps1
```

## 📊 性能对比

| 模式 | 首次构建 | 增量构建 | 体积 | 性能 |
|------|---------|---------|------|------|
| **开发模式** (`LCA.ps1`) | 5-10 分钟 | ⚡ 10-30 秒 | ~10MB | 未优化 |
| **生产模式** (`LCA-Release.ps1`) | 5-10 分钟 | 🐌 2-5 分钟 | ~3MB | 完全优化 |

## 🛠️ 启动流程

### 开发模式 (`LCA.ps1`)

```
1. 构建 Web 前端
   └─ npm run build（web/dist）

2. 构建 Tauri 桌面应用（Debug 增量）
   └─ cargo build（target/debug/langchainagent.exe）

3. 启动后端服务（3 个标签页）
   ├─ API Server（8001 端口）
   ├─ Scheduler（定时任务）
   └─ Feishu Remote（飞书远程控制）

4. 启动桌面应用
   └─ target/debug/langchainagent.exe
```

### 生产模式 (`LCA-Release.ps1`)

```
1. 构建 Web 前端
   └─ npm run build（web/dist）

2. 构建 Tauri 桌面应用（Release 优化）
   └─ cargo build --release（target/release/langchainagent.exe）

3. 启动后端服务（3 个标签页）
   ├─ API Server（8001 端口）
   ├─ Scheduler（定时任务）
   └─ Feishu Remote（飞书远程控制）

4. 启动桌面应用
   └─ target/release/langchainagent.exe
```

## ⚙️ 环境要求

### 必需
- **Node.js** 18+
- **Python** 3.10+（虚拟环境 `.venv`）
- **Rust** 1.70+（`rustup`）
- **Windows Terminal**（用于多标签后端服务）

### 路径配置
脚本会自动检测并添加：
- `$env:USERPROFILE\AppData\Local\cargo\bin`（Rust cargo）

### 首次运行
```powershell
# 1. 设置 Rust 默认 toolchain
rustup default stable

# 2. 安装 Node 依赖
cd web
npm install

# 3. 安装 Python 依赖
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 4. 运行
.\LCA.ps1
```

## 📝 注意事项

1. **首次构建慢**：Rust 编译所有依赖需要 5-10 分钟，之后增量构建很快
2. **后端独立**：桌面应用不内置后端，需独立运行 `api.server`
3. **端口配置**：确保 `config/server_config.json` 中的端口与前端一致
4. **Windows Terminal**：如果没有 `wt` 命令，脚本会失败（需要安装 Windows Terminal）

## 🔧 故障排查

### 问题：找不到 cargo
```powershell
# 设置 Rust 默认 toolchain
rustup default stable

# 验证
cargo --version
```

### 问题：找不到 langchainagent.exe
```powershell
# 首次需要完整构建
.\LCA.ps1 --build

# 或直接运行（会自动构建）
.\LCA.ps1
```

### 问题：后端连接失败
```powershell
# 1. 检查后端是否运行
Get-Process python

# 2. 检查端口占用
netstat -ano | findstr 8001

# 3. 重新构建前端（确保端口注入正确）
.\LCA.ps1 --build
```

## 📖 更多文档

- [Web 前端文档](../web/README.md)
- [Tauri 桌面端文档](../src-tauri/README.md)
- [API 后端文档](../README.md#api-%E6%9C%8D%E5%8A%A1%E7%AB%AF)
