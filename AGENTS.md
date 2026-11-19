# Agent Instructions: 项目名称

## 项目总览

这是一个基于LangChain的agent架构的工作目录，包含源文件和README.md

## 技术栈

python 3.14

## 架构约束【OMO重点】

> 编排器Sisyphus、Oracle架构专家会读取这一节做任务拆分评审

1. 目录结构约定：
2. 模块依赖规则：禁止循环导入
3. 引入包安装依赖在对应的项目虚环境中
4. 新增功能需要同步写单元测试
5. 修改功能或文件后同步更新README.md
6. 用中文回复
7. 对话上下文规则：对话轮次超 10 轮自动执行`/compact`压缩上下文

## 代码风格规范

- 类型注解必须完整
- 禁止裸except，捕获具体异常类
- 注释和文档使用中文

## 可执行命令【OMO会自动调用】

- 安装依赖：`uv uv pip install -r requirements.txt`
- 运行服务：`uv run python main.py`
- 单元测试：`uv run pytest tests/ -v`
- 静态检查：`uv run ruff check .`

## OMO多Agent工作约束

1. 修改超过3个文件，必须拆分成多次git commit
2. 重大改动，优先调用Oracle做架构评审，再交给Hephaestus实现
3. 重大改动完成后检查改动文件的同级目录有没有死代码
4. 查询代码/阅读源码任务交给Librarian/Explore，不要让主agent读大量文件
5. 删除文件、git push、数据库变更操作必须向用户确认，不能直接执行

## 文件排除提示

不需要修改：node_modules/, dist/, build/, *.pyc
