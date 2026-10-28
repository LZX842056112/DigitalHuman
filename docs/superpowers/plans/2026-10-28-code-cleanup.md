# 项目代码清理计划与记录

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 清除已验证的冗余实现，保持人物构建、声音、渲染、接口和既有测试。

**Architecture:** 保留现有目录和入口。公共下载归入 `fetch_artifact.py`，归档操作归入 `model_archives.py`，音频表情请求归入一个小型客户端。删除无调用方的内部代码。

**Tech Stack:** Python 3.10、FastAPI、React/TypeScript、PowerShell、WSL。

**Spec:** 用户本轮要求：审查当前分支的整个项目，删除或合并冗余代码，保留核心逻辑与对外接口，并提供简要摘要。

## 约束

- 保留 HTTP 路径、CLI 参数、模型配置、状态格式和下载锁格式。
- 保留 Hugging Face 与 ModelScope 两个下载入口及历史函数导入路径。
- 保留两个下载方各自的解包标记命名，避免重新解包既有模型。
- 不修改人物素材、权重、运行环境或历史验收记录。
- 不改变渲染数学、严格权重加载、音量处理和声音播放。
- 本轮不提交 Git；修改留在当前分支供检查。

## 审查重点

- CLI、FastAPI 装饰器和动态模型补丁不能按静态调用次数误删。
- 模型下载必须保持断点续传、校验和版本锁保护。
- 表情请求必须保持 180/300 秒超时、原请求头及异常传播。
- 状态加载须使用同一份描述文件，避免重复读取跨越版本切换。
- 音频输出格式、视频统计和关闭流程保持原行为。

## 执行步骤

- [x] 核对 152 个跟踪文件的入口、引用及基线测试；独立只读审查。
- [x] 删除无调用方的路径模块、无效释放函数、重复视频计数及无用导入。
- [x] 为归档标记兼容、表情请求和状态快照增加行为测试。
- [x] 合并公共下载、解包及表情请求；诊断 WAV 使用既有写入函数。
- [x] 完整回归、前端构建、CLI 检查和独立差异复审。
- [x] 记录实际删除与合并清单、验证结果和保留理由。

## 基线

分支 `feat/local-digital-human`，起点 `ed35db1`。WSL 测试：150 通过、2 项 Windows 测试跳过；TypeScript 未使用声明检查通过。

## 变更摘要

- 删除 `scripts/common_paths.py` 和唯一的未使用导入。该模块没有实际调用、CLI 或文档入口。
- 删除 `backend/pipeline/build.py` 中无人调用的 `release(renderer)`，以及视频轨道中只写不读的重复计数。对外指标仍使用 `session.frames`。
- 清除校准、素材解析、会话、头部构建和服务管理中的无用导入及局部变量；合并重复的 pytest 忽略规则。
- 公共下载迁入 `scripts/fetch_artifact.py`，消除通用文件和 wheel 下载对 ModelScope 提供方的依赖。旧 `fetch_file` 导入路径继续可用。
- 两个模型提供方共用 `scripts/model_archives.py` 的归档识别、路径检查和解包循环。历史函数导入和两种解包标记均保留，避免重复解包。
- 会话和构建验证共用 `backend/avatar/a2e_client.py`。超时、请求头、WAV 内容、HTTP 错误及安全 NPZ 加载行为保持一致。诊断录音复用既有 `write_wav()`。
- 状态加载和资产列表共用张量路径解析。加载只读取一份描述文件，避免版本切换时混用两个版本的数据。

## 保留理由

安装脚本、两种下载提供方、根目录 PowerShell 入口仍有独立用途或属于既有接口。模型动态补丁、CUDA 回退、配置字段、状态格式和音频播放器均有兼容或业务职责，因此保留。前端未发现需要删除的未使用声明，也未增加新的组件封装。第三方代码、素材、权重和历史输出不属于本轮代码清理范围。

## 验证结果

- 新增 7 个行为用例，已确认清理前失败、清理后通过；38 项针对检查通过。
- WSL 完整回归：157 通过、2 项 Windows 专项跳过。两个 Windows 专项随后在 Windows 上均通过。
- 前端 3 项原生音频测试、生产构建、未使用声明检查均通过。
- 13 个 Python CLI 帮助入口和 15 个 PowerShell 脚本语法检查通过。
- v1/v2 状态测试包各 5 个张量，与 `ed35db1` 加载结果逐位一致；描述文件未被修改。
- 独立差异复审未发现阻止完成的问题。移动的下载和安全解包函数与基线 AST 一致。

测试日志位于 `.runtime/logs/cleanup-full-native.log`。CLI 和状态兼容报告位于 `outputs/cleanup-cli-20261028-165738.json`、`outputs/cleanup-state-compatibility-20261028-170010.json`。
