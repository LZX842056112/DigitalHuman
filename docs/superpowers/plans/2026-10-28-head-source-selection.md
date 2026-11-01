# 头像来源选择 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Execute tasks in this session and request one final independent review.

**Goal:** 分别生成视频与每张照片的三维头像，保存用户选择，并用于后续人物构建。

**Architecture:** 候选保存与身体无关的高斯参数及三视图。选择记录写入人物清单。完整构建将所选参数注册到身体，复用既有头部加载器。GPU 操作串行执行。

**Tech Stack:** Python 3.10、FastAPI、PyTorch 2.3、GAGAvatar、safetensors、React/TypeScript。

**Spec:** `docs/superpowers/specs/2026-10-28-head-source-selection-design.md`，用户已确认。

## Global Constraints

本机运行；保留现有接口和现有人物；保留身体和声音来源；不混合头像来源；不修改 Silent 电源设置；不访问未授权浏览器；本轮修改保留供用户审阅，不自动提交。

## Review Focus

- 无视频人脸时仍可生成照片候选；单张照片失败不影响其他候选。
- 候选参数及源文件被修改后不能继续选择或构建。
- 改选后禁止使用旧头像启动；重新生成失败时保留已生效人物。
- 构建或交互中禁止改选；跨进程使用 GPU 锁。
- 重启恢复选择；旧人物和命令行入口兼容。

## Task 1: 候选存储、来源收集与选择

**Files:** 新增 `backend/avatar/head_candidates.py`、`backend/pipeline/head_sources.py`、`tests/test_head_candidates.py`。修改 `backend/pipeline/media.py`、`backend/storage.py`。

**Interfaces:** `read_catalog(folder) -> dict`；`save_candidate(folder, candidate_id, tensors, metadata, directory) -> dict`；`load_candidate(folder, entry) -> (tensors, metadata)`；`select_candidate(store, avatar_id, candidate_id) -> dict`；`collect_sources(folder, manifest, directory) -> list[dict]`。清单文件 `head_candidates.json`。人物选择字段 `head_choice`，新导入字段 `head_selection_required`。

- [x] 先测试多来源独立、参数/来源校验、目录边界、持久化选择、同一选择幂等、改选使 ready 失效。
- [x] 实现安全的候选存储、照片检测复用、视频清晰帧选择，及旧人物兼容。
- [x] 运行对应测试，记录 RED 和 GREEN。

## Task 2: 三维候选重建与选定头像绑定

**Files:** 新增 `backend/avatar/head_builder.py`、`backend/pipeline/heads.py`、`tests/test_head_selection_build.py`。修改 `scripts/build_head.py`、`backend/pipeline/build.py`、`backend/avatar/head_reference.py`。

**Interfaces:** `encode_candidate(root, image_path, output_folder, metadata) -> dict`，保存后重载并渲染三视图；`prepare_candidates(store, avatar_id, job_id)`，原子替换候选清单；`bind_selected_head(root, folder, choice, renderer)`，仅写头部 sidecar；`build_signatures` 将头像选择加入完整签名，身体签名不包含选择。

- [x] 先测试单项失败隔离、重新生成失败保留旧清单、选择前禁止构建、所选源独占、切换不重复重建身体、头像资产进入最终校验。
- [x] 抽出已有头部编码与注册逻辑，增加真实三视图，接入构建；选定流程不执行旧混合校准。
- [x] CLI 支持所选头像及自动来源，取消固定第4秒。
- [x] 运行相应测试。

## Task 3: API 与选择界面

**Files:** 修改 `backend/api/app.py`、`frontend/src/main.tsx`、`frontend/src/style.css`、`scripts/build_avatar.py`、`README.md`。新增 `tests/test_head_candidates_api.py`。

**Interfaces:** `GET /api/avatars/{id}/heads`；`POST /api/avatars/{id}/heads/prepare` 返回 `job_id`；`PUT /api/avatars/{id}/heads/selection` 接收 `{candidate_id}`。

- [x] 先测未完成候选拒绝选择、任务/会话/进程锁忙碌保护、新导入要求选择、旧人物不被强制失效。
- [x] 加入候选来源、参考图、三视图、失败原因、已选标识和使用按钮。选择完成后才开放构建按钮。
- [x] 完整后端测试、Windows 两项测试、原生音频测试、前端生产编译。

## Task 4: 实际素材与独立检查

- [x] GPU 空闲时，通过接口为当前人物生成视频及照片候选。
- [ ] 验证所选头像保存、身体绑定、卸载/重载；核对身体与声音来源。
- [ ] 运行一次完整分支审查，修复发现的问题并重跑受影响测试。
- [ ] 记录未验证项目，提供操作说明。不宣称人工相似度或浏览器体验已验收。
