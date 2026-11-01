# 代码清理摘要

日期：2026-11-01。分支：`feat/local-digital-human`，依据 Git HEAD 读取。

## 完成范围

已静态审查项目自有的后端、前端、脚本、配置和相关测试。未修改第三方源码、模型权重或人物素材。本轮使用精确文本替换，没有整文件覆盖现有源码。命令环境恢复后，已核对 Git 工作区和本轮差异。本次提交只包含本轮清理相关文件，其他已有修改保留在工作区。

## 删除内容

删除声音引擎和照片引擎中未使用的 `time` 导入，删除照片缓存中的 `os` 导入。删除 `test_photo_assets.py` 中重复的局部 `pytest` 导入。未删除整个模块或公开命令入口。

删除 `scripts/download_models.py` 中重复的模型名校验。实际下载函数 `photo_assets.fetch_models()` 仍在任何下载前校验未知模型，保留原错误文本。公开 `download()`、命令参数和返回结果过滤保持不变。

## 合并内容

在 `backend/media.py` 中统一单声道 PCM16 WAV 编码。文件写入和声音 HTTP 响应共用 `write_wav_stream()`。保留原 `write_wav()` 签名及自动创建目录的行为。裁剪范围、32767 缩放、小端编码和 WAV 参数不变。

在 `backend/portrait/assets.py` 中统一照片缓存版本信息。照片引擎、照片服务和人物资产校验共用 `portrait_version()`。缓存目录仍使用原 `photo/model/code/size` 字段、JSON 序列化方式及 24 位散列。资产校验仍执行完整元数据匹配，缓存加载器仍采用原有的预期字段匹配。

在 `scripts/services.py` 中统一 worker 定义和对话服务命令。启动计划与旧服务检测使用相同的命令构造代码。启动顺序、端口、环境、健康检查路径和参数不变。仅启动界面时，不新增模型文件或推理环境可用性检查。

## 保留内容及原因

保留声音和照片 worker 模块。它们通过 Uvicorn 字符串入口启动，不属于死代码。保留旧人物资料转换、声音归一化、路径边界校验和资产哈希校验，它们承担兼容性或数据保护职责。

保留前端依赖、旧任务阶段别名及公开安装工具。静态无引用不足以证明这些内容可以安全删除。旧 `configure_pip.sh` 引用已不存在的约束文件，默认配置也存在未消费项；本轮未修改这些入口或配置契约。

## 回归测试

新增 `tests/test_cleanup_audio.py`，检查文件与内存 WAV 的完整字节一致性、裁剪和量化边界、采样率、声音服务响应头及错误状态。

新增 `tests/test_cleanup_portrait.py`，检查当前目录清单版本、原缓存键和响应格式，以及资产校验对额外元数据的严格拒绝。

新增 `tests/test_cleanup_services.py`，检查完整启动命令和顺序、仅启动界面的独立性，以及现有 worker 的兼容与旧参数拒绝。

独立静态复核未发现其他有证据的新增缺陷，但不能代替运行验证。复核发现：额外缓存元数据会被资产校验拒绝，但真实引擎不一定重建该缓存。底层匹配策略未在本轮改变；新增测试只验证拒绝行为，不声称该场景能够自动恢复。

## 验证状态：部分完成

此前命令工具持续返回 `Workspace still starting`。提交前，命令环境已恢复。本轮 14 个 Python 文件通过 Python 3.10 语法检查；服务命令、启动顺序和 studio-only 计划通过标准库断言检查；本轮已跟踪源码的 `git diff --check` 通过。

当前环境缺少 `pytest`、`av`、`torch`、`fastapi` 等依赖。未运行现有自动测试、前端构建或 Node 测试。不能确认现有功能和既有测试全部通过，也不能将提交视为完成验收。

在完整项目运行环境中，在项目根目录的 PowerShell 中运行：

```powershell
git status --short
git diff --check
wsl -d DigitalHuman -- bash /mnt/d/Projects/3d/DigitalHuman/scripts/run_python.sh /mnt/d/Projects/3d/DigitalHuman -m pytest -q
cd frontend
npm.cmd run build
node --experimental-strip-types --test tests/native-audio-output.test.ts
```

真实模型推理、画面效果和声音效果仍需实际验收。本轮没有执行这些检查。
