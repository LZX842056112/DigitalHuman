# 本机照片数字人

给一张照片、一段带人声的视频，就能在浏览器里和它实时对话：用克隆出来的声音回答，口型、表情和头部动作跟着声音走，画面保持照片的原比例。所有模型都在本机推理（声音与画面用显卡，对话模型用 CPU），素材和对话不出网。

这个仓库不是调用云端数字人 API 的壳，而是一条从**素材准备 → 声音克隆 → 动作扩散 → 画面合成 → WebRTC 播放**完整跑通的本地流水线。它关心的是"第一句多久能出声"，而不是"一次能生成多少分钟视频"。

## 能力一览

| 能力 | 现状 | 说明 |
|---|---|---|
| 照片驱动 | 支持 | Ditto 系模型，1024 长边，保留原图比例 |
| 声音克隆 | 支持 | CosyVoice 3 zero-shot，6 秒参考音频，四种语气 |
| 情感表达 | 支持 | `neutral / happy / sad / angry`，同时作用于语音与表情 |
| 对话 | 支持 | 本机 Qwen2.5-3B（GGUF/CPU），JSON 结构化输出 |
| 语音输入 | 支持 | 浏览器录音 + faster-whisper（CPU/int8） |
| 实时传输 | 支持 | WebRTC（VP8 + Opus），25 fps，音画同源时钟 |
| 逐句流式 | 可选 | `runtime.photo_streaming`，默认关闭 |
| 多人并发 | 不支持 | 单进程单会话，同一时间一个人物 |

## 它是怎么跑起来的

四个进程都在 WSL2 的 `DigitalHuman` 发行版里，Windows 侧只有启动脚本和浏览器。

```
浏览器
  │  HTTP :7860  +  WebRTC
  ▼
studio :7860   backend.api.app
  ├─ NDJSON ─► voice :9880      CosyVoice 3  ─┐
  ├─ NDJSON ─► portrait :9881   Ditto        ─┴─► 同一块 NVIDIA GPU
  └─ SSE ────► dialogue :8081   Qwen2.5-3B   ───► CPU（llama.cpp）
```

- **studio** 只做编排：人物资料、任务、会话、WebRTC 协商、前端托管，自己不加载大模型。
- **voice / portrait** 是两个独立 worker，各自持有模型，通过 NDJSON 流式返回分块。
- **dialogue** 是 llama.cpp 的 `llama-server`，用 CPU 跑 3B 量化模型，`-ngl 0`。
- 显存由 `backend/gpu.py` 的租约协调：准备人物时独占，对话时按会话共享，余量不足自动退化为串行。

## 环境要求

- Windows 10/11，开启 WSL2 与虚拟化；安装脚本会导入一个 Ubuntu 22.04 发行版（约 2 GB 下载）。
- NVIDIA 显卡与可用驱动（PyTorch 2.3.0 + CUDA 12.1）。
- 项目盘至少 10 GiB 空闲（模型与缓存放在项目内）；WSL 系统盘另需容纳两套 Python 环境与 llama.cpp（`/opt/digitalhuman`）。
- 运行期不需要联网：`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，模型哈希在安装时校验。

## 安装与启动

在项目根目录用 PowerShell：

```powershell
.\setup.ps1                  # 建 WSL 环境、装依赖、下模型、编译 llama.cpp、构建前端
.\setup.ps1 -EnvironmentOnly # 只装环境，不下权重
.\setup.ps1 -SkipModels      # 环境 + 源码，不下权重
.\start.ps1                  # 启动四个服务，模型在后台预热
.\start.ps1 -StudioOnly      # 只启动 7860（调试前端时用）
.\stop.ps1                   # 停止本项目启动的进程
.\doctor.ps1                 # 只读自检：环境、模型哈希、端口、前端构建戳
```

安装完成后打开 <http://127.0.0.1:7860>。

`start.ps1` 会给子进程注入 `DIGITALHUMAN_WARMUP=1`，voice 与 portrait 启动后立刻加载权重再退回 CPU，所以第一次回答不用再等冷启动（本机实测预热约 65 s / 30 s，分别对应声音与画面）。想等预热完成再测延迟，用：

```powershell
.\scripts\restart-and-warm.ps1   # 停 → 起 → 等健康 → 等预热 → 重建失败人物 → 打印 ready_avatars
```

## 第一次使用

1. **导入人物**：左侧填名称，上传一张清晰单人照片（jpg/png）和一段带人声的视频（mp4/mov/mkv/webm，至少 6 秒清晰语音，音轨不超过 10 分钟）。视频只取音轨，不用画面。
2. **准备人物**：点"准备人物"。任务依次做：抽取参考声音 → 提取照片特征并生成 4 秒待机画面 → 停下等确认。
3. **确认声音**：试听参考音频（脚本自动挑一段 6 秒、噪声低、不爆音的片段）。确认后才允许合成，避免未经试听就用了别人的声音。
4. **开始对话**：点"开始对话"，浏览器建立 WebRTC；第一次要先点一次"开启声音"（浏览器自动播放策略要求用户手势）。
5. 输入文字，或按住麦克风说话。回答生成过程中文字会逐句出现，声音与画面同步播放。

人物状态机：`imported → preparing → voice_pending → ready`，任一步失败进入 `failed`，重新点"准备人物"即可（已完成的照片缓存会复用）。

## 配置

全部在 `configs/app.yaml`，改完重启对应服务生效（渲染提速的三个开关例外，见下）。

| 键 | 默认 | 作用 |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `7860` | studio 监听地址 |
| `portrait.base_url` | `http://127.0.0.1:9881` | 画面 worker |
| `dialogue.base_url` | `http://127.0.0.1:8081/v1` | OpenAI 兼容对话接口 |
| `dialogue.model` | `qwen2.5-3b-instruct` | 与 llama-server 的 `--alias` 一致 |
| `dialogue.max_tokens` | `128` | 回答上限（配合"最多两句"的提示词） |
| `dialogue.context_turns` | `6` | 带多少轮历史 |
| `voice.base_url` | `http://127.0.0.1:9880` | 声音 worker |
| `asr.model` | `models/asr/small` | faster-whisper 模型目录 |
| `asr.device` / `asr.compute_type` | `cpu` / `int8` | 识别放在 CPU，避开显存 |
| `runtime.photo_streaming` | `false` | 音画流式总开关 |
| `runtime.dialogue_streaming` | `true` | 回答按 SSE 流式返回并逐句切分 |
| `runtime.portrait_sampling_steps` | `12` | 动作扩散步数（模型原生 50） |
| `runtime.portrait_frame_batch` | `4` | 离线渲染的批大小，显存不足自动退回逐帧 |
| `runtime.portrait_frame_stride` | `2` | 每 N 帧只算一张再重复写出，25 fps 不变 |
| `runtime.first_clause_chars` | `14` | 首段在逗号处提前切出的字数上限，`0` 关闭 |
| `runtime.idle_timeout_seconds` | `900` | 会话空闲回收 |
| `runtime.disk_reserve_gib` | `10` | 流式生成要求保留的磁盘空间 |

想完全回到原始画质：`portrait_sampling_steps: 50`、`portrait_frame_batch: 1`、`portrait_frame_stride: 1`。这三个值由渲染端每次重读配置文件，改完下一句即生效，不必重启。流式开关只在会话开始时读取，必须重启 studio。

## 目录结构

```
backend/
  api/app.py              studio：人物、任务、会话、WebRTC 协商、静态前端
  dialogue/client.py      对话模型客户端（SSE + 增量 JSON 提取）与语音识别
  voice/                  CosyVoice 引擎、HTTP 服务、客户端
  portrait/               Ditto 引擎、分块流、照片缓存、HTTP 服务、客户端
  transport/
    photo_session.py      一轮对话的编排：切句、派发、打断、计时
    photo_timeline.py     以 PCM 为主时钟的句级播放时间线
    photo_webrtc.py       WebRTC 音视频轨道与麦克风采集
    video_quality.py      固定码率与参数的本地 VP8 编码器
  gpu.py                  跨进程 GPU 租约、显存准入、取消标记
  stream_jobs.py          临时目录、音频 spool、NDJSON 事件校验
  stream_worker.py        同步原生生成器的后台线程与断连保护
  stream_audio.py         连续重采样与整句响度归一
  photo_profiles.py       人物资料、素材校验、资产哈希
  storage.py              原子 JSON、哈希、人物目录
frontend/src/             React 19 + Vite 前端（photo-main.tsx 是主界面）
scripts/                  安装、启动、自检、验收、基准、预热脚本
tests/                    250 个测试函数（约 310 项用例）
configs/                  app.yaml、模型清单与两套 Python 依赖
docs/                     PRD、代码设计、技术讲解
data/                     人物资料与运行产物（不入库）
```

## 接口

studio（`/api` 前缀，`GET /api/health` 可用于探活）：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health`、`/system`、`/avatars`、`/avatars/{id}` | 健康、自检、人物列表与详情 |
| POST | `/avatars/import` | multipart：`name`、`photo`、`voice_video` |
| PUT | `/avatars/{id}/photo`、`/voice-video` | 替换素材（会清空对应校验） |
| PUT | `/avatars/{id}/voice` | `{"confirmed": true}`，确认参考声音 |
| DELETE | `/avatars/{id}` | 删除人物（只删项目内副本） |
| POST | `/avatars/{id}/build` | 开始准备，返回 `job_id` |
| GET | `/jobs/{id}` | 任务进度：`stage` / `progress` / `status` |
| POST | `/sessions` | 新建会话，返回宽高与 25 fps |
| POST | `/sessions/{id}/offer` | WebRTC SDP 交换 |
| DELETE | `/sessions/{id}` | 关闭会话并回收 |
| GET | `/sessions/{id}/metrics` | 帧数、丢帧、队列、分块缓冲、最近一轮计时 |

内部 worker（只监听回环，不对外）：

| 服务 | 接口 |
|---|---|
| voice 9880 | `GET /health`、`POST /tts`（WAV）、`POST /tts/stream`（NDJSON） |
| portrait 9881 | `GET /health`、`POST /prepare`、`POST /render`、`POST /render/stream`（NDJSON） |

浏览器与 studio 之间的控制通道是 WebRTC DataChannel：

- 上行：`chat_text`、`start_recording`、`stop_recording`、`interrupt`
- 下行：`transcript`、`reply_partial`、`status`、`error`、`turn_complete`

## 工具脚本

| 脚本 | 用途 |
|---|---|
| `scripts/acceptance.py` | 真实 WebRTC 连续运行验收，记录帧率、断流、队列与显存峰值 |
| `scripts/benchmark_latency.py` | 测首帧/首声延迟，不代替浏览器可听性验收 |
| `scripts/profile_streaming.py` | 双模型同卡显存探测，产出 `.runtime/stream-admission.json` 供准入使用 |
| `scripts/render-bench.py` | 画面 A/B：改 `steps,batch,stride` 跑固定音频，比较各阶段耗时 |
| `scripts/restart-and-warm.ps1` | 重启并等到预热完成，适合测冷/热启动 |
| `scripts/check_frontend.py` | 校验 `frontend/dist` 与源码摘要一致（`doctor` 也会查） |

## 测试

```bash
# 在 WSL 里，用项目的 portrait 环境
/opt/digitalhuman/portrait/bin/python -m pytest tests -q
```

覆盖素材校验、人物准备、句级时间线、流式分块、GPU 租约、WebRTC 轨道、对话流解析等。以下用例在纯 Windows 上无法收集或运行，需要 WSL 的 `/proc`、`fcntl` 共享锁或 `aiortc` 音视频栈：`test_photo_webrtc`、`test_video_quality`、`test_frame_rate_measurement`、`test_acceptance_guards`，以及 `test_stream_gpu` 中依赖进程签名的部分。

## 数据、隐私与清理

- `data/avatars/<id>/`：原始素材副本、`avatar.json`、参考音频、照片特征缓存、待机视频。
- `data/sessions/<id>/`：每轮的 WAV/MKV 与 `generation.json`（打断标记）。
- `.runtime/`：日志、GPU 锁、服务登记、模型下载缓存。
- 流式临时文件写系统临时目录（`/tmp/digitalhuman-photo/<项目哈希>/...`），句末或打断后删除。
- 删除人物只删项目内副本，原始文件不动；`outputs/` 与 `data/` 都不入库。

## 常见问题

**提示"请先停止准备任务或交互会话"**：同一时间只允许一个会话或一次准备。先断开连接/等任务结束。

**"照片服务：Internal Server Error"**：portrait worker 崩了。看 `.runtime/logs/` 下对应日志；`doctor.ps1` 会指出是模型哈希、环境还是端口问题。

**首句很慢**：确认启动时带了预热（`restart-and-warm.ps1`），并检查 `GET /api/sessions/{id}/metrics` 里的 `last_turn_timing`：`llm_ms` 高是对话模型慢，`tts_first_ms` 高是声音冷启动或显存换入，`portrait_first_ms` 高是画面在等第一批动作。

**声音断续/画面停顿**：`metrics.stream_underruns` 与 `paired_buffer_seconds` 能看出是配音不足还是画面不足；流式模式可先退回 `photo_streaming: false` 确认基线。

**浏览器没有声音**：点一次"开启声音"；`diagnostics=1` 页面能看到音画偏差与解码帧数。

**改了前端没生效**：`cd frontend && npm run build`，再跑 `python -m scripts.check_frontend`；studio 只托管 `frontend/dist`。

**端口被占用**：`.\stop.ps1` 后重试；`doctor.ps1` 会列出被外部进程占用的端口。

## 文档

- [产品需求文档](docs/PRD.md)：目标、用户场景、功能与非功能需求、验收标准。
- [代码设计](docs/architecture.md)：进程与模块、数据模型、时序、时间线、流式链路、GPU 并发、错误处理。
- [项目技术讲解](docs/technical-guide.md)：模型原理、延迟预算、音画同步、调参与排错，适合第一次接手的人按顺序读。
