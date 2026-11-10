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
| `runtime.voice_reference_seconds` | `8` | 参考声音取样长度（秒），越长越像但合成略慢 |
| `runtime.voice_clone_mode` | `hybrid` | `hybrid` 平语零样本、有情绪走指令；`zero_shot` 全程零样本；`instruct` 全程指令 |
| `runtime.tone_voice` | `true` | 非"自然"语气是否也让声音生效；关掉则语气只改文案 |
| `runtime.max_photo_variants` | `8` | 每个数字人最多几套造型（不同穿搭/场景照片） |
| `runtime.portrait_seed` | `null` | 仅诊断用：固定动作扩散噪声，便于同机 A/B |
| `runtime.idle_timeout_seconds` | `900` | 会话空闲回收 |
| `runtime.disk_reserve_gib` | `10` | 流式生成要求保留的磁盘空间 |

想完全回到原始画质：`portrait_sampling_steps: 50`、`portrait_frame_batch: 1`、`portrait_frame_stride: 1`。这三个值由渲染端每次重读配置文件，改完下一句即生效，不必重启。流式开关只在会话开始时读取，必须重启 studio。

声音更像本人有三步：

1. **重新取样**：`voice_reference_seconds`（默认 8）决定参考片段长度。取样会按本轨电平筛窗、按说话人向量选段、去直流并统一电平，参考音频按 24 kHz 保存。取样式一旦变化（`reference_version` 不匹配），旧参考音频会自动重建一次。
2. **选克隆方式**：`voice_clone_mode` 默认 `hybrid`——平语走零样本，需要参考音频的转写；准备人物时本地 faster-whisper 自动生成并写进 `voice.text`，没有转写就退回指令模式。这个值在声音服务启动时读取，改完要重启 voice（`.\scripts\restart-and-warm.ps1` 会一起重启）。
3. **量化验证**：`.\scripts\voice-similarity.ps1 -AvatarId <id> -Diagnose`。默认比对人物目录里的 `voice/sample.wav`；加 `-Text "..."` 会先让声音服务合成再比对。它用模型自己的 campplus 编码给出参考音频与合成音频的余弦相似度。

## 个性化：语气、表情与造型

三个维度互相独立，都是"每个人物一份"的设置，存在 `avatar.json`：

| 设置 | 取值 | 默认 | 对最终呈现的影响 |
|---|---|---|---|
| 语气 | 自然 / 亲切 / 专业 / 活泼 | 自然 | 只改"怎么说"：措辞（句长、称呼、语气词、感叹号）以及平语句的发音指令；不改长相、嘴型、帧率 |
| 表情 | 自然 / 微笑 / 严肃 / 惊讶 | 自然 | 只在平语回答时改脸：Ditto 情绪软标签 + 表情动态幅度；有情绪的回答按情绪演 |
| 造型 | 1–8 套 | 1 套 | 每套一份画面缓存。可以上传另一张照片，也可以直接对当前照片换装（换色 / 换款式） |

保存与生效：`PUT /api/avatars/{id}/settings`（或会话内的 `{"type":"settings"}` 消息）写回清单，下一句生效；切换造型必须断开连接，因为会话建立时已固定画面尺寸与待机视频，界面会提示"下次连接生效"。

为什么换装必须改照片：渲染只重绘 512×512 的头部裁剪区，再经软边掩膜贴回原图；衣领以下、背景和光线都是源照片的像素，模型不会重绘。所以换装不是渲染参数，而是先把源照片里的衣服换掉，生成一张同人新图，再交给准备流程。

界面只做一件事：**上传一张服装图片，按图给照片里的人物换装**（左侧「服装」→ 上传图片 → 可填一句描述 → 开始换装）。先填描述、再选图片、最后点开始换装；描述是选填的，但填了更准。

原理：先用本地 Segformer（`models/outfit/clothes-seg`）分出发型/上装得到软边掩膜，再用 SD1.5 inpainting（`models/outfit/inpaint`）重绘；服装图片由 IP-Adapter（`models/outfit/ip-adapter`）注入，提供颜色、材质和大致力。中文描述先由本地对话模型翻成英文名词短语，因为 SD1.5 只认英文，且"外套 / 衬衫 / 毛衣"这类名词决定模型画出的轮廓。约 40 秒，脸和背景逐像素不变。

旧的「换色」（8 个预设色）和「换款式」（6 个预设）已从界面移除，仍保留在 `POST /api/avatars/{id}/outfit` 接口里，需要时可脚本调用。

两种结果都会作为一套新造型加入 `variants`，需要点"准备人物"生成缓存后才能切换。当前换款式是实验性能力：针织衫、大衣一类的纯色衣物效果稳定，衬衫的领口偶尔偏薄，源照片自带的水印若落在衣服上会被重绘成乱码（建议用无水印素材）。

"自然"是最像参考的选择：语气为自然时平语句仍走零样本克隆；选其他语气且 `tone_voice: true` 时，平语句改走 instruct2，用一点音色相似度换取可听出的语气差异。表情同理——有情绪的回答完全忽略用户预设，避免"一直笑"地说严肃内容。

## 目录结构

```
backend/
  api/app.py              studio：人物、任务、会话、WebRTC 协商、静态前端
  dialogue/client.py      对话模型客户端（SSE + 增量 JSON 提取）与语音识别
  voice/                  CosyVoice 引擎、克隆路径、相似度度量、HTTP 服务、客户端
  style.py                语气与表情预设（唯一的取值与默认值来源）
  portrait/               Ditto 引擎、分块流、照片缓存、HTTP 服务、客户端
  asr.py                  参考音频转写（本地 faster-whisper，CPU）
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
backend/outfit.py         换装：服装分割 + Lab 换色 + 本地修补模型
models/outfit/            换装用的分割模型与修补模型（可选）
scripts/                  安装、启动、自检、验收、基准、预热、相似度、表情标定脚本
tests/                    304 个测试函数（约 374 项用例）
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
| PUT | `/avatars/{id}/settings` | `{"tone": "...", "expression": "..."}`，可只传一个字段 |
| POST | `/avatars/{id}/variants` / PUT、DELETE `/variants/{vid}` | 新增、替换、删除一套造型（照片） |
| POST | `/avatars/{id}/outfit` | 预设换色 / 预设款式（界面已隐藏，仍可由脚本调用） |
| POST | `/avatars/{id}/outfit/wear` | multipart：`garment`（衣服图片）+ `prompt`（中文描述）+ `scale`，按图换装 |
| GET | `/outfit/options` | 可用的换装颜色与款式，以及本机是否装了生成模型 |
| PUT | `/avatars/{id}/variants/{vid}/active` | 切换使用中的造型，要求该套已准备且当前无会话 |
| POST | `/avatars/{id}/portrait/gc` | 回收没有任何造型引用的画面缓存 |
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
| `scripts/fetch_outfit_model.py` | 下载可选的换装模型（分割 110 MB + 修补 2.6 GB + 参考图适配器 2.5 GB） |
| `scripts/expression_calibrate.py` | 为一个人物渲染四种表情，输出差异区域、帧数与边缘一致性报告 |
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

**换装和换照片有什么区别**：换装改的是同一张照片里的衣服，脸和背景不动；换照片是另一张素材。两者都会成为一套新造型，都要准备后才能切用。

**换款式生成的图有点假**：当前用的是 SD1.5 修补模型，纯色针织/大衣最稳，衬衫领口偶尔偏薄；提高 `steps`、换更好的修补模型（把权重放进 `models/outfit/inpaint`）都能改善。换色路径没有生成环节，结果与照片一致。

**切换造型后画面没变**：造型切换要断开连接再连；若该套还没准备，接口会返回 409"请先准备该造型"。

**表情看起来没区别**：预设只在平语回答生效，有情绪的回答会按情绪演；本机实测变化像素 8–18%、平均差 6–10/255；换人物后先用 `expression_calibrate.py` 复核一次。

**改了前端没生效**：`cd frontend && npm run build`，再跑 `python -m scripts.check_frontend`；studio 只托管 `frontend/dist`。

**端口被占用**：`.\stop.ps1` 后重试；`doctor.ps1` 会列出被外部进程占用的端口。

## 文档

- [产品需求文档](docs/PRD.md)：目标、用户场景、功能与非功能需求、验收标准。
- [代码设计](docs/architecture.md)：进程与模块、数据模型、时序、时间线、流式链路、GPU 并发、错误处理。
- [项目技术讲解](docs/technical-guide.md)：模型原理、延迟预算、音画同步、调参与排错，适合第一次接手的人按顺序读。
