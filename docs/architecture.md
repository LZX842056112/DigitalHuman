# 照片数字人 · 代码设计

| 项目 | 内容 |
|---|---|
| 版本 | v0.2 |
| 代码规模 | 后端约 4 200 行、前端约 380 行、脚本约 2 000 行 |
| 运行形态 | Windows + WSL2（Ubuntu 22.04）+ 单块 NVIDIA GPU |
| 关联文档 | [PRD](PRD.md) · [技术讲解](technical-guide.md) |

本文只讲"代码为什么这样组织"。模型原理与调参见 [技术讲解](technical-guide.md)。

---

## 1. 设计约束

先把不能改的前提摆出来，后面的结构都是被这几条推出来的：

1. **单机单卡**：声音与画面模型必须共享一块显卡，不能各占一张。
2. **单会话**：一个 studio 进程只服务一个浏览器会话，换人要先断开。
3. **可中断**：用户随时可能打断，正在跑的扩散/合成不能拖住下一轮。
4. **两个原生 worker**：CosyVoice 与 Ditto 都是"同步生成器 + 后台线程"的原生实现，HTTP 断开不等于计算停止。
5. **进程隔离**：模型各自在独立进程里，崩溃不能带走 studio；但这也意味着跨进程状态只能靠文件与锁。
6. **离线**：运行期不访问网络，模型哈希在启动前校验。

## 2. 进程拓扑

```
                     ┌──────────────────────────────┐
 浏览器 ──HTTP──────► │ studio  :7860                │
        ◄─WebRTC──── │ backend.api.app:create_app   │
                     └──┬───────────┬───────────┬───┘
                NDJSON  │           │ NDJSON    │ HTTP(SSE)
                        ▼           ▼           ▼
              ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐
              │ voice :9880 │ │ portrait    │ │ dialogue :8081   │
              │ CosyVoice 3 │ │ :9881 Ditto │ │ llama-server Q4  │
              └──────┬──────┘ └──────┬──────┘ └──────────────────┘
                     └──────┬────────┘
                            ▼
                    GPU（flock 协调）
```

四个进程由 `scripts/services.py` 拉起，登记在 `.runtime/services.json`（pid + `/proc` 启动时刻 + 命令行）。停止时只杀自己登记过的进程，PID 复用不会误杀。

studio 用 `ThreadPoolExecutor(max_workers=1)` 串行跑"准备人物"，用事件循环跑会话；两个 worker 各自持有一把进程内 `threading.Lock`，保证同一时刻只有一个原生推理。

## 3. 模块职责

| 模块 | 职责 | 不做什么 |
|---|---|---|
| `backend/api/app.py` | 人物 CRUD、任务、会话生命周期、WebRTC 协商、静态前端 | 不加载模型 |
| `backend/transport/photo_session.py` | 一轮对话的编排：切句、派发、背压、打断、计时、事件 | 不做推理、不碰编码 |
| `backend/transport/photo_timeline.py` | 句级播放队列、PCM 主时钟、画面选帧、缓冲判定 | 不知道文本与情感 |
| `backend/transport/photo_webrtc.py` | 音视频轨道、麦克风采集、SDP | 不决定"播什么" |
| `backend/transport/video_quality.py` | 本地 VP8 编码参数与发送节拍 | 不改动已安装的 aiortc |
| `backend/voice/*` | 语音合成引擎、HTTP 服务、客户端 | 不关心句子从哪来 |
| `backend/portrait/*` | 照片准备、动作扩散、画面合成、分块流 | 不关心音频怎么合成 |
| `backend/dialogue/client.py` | 对话 SSE、JSON 增量解析、ASR | 不做情感到模型的映射 |
| `backend/gpu.py` | 跨进程 GPU 租约、显存准入、取消标记 | 不管理模型权重 |
| `backend/stream_jobs.py` | 临时目录、音频 spool、NDJSON 校验 | 不做业务判断 |
| `backend/stream_worker.py` | 原生生成器后台线程、断连后的生命周期 | 不解析事件含义 |
| `backend/photo_profiles.py` | 人物资料、素材哈希、资产校验 | 不做推理 |
| `backend/storage.py` | 原子 JSON、哈希、人物目录 | 不做业务 |

分层原则：**编排层不碰张量，推理层不碰 WebSocket**。所有跨进程调用都经过 `client.py`，所有跨进程状态都落在文件上。

## 4. 数据模型

### 4.1 人物资料 `data/avatars/<id>/avatar.json`

```jsonc
{
  "schema_version": 2,
  "mode": "photo",
  "id": "a65c8bdc...",
  "name": "小夏",
  "status": "ready",              // imported | preparing | voice_pending | ready | failed
  "photo": "source/photo/<uuid>/me.jpg",
  "voice_video": "source/voice_video/<uuid>/ref.mp4",
  "sources": { "<相对路径>": "<sha256>" },
  "voice": {
    "confirmed": true,
    "language": "zh",
    "reference": "voice/reference.wav",
    "source_sha256": "...",       // 来源视频哈希，换视频即失效
    "reference_sha256": "...",    // 参考音频哈希，换音频即失效
    "start_seconds": 12.5,
    "duration_seconds": 8,
    "sample_rate": 24000,
    "reference_version": 2,       // 取样流程版本，旧版本会重建一次
    "text": "大家好，今天想和大家聊聊……",  // 参考音频转写，零样本克隆用
    "text_language": "zh",
    "text_model": "faster-whisper-small"
  },
  "validation": { "photo": true, "voice": true },
  "portrait": { "folder": "portrait/<24hex>", "width": 768, "height": 1024, "fps": 25 },
  "artifact_hashes": { "<相对路径>": "<sha256>" },
  "last_job_id": "..."
}
```

设计要点：

- **哈希链**：素材 → 参考音频 → 照片缓存 → 资产清单，逐级用 SHA-256 绑定。任何一环变了，`require_photo_ready()` 都会拒绝建会话。
- **schema 迁移**：旧格式（`body_photos`）在第一次写入前备份为 `avatar.legacy.json`，再迁移成 v2。
- **状态即门禁**：`status` 不只是展示，`/sessions` 与 `/tts` 都直接读它。

### 4.2 照片缓存 `portrait/<key>/`

```
source.json   {schema_version, metadata{photo_sha256, model_revision, code_commit, max_size}, file, sha256}
<uuid>.npz    f_s / M_c2o / eye_open / eye_ball / img_rgb / motion_* / sc
ready.json    {metadata, idle_sha256, width, height, video_format: "h264_rgb_lossless_v1"}
idle.mkv      4 秒待机循环（100 帧）
```

`key = sha256({photo, model_revision, code_commit, max_size})[:24]`。缓存只用 `np.savez_compressed` 存数组，加载时 `allow_pickle=False`，杜绝把缓存当代码执行。

### 4.3 会话 `data/sessions/<sid>/`

| 文件 | 内容 |
|---|---|
| `generation.json` | `{generation, closed}`，跨进程的"这一轮还算不算数" |
| `<gen>_<idx>.wav` / `.mkv` | 整句模式的中间产物，播放完即删除 |

### 4.4 流式临时目录

系统临时目录下 `digitalhuman-photo/<项目路径哈希>/<session>/<gen>_<idx>/`：

| 文件 | 内容 |
|---|---|
| `audio.json` | `{rate, chunks, samples, finished, error, report}` |
| `audio-N.npy` + `audio-N.json` | 一块 PCM 与它的元数据（含 sha256） |
| `video-N.mkv` | 25 帧一块的无损 RGB 视频 |
| `voice-job.json` / `portrait-job.json` | 原生 worker 的存活标记（pid、tid、进程/线程签名、state） |
| `lease.json` | 当前 GPU 会话 owner（见 §9） |

放在系统临时目录而不是项目盘：WSL 的 `/mnt/d` 是 9p 文件系统，小文件写入慢且 fsync 语义弱；临时块不需要持久化。

## 5. 关键流程

### 5.1 准备人物

```
POST /avatars/{id}/build
   └─ 线程池（单线程）
        ├─ prepare_voice: 解码 24 kHz 音轨 → choose_reference(Ns, 说话人向量选段)
        │                 → clean_reference(去直流/淡入淡出/统一电平) → voice/reference.wav
        │                 → faster-whisper 转写 → voice.text
        ├─ portrait.prepare: 人脸检测 → 特征提取 → 缓存 npz → idle.mkv
        ├─ 未确认声音? → status=voice_pending, job=waiting(reference_confirmation) → 返回
        ├─ 已确认 → /tts 合成试听句 → voice/sample.wav
        └─ 资产哈希 → status=ready, job=complete
```

参考音频的取样不是简单找最响的一段：先用相对本轨电平的门限筛掉静音与削波窗，再按 0.25 s 网格铺满全片候选，取说话人向量（campplus）离短名单质心最近的一段——音乐垫底、第二个说话人或偏远麦克风的位置会被这条规则排掉。转写失败只记 `transcription_error`，不影响准备，克隆自动退回指令模式。

"准备"与"会话"互斥：`busy()` 在 `building / sessions / loading / deleting` 任一非空时返回 409。这样避免了"准备到一半就被拿去对话"的半成品状态。

### 5.2 建立会话

```
POST /sessions
  ├─ require_photo_ready: 状态 + 素材哈希 + 资产哈希 + 声音绑定
  ├─ PhotoSession(...) 在 to_thread 里构造（此时没有事件循环）
  ├─ 创建 PhotoTimeline（加载 idle.mkv 到内存，≤100 帧）
  └─ 返回 {session_id, width, height, fps:25}

POST /sessions/{id}/offer
  ├─ install_local_video_encoder（按长边选 512/768/1024 与码率）
  ├─ 添加 PhotoVideoTrack / PhotoAudioTrack
  ├─ DataChannel "control" → session.message
  └─ 只保留 VP8 + RTX
```

会话构造放在线程里，是因为 `PhotoTimeline` 要做视频解码与帧缓存，属于阻塞 IO；构造完再交回事件循环。

### 5.3 一轮文本回答

```
message(chat_text)
  ├─ interrupt(): generation+1，清队列，取消旧任务
  ├─ begin_timing()
  └─ _start(respond(text, generation))

respond
  ├─ ReplyFeed(first_clause_chars)
  ├─ _spawn(_pump_reply)  ── SSE → 句子队列（与合成并行）
  ├─ 分支 A：photo_streaming=false
  │    for 每句: 等队列<2 → TTS → 写 WAV → /render → SentenceClip → timeline.append
  └─ 分支 B：photo_streaming=true
       _ensure_stream_lease → _stream_with_fallback
         for 每句: 并发 voice.stream + portrait.stream，块写入 timeline
  ├─ pump 结束 → 等 timeline.busy 变空
  └─ turn_complete
```

## 6. 会话编排：generation 与三类任务

### 6.1 generation 失效模型

`PhotoSession._generation` 是唯一权威轮次号。`PhotoTimeline` 里也有一份 `_generation`，两者通过 `begin_turn()` 同步。

一次打断做四件事：

1. `_finish_timing(old, 'interrupted')`；
2. `timeline.begin_turn()` 让 generation +1；
3. 写 `generation.json`（跨进程通知 worker 停）；
4. 取消 `_tasks` 里的协程，清空视频待发帧。

之后所有写操作都先问 `_current(generation)`：不是当前轮就直接丢弃。这让"迟到的结果"天然无害，不需要复杂的取消传播。

### 6.2 三类任务，三种回收语义

| 集合 | 内容 | 打断时 | 关闭时 |
|---|---|---|---|
| `_tasks` | 一轮对话的协程（respond、pump） | 立即 cancel | cancel + gather |
| `_model_tasks` | 正在跑的原生任务（TTS/渲染/ASR）与迟到结果的清理任务 | **不取消**，等它自己结束 | 必须等空 |
| `_transport_tasks` | 麦克风读取等传输任务 | 保留 | cancel + gather |

`_model()` 是关键：内部 `create_task` 后用 `asyncio.shield` 等待。调用方被取消时，原生任务继续跑完，然后：

- 如果给了 `discard` 回调（例如 `SentenceClip`），把结果交给回调关掉；
- 否则等任务结束触发 `_cleanup(generation)`。

**为什么不直接 cancel**：CosyVoice 与 Ditto 的生成器背后有非 daemon 线程持有 CUDA context。协程取消不会让它们停下，强行释放锁会导致"权重被搬到 CPU 时后台线程还在算"，直接段错误。所以设计成"逻辑上立刻放弃，物理上等它收尾"。

### 6.3 背压

- 播放队列最多 2 句（`PhotoTimeline.append` 超了抛 `BufferError`）。
- 整句模式在**推理前**检查 `queue_depth >= 2`，避免第三句已经算完却没地方放。
- 流式模式同时限制"活跃句任务 < 2"且"队列深度 < 2"，并且下一句要等上一句的声音结束（`last_voice.is_set()`）。

### 6.4 计时

`_timing` 是一个扁平字典，只在 `generation` 匹配且 `status == 'running'` 时写入，天然防止旧轮次覆盖新轮次。`_first_timing` 只写第一次，`_phase_timing` 可覆盖（用于"完成时刻"）。

## 7. 播放时间线

### 7.1 为什么以 PCM 为主时钟

如果用墙上时间选帧，只要生成或传输抖动一次，音画就会永久错位。这里让音频轨道每 20 ms 读 960 个采样，时间线记录"已经消费到第几个采样"，画面按 `cursor / 48000` 秒去视频里取帧。声音永远连续，画面追声音。

```
PhotoAudioTrack.recv() ──read_audio(960)──► PhotoTimeline
                                              │ cursor += 960
                                              ▼
PhotoVideoTrack._produce() ──video_frame()──► clip.video.at(cursor/48000)
```

### 7.2 两种 clip

| 类型 | 用于 | 数据来源 | 结束条件 |
|---|---|---|---|
| `SentenceClip` | 整句模式 | 完整 WAV + 完整 MKV | 播放完即 `close()` 并删文件 |
| `StreamingSentence` | 流式模式 | 追加的 PCM + 分块 MKV | 声音与画面都 `finished` |

`StreamingSentence` 的关键约束：

- 画面块必须**连续**：`start == frames`，`1 ≤ count ≤ 25`，否则抛错。缺块比卡顿更糟，宁可报错重来。
- 可播采样数 `available_samples = min(len(pcm), frames * 1920)`——48 kHz 下每帧正好 1920 个采样（40 ms），音画按帧对齐。
- 结束时校验 `frames == ceil(samples / 1920)`。

### 7.3 缓冲与欠载

流式播放最怕"刚开始就断"。`_enough_buffer()` 的策略：

- 已经结束：只要还有可播数据就继续。
- 声音已结束、画面未完：按画面速率估算需要提前多少，`target = min(remaining, max(1, remaining*(1-rate)+1))`。
- 都未完：要求最近 3 个画面块的速率与声音速率都 ≥ 1.1× 实时，且可播 ≥ 2 s。

不满足就不启动（首句），或暂停并记一次 `underruns`。首句还有一个额外门槛：如果后面还有第二句，等第二句也缓冲够了再开第一句，避免"第一句 1 秒就播完，第二句还没来"。

### 7.4 读者生命周期

视频解码在事件循环外（`to_thread`），可能在 `close()` 之后才返回。时间线用引用计数 `_readers[clip]` + `_retired` 集合：正在解码的 clip 先标记退休，最后一个读者退出时才真正 close。避免了"解码线程读已关闭的容器"。

## 8. 流式分块协议

### 8.1 NDJSON 事件

worker 返回 `application/x-ndjson`，每行一个 JSON：

```jsonc
{"type":"start","rate":24000,"generation":3,"sentence_index":0}
{"type":"chunk","chunk_index":0,"sample_start":0,"samples":7200,"rate":24000,"generation":3,"sentence_index":0}
{"type":"complete","samples":72000,"chunks":10,"report":{...},"generation":3,"sentence_index":0}
{"type":"error","text":"...","code":"stream_memory","generation":3,"sentence_index":0}
```

客户端 `iter_worker_events()` 做四类校验：

1. **归属**：`generation` 与 `sentence_index` 必须与请求一致。
2. **顺序**：`start` 只能一次；`start` 之前不能有 `chunk`；`complete` 之后不能再有事件。
3. **长度**：单行 ≤ 64 KiB。
4. **完整性**：流结束必须见过 `complete`，否则视为断开。

分块内容本身还有一层校验：`audio-N.json` 里的 `sample_start`、`samples`、`sha256` 必须与 spool 记录一致；`video-N.mkv` 必须已经落盘。

### 8.2 StreamWorker：断连不等于停止

FastAPI 的 `StreamingResponse` 在客户端断开时会取消生成器，但原生生成器还在跑。`StreamWorker` 把生产放在独立非 daemon 线程，通过 `queue.Queue(maxsize=2)` 与 HTTP 消费端解耦：

```
HTTP 协程 ──events()──► queue ──► producer 线程（原生生成器）
     │                    ▲
     └─ 断开 → stopped.set()│
                           └─ 最多再等一个队列槽位
```

同时写 `voice-job.json` / `portrait-job.json`，记录 pid、tid 与 `/proc` 里的进程/线程签名。`wait_worker_done()` 只有在"进程签名与线程签名都还在"时才继续等——进程退出了就不必再等，线程没了说明原生任务已经结束。

### 8.3 音频 spool

`AudioSpool` 用"先写临时文件再 `os.replace`"保证原子性，顺序是：

1. 写 `audio-N.npy`（临时名 + replace）；
2. 写 `audio-N.json`（含 sha256）；
3. 最后更新 `audio.json` 的 `chunks/samples`。

读者只有在 `audio.json` 计数增加后才会去找 `audio-N.json`，所以不会读到半个文件。

## 9. GPU 并发模型

三个锁，职责不同：

| 锁 | 位置 | 模式 | 保护什么 |
|---|---|---|---|
| 全局操作锁 | `.runtime/gpu-operation.lock` | 独占 `flock`（Windows 用 `msvcrt`，不支持共享） | 准备人物、整句渲染、非流式会话 |
| 流式会话锁 | `<tmp>/.../lease.json` + 共享 `flock` | 共享 | 同一会话内 voice/portrait 两路并行 |
| 串行计算锁 | `<tmp>/.../compute.lock` | 独占 | 串行模式下一次只允许一个模型算 |

### 9.1 会话租约

```
SessionGpuLease.acquire()
  ├─ 独占 flock(.runtime/gpu-operation.lock)     ← 排除准备/离线渲染
  ├─ 校验 /proc/<pid> 的进程签名
  ├─ 写 lease.json {session_id, token(32B), pid, signature, concurrent}
  └─ 降级为共享 flock                            ← 允许本会话内两路并行
```

`session_operation()` 每次推理前都重新校验 owner（token 用 `compare_digest`，pid 签名必须匹配），防止别的进程拿着过期 token 混进来。

### 9.2 准入与余量

- `stream_admission()` 读 `.runtime/stream-admission.json`：必须 `status=passed`、`code_key` 与当前六个源码文件一致、`qualification_version=2`、GPU UUID 一致，并且当前空闲显存 ≥ 实测峰值 + 1024 MiB。
- 通过 → `concurrent=True`，两路共享 GPU；不通过 → 串行。
- 运行中每次 `check_stream_headroom()` 再看一眼：并行模式下空闲显存 < 1024 MiB 就抛 `StreamMemoryUnavailable`。

### 9.3 串行回退

```
_stream_sentence 抛 StreamMemoryUnavailable
  ├─ 记录 _memory_failures[gen] = {retry, waiter, generation}
  ├─ begin_turn()（丢弃本轮）
  ├─ 如果"本轮还没有任何一句开始播放" → retry=True
  └─ _stream_with_fallback 捕获后：
        set_concurrent(False) → 释放 waiter → 用新 generation 重跑
        界面提示"显存余量不足，正在使用串行方式重新准备声音和画面"
```

`retry` 的条件很严：只要有任意一句已经 `started`，就不重试，直接报错。用户不能在听到一半时被"从头再来"。

## 10. 画面引擎

### 10.1 装配

`PhotoEngine._load()` 只做一次：读上游 pkl 配置（把 numpy 2 的 `numpy._core` 映射回 `numpy.core`），强制人脸检测/关键点/HuBERT 走 CPU，实例化六个组件：

```
AvatarRegistrar  → 外观特征 f_s、相机矩阵 M_c2o、x_s_info
Wav2Feat         → HuBERT 特征
Audio2Motion     → 扩散生成动作序列
MotionStitch     → 关键点拼接、眨眼、淡入淡出
WarpF3D          → 形变到目标表情
DecodeF3D        → 解码成人脸 RGB
```

权重用 `network.cuda().eval()` 上卡，`offload()` 再搬回 CPU 并 `empty_cache()`。**预热 = 上卡一次再搬回**，目的是把 CUDA context、cuDNN 句柄、kernel 编译都提前做掉，而不是常驻显存。

### 10.2 两条渲染路径

| | `render()`（整句） | `stream_render()`（分块） |
|---|---|---|
| 动作来源 | 整段音频一次算完 | `_stream_motions()` 边收 PCM 边算 |
| 出图 | `_frames()`，可 batch/stride | 逐帧，3 帧前瞻后立刻出块 |
| 编码 | 一个完整 MKV | 每 25 帧一个 MKV |
| 结束校验 | 帧数 = `ceil(samples*25/rate)` | 同上，且 `frame_start` 连续 |
| 提速开关 | 生效 | **不生效**（块要尽快发布，批量会拖后） |

`_frames()` 的 stride 语义要特别注意：**运动与眨眼仍然逐帧推进**，只有"出图"按 stride 抽样，抽到的帧重复写出补满 25 fps。所以动作时间轴不会变短，只是视觉上略粗。

### 10.3 稳定前缀

流式动作不能"算到哪发到哪"，因为扩散的 overlap 融合与平滑会修正尾部。`MotionStream` 保留 `overlap_v2 + 6` 帧不发布，并且只对"再也不会被修正"的前缀调用一次原始平滑器：

```
发布上界 = min(已算帧数, 序列长度 - overlap_v2 - 6)
```

`FeatureStream` 同理：HuBERT 每次吃 6480 个采样（405 ms）、产出 5 帧、前进 3200 个采样（200 ms），窗口重叠 50%，保证特征连续。

### 10.4 合成与编码

`PhotoCompositor` 把"固定背景 × (1-α)"预计算一次，每帧只做：形变人脸 → 乘 α → 加背景 → clip → uint8。相比每帧重算两张 float 图，省掉约 40% 的合成时间，且像素结果与旧实现一致（有测试对比）。

中间编码用 `libx264rgb`、`crf=0`、`ultrafast`、`zerolatency`：无损、快、能随机 seek。体积大没关系，块播完就删。

## 11. 声音引擎

- 加载时替换 `cosyvoice.load_hyperpyyaml`，剥掉训练用的 `data_pipeline/train_conf` 节点，只保留推理图。
- `wetext` 的模型下载被重定向到本地目录，运行期不联网。
- 说话人缓存：`prompt-<sha256(参考音频+情感指令+模型修订)>.safetensors`，第一次 `add_zero_shot_spk`，之后直接 `load_file`。
- `_native_chunks()` 包住上游的 `llm_job`，记录每个任务的线程与异常；生成器被关闭时先 drain 再 offload，并清理 `tts_speech_token_dict` 等四张表，防止内存缓慢增长。
- `SpeechNormalizer`：用前 0.5 s 估一个整句增益（目标 RMS 0.1、峰值 ≤ 0.891），再过 5 ms 前瞻的限幅器。**不逐样本 AGC**，否则停顿会被放大、音量动态被抹平。

## 12. 对话客户端

`ReplyStream` 是一个逐字符状态机：`seek → key → colon → value → string`。它解决三个现实问题：

1. **半截转义**：`\"`、`\uXXXX` 可能被拆到两个 SSE 分块里，必须等齐再解码。
2. **半截中文**：UTF-8 字节可能被切断，按字符迭代天然规避。
3. **只要 text**：schema 里 `emotion` 在前，收到第一个字段就能确定语气，不必等 `text`。

服务端忽略 `stream` 返回整段 JSON 时，走降级分支：整段喂给同一个状态机，行为一致。

## 13. 传输层

- 只保留 VP8 + RTX，避免协商出 H.264 后浏览器行为不一致。
- 码率按编码长边：512→4 Mbps、768→6 Mbps、1024→8 Mbps；关掉 VP8 噪声抑制（会把衣服纹理当噪声抹掉），`rc_max_quantizer=32`。
- `send_at` 让编码线程按帧节拍等待，避免编码抖动导致音频读取得太早、画面超前。
- 视频 pts 由音频采样位置换算：`pts = audio_samples * 90000 // 48000`。若不大于上一帧，丢帧重试，**绝不发重复时间戳**（接收端会把两张图合并）。
- `clock.install_rtc_clock()` 用单调时钟重建 aiortc 的 UTC 时间，避免系统对时导致 RTP 时间跳变。

## 14. 前端

`photo-main.tsx` 一个组件承担全部界面，状态分四组：

| 组 | 内容 |
|---|---|
| 人物 | 列表、当前人物、任务进度、素材替换、删除 |
| 连接 | `connecting/connected`、PeerConnection、DataChannel、麦克风授权 |
| 对话 | 消息列表、输入框、录音、生成中、打断 |
| 诊断 | `?diagnostics=1` 时的 RTC 采样、音画偏差、本轮计时 |

消息渲染有一条容易踩的规则：`reply_partial` 与最终 `transcript` 必须落在**同一条气泡**里。实现是"如果最后一条是 partial 的 assistant，就替换它"，而不是追加。

## 15. 错误处理与资源回收

| 场景 | 处理 |
|---|---|
| 素材/资产哈希不符 | `NotReady` → 409，提示重新准备 |
| 上传越界/格式错 | 400/413，临时目录 `finally` 删除 |
| worker 500 | 客户端抛 `RuntimeError('照片服务：...')`，界面展示原文前 500 字 |
| 显存不足 | `StreamMemoryUnavailable` → 串行回退或明确报错 |
| 打断 | generation 失效 + 取消协程 + 清队列 + 删旧文件 |
| 关闭会话 | 依次：打断 → revoke 租约 → 取消任务 → 停轨道 → 关 peer → 等原生任务 → 清文件 → 释放租约 |
| Windows 文件占用 | `_cleanup` 遇到 `PermissionError` 先记账，等模型任务结束再重试 |

关闭顺序不能换：**先 revoke 再 release**，否则别的进程可能在本进程还持有共享锁时拿到 owner 记录；**先等 `_model_tasks` 为空再删文件**，否则 worker 会把文件重新写出来。

## 16. 性能设计

- 预热：worker 启动即上卡一次再退回 CPU，代价是启动多 30–65 s，收益是首轮少 20 s 以上。
- 动作扩散步数 12（原生 50）：质量换速度的主要杠杆。
- 出图批 4：把 warp/decode 的 kernel 启动开销摊薄；失败自动退回逐帧。
- stride 2：出图量减半，25 fps 与总帧数不变。
- 首段逗号切分：让第一段更早进入队列，不改变文字。
- 合成器原地运算：少两次全图分配。
- 分块编码 25 帧一块：编码与传输重叠，块大小等于 1 秒。

## 17. 扩展点

| 想做的事 | 改哪里 |
|---|---|
| 换画面模型 | `backend/portrait/engine.py` 的装配与 `_frames`；保持 `photo_profiles` 的缓存契约 |
| 换 TTS | `backend/voice/engine.py`，实现 `synthesize` 与 `stream_synthesize` |
| 换对话模型 | 只改 `configs/app.yaml` 的 `dialogue.*`，需兼容 `response_format.schema` |
| 加语言 | `emotion.py` 的指令文案 + ASR 语言 + 提示词 |
| 多人排队 | 把 `sessions` 从"单会话"改成队列，并在 GPU 层按会话发租约 |
| 换传输 | 保留 `PhotoTimeline` 接口，替换 `photo_webrtc.py` |

## 18. 测试策略

268 个测试函数（约 327 项用例，含参数化），分四层：

1. **纯函数**：切句、情感映射、响度归一、分块校验。
2. **契约测试**：客户端与 worker 的事件顺序、错误码、缓存身份。
3. **集成**：用假模型跑完整一轮（含打断、并发回退、清理）。
4. **环境相关**：GPU 租约、WebRTC 轨道、帧率测量——必须在 WSL 内跑，Windows 上会因缺 `/proc`、`fcntl` 共享锁或 `aiortc` 而跳过。

约定：测试不依赖真实模型权重；需要真实推理的验证放在 `scripts/acceptance.py`、`benchmark_latency.py`、`profile_streaming.py`、`render-bench.py`，产物写 `outputs/`，不入库。
