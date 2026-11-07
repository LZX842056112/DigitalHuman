# 照片数字人 · 项目技术讲解

这是一份"读完能改代码"的讲解，不是 API 手册。建议按章节顺序读：先建立全局时间轴，再分别看三个模型，最后看它们怎么被拼成一条不断流的播放链路。

代码结构见 [代码设计](architecture.md)，需求与验收见 [PRD](PRD.md)。

---

## 1. 一句话链路，以及它为什么慢

一次回答要经过六段：

```
文字/语音 → 对话模型（文字+情感）→ 语音合成 → 动作扩散 → 人脸渲染 → WebRTC 播放
   20ms       1.3–1.9s            3–5s/句     5.3s/6s音频   12.5s/6s音频   实时
```

用一句话概括这台机器的现状：**"想得快、说得慢、画得更慢"**。对话模型 1.3 s 就能给出首句，声音 6 s 左右能出第一句，但画面要 21 s 才能凑齐第一段——瓶颈完全在渲染。

所以整个项目的优化顺序是：

1. 先让回答**逐句**流出来，把"等整段"变成"等一句"（对话 SSE + 切句）。
2. 再让声音**分块**流出来，边合成边入队（CosyVoice 原生 stream）。
3. 最后让画面**分块**流出来，边渲染边配对（Ditto 稳定前缀 + 25 帧块）。
4. 用显存租约保证声音和画面能同时跑，不互相踢下卡。

## 2. 模型一：Ditto，把一张照片变成会说话的人

### 2.1 它到底做了什么

输入是**一张照片 + 一段 16 kHz 单声道音频**，输出是**25 fps 的人脸视频**，尺寸与照片等比（长边 ≤ 1024）。它不生成背景，只生成人脸区域，再贴回原图——所以衣服、背景、光线都保持原样，这也是"像本人"的关键。

六个组件，按数据流排列：

```
照片 ─► AvatarRegistrar ─► f_s（外观特征）、M_c2o（相机矩阵）、x_s_info（初始关键点）
音频 ─► Wav2Feat(HuBERT) ─► 每 40ms 一帧的音频特征
        Audio2Motion(扩散) ─► 动作序列（关键点 + 表情 + 头部姿态）
        MotionStitch ─► 与初始姿态拼接、眨眼、淡入淡出
        WarpF3D ─► 把人脸形变到目标表情
        DecodeF3D ─► 解码成 RGB 人脸
        贴回原图（mask 混合）─► 最终帧
```

### 2.2 三个容易误解的点

**第一，动作不是"跟着音频波形"生成的。** HuBERT 先把音频压成语义特征，扩散模型再在特征序列上生成动作。所以语速、停顿、重音会影响动作，但动作本身是"学出来的说话习惯"，不是物理仿真。

**第二，25 fps 是硬约束。** 音频侧所有窗口都按 25 fps 切：16 kHz 下每帧 640 个采样（40 ms），HuBERT 每次吃 6480 个采样（405 ms）、产出 5 帧、前进 3200 个采样（200 ms）——窗口重叠 50%，保证特征不跳变。

**第三，流式不能"算到哪发到哪"。** 扩散的 overlap 融合和末尾平滑会修正已经算出来的尾部。`MotionStream` 因此保留 `overlap_v2 + 6` 帧不发布，只把"再也不会被改"的前缀送出去：

```python
end = min(len(features), sequence.shape[1] - overlap_v2 - 6)   # 非最终
```

这也是为什么流式画面比整句模式"慢半拍"：它在等一个安全边界。

### 2.3 画质是怎么保住的

- 中间视频用 `libx264rgb` + `crf=0`：无损 RGB，随便 seek，块播完就删。
- 人脸区域用固定 mask 混合，背景不重算，所以不会出现"背景每帧轻微抖动"。
- 传输用 VP8，但关掉了噪声抑制、把最大量化步长压到 32，避免衣服纹理被当成噪声抹平。
- 码率按尺寸给：512 长边 4 Mbps、768 给 6 Mbps、1024 给 8 Mbps。

## 3. 模型二：CosyVoice 3，6 秒克隆一个声音

### 3.1 zero-shot 与 instruct2

项目用的是 `inference_instruct2`：参考音频 + 一句指令 + 待合成文本。指令来自情感映射：

| 情感 | 指令 |
|---|---|
| neutral | 空 |
| happy | 请非常开心地说一句话。 |
| sad | 请非常伤心地说一句话。 |
| angry | 请非常生气地说一句话。 |

同一个参考音频 + 同一个情感，说话人缓存键是固定的：

```
key = sha256(参考音频哈希 + 指令 + 模型修订)
```

第一次 `add_zero_shot_spk` 后把说话人张量存成 `prompt-<key>.safetensors`，之后直接加载。所以换情感会换一份缓存，但同一情感的第二句不用再提取音色。

### 3.2 流式合成的三段

CosyVoice 的流式是"token → flow matching → HiFi-GAN"三段流水：

```
LLM 逐 token 出 ─► flow 逐块生成 mel ─► hift 逐块出波形
```

`stream=True` 时上游会按 `token_hop_len=25` 切块。项目在这里加了一层保护：包住上游的 `llm_job`，记录每个任务用的线程与异常，生成器关闭时先 drain 再 offload，并清掉 `tts_speech_token_dict` 等四张表。原因是上游的后台 LLM 线程不会因为 Python 生成器被 close 就停下，直接搬权重会踩到还在用的 CUDA context。

### 3.3 响度：一个整句一个增益

`SpeechNormalizer` 不是逐样本 AGC。它做三件事：

1. 用前 0.5 s 的有效语音估一个增益：目标 RMS 0.1，峰值不超过 0.891（约 -1 dBFS）。
2. 用 5 ms 前瞻的限幅器压住瞬态，释放时间 50 ms。
3. 整句共用这个增益。

好处是停顿不会被放大、语气动态保留；坏处是如果参考音频前半句特别轻，整句会偏响。参考片段的选择因此很重要：`choose_reference` 会挑语音占比 > 55%、爆音 < 1%、响度接近目标值的 6 秒窗口。

## 4. 模型三：Qwen2.5-3B，以及为什么让 emotion 排在 text 前面

对话模型跑在 llama.cpp 的 `llama-server` 上，Q4_K_M 量化、CPU 推理（`-ngl 0`）、6 线程、上下文 4096。选 3B 是为了不和显卡抢资源——它只要 CPU，画面和声音才能独占 GPU。

输出被约束成 JSON：

```json
{"emotion": "happy", "text": "今天天气很好，适合出去走走。"}
```

**字段顺序是有意的**：llama.cpp 按 schema 顺序生成，`emotion` 在前意味着第一个字段结束（十来个 token）就能确定语气，语音合成可以立刻用正确的指令开始，而不是先按 neutral 合成再返工。

`ReplyStream` 是一个逐字符状态机，专门处理三种"半截"：

- `\"` 被拆成两个分块 → 等齐再解码；
- `\u4e2d` 被拆开 → 等够 4 位十六进制；
- UTF-8 中文被拆开 → 按字符迭代，不按字节。

服务端如果忽略 `stream` 返回整段 JSON，同一个状态机照样能解析——这条降级路径保证换后端不会挂。

## 5. 为什么按"句"切，而不是按"字"

按字合成会让韵律破碎（每个字都要重新起调），按整段又太慢。句子是韵律完整、又能尽早开工的最小单位。

首句还额外做了一件事：**在 14 个字内遇到逗号就提前切**。

```
"今天天气很好，我们一起"  →  "今天天气很好，" + "我们一起去..."
```

只切第一句、只在已有标点处切、切完不增删任何字。这样首段从 20 字降到 8 字，TTS 与渲染的输入直接减半。

实测（本机 CPU 对话模型）：

| 问题 | 首段就绪 | 整段完成 | 提前 |
|---|---|---|---|
| 房间里的光线和气氛 | 1.33 s | 1.78 s | 0.45 s |
| 周末下雨适合做什么 | 1.27 s | 1.61 s | 0.34 s |
| 规律作息（无逗号可分） | 1.81 s | 1.81 s | 0 |

对话模型越快，收益越小；但在"边说边想"的长回答、或者换成更慢的模型时，收益是线性的——因为 TTS 与渲染和剩余解码是重叠的。

## 6. 三条"流"怎么对齐

### 6.1 句子流

`_pump_reply` 把 SSE 的文字喂给 `SentenceSplitter`，每闭合一句就放进 `ReplyFeed`。合成侧 `feed.get(index)` 拿到一句就开始，不用等 `done`。

`photo_streaming=false` 时这条流同样生效——所以即使不开流式音画，回答文字也是逐句出现的。

### 6.2 声音流

`/tts/stream` 每产出一块 PCM 就写进 `AudioSpool`，同时通过 NDJSON 告诉 studio "第 N 块、从第几个采样开始、多少采样、sha256"。studio 读到事件后再去读 `.npy`，校验范围与哈希。

为什么不在 NDJSON 里直接塞 base64 PCM？因为一块可能几百 KB，JSON 膨胀 33%，而且失去"文件落盘即可校验"的能力。

### 6.3 画面流

`/render/stream` 每 25 帧（1 秒）写一个 `video-N.mkv`，事件里带 `chunk_index`、`frame_start`、`frames`。studio 端强制要求 `frame_start` 连续、`1 ≤ frames ≤ 25`——缺一块就报错，不允许"跳过一块继续播"，否则音画会永久错位。

### 6.4 配对规则

48 kHz 下，一帧画面正好对应 1920 个采样：

```
1920 / 48000 = 0.04 s = 1 / 25 fps
```

所以一句可播的长度是 `min(已有采样数, 已有帧数 × 1920)`。这句代码是整个流式链路的地基：

```python
available_samples = min(len(pcm), frames * 1920)
```

只要这个不变，声音和画面就不可能越走越偏。

## 7. 时间线：让声音当主时钟

### 7.1 为什么不用墙上时间

墙上时间会被 NTP 调整、被 GC 暂停、被编码抖动影响。一旦某次抖动，音画就永久差几十毫秒。这里的做法是：

- 音频轨道每 20 ms 读 960 个采样（`read_audio(960)`），时间线把 `cursor` 往前推；
- 画面按 `cursor / 48000` 秒去当前 clip 里取帧；
- RTP 视频时间戳由音频采样换算：`pts = audio_samples * 90000 // 48000`。

声音是"消耗品"，播出去就回不来；画面是"查询品"，可以重复、可以晚到。所以让画面追声音，而不是反过来。

### 7.2 缓冲策略：宁可晚开，不要开一半断

流式首句启动要满足三个条件（`_enough_buffer`）：

1. 最近 3 个画面块的速率 ≥ 1.1× 实时；
2. 声音速率 ≥ 1.1× 实时；
3. 可播数据 ≥ 2 秒。

如果后面还有第二句，还要等第二句也缓冲够——否则第一句 1 秒播完，第二句还在渲染，用户会听到一段尴尬的静音。

中途供不上时，时间线暂停并记一次 `underruns`，界面显示"正在缓冲声音和画面"。这个计数是验收指标：正常对话应该是 0。

### 7.3 丢帧的五个原因

`PhotoVideoTrack.drop_reasons` 会分别计数，排错时非常有用：

| 原因 | 含义 |
|---|---|
| `stale_queued_generation` | 队列里的帧属于被打断的旧轮次 |
| `cleared_pending` | 打断时主动清掉待发帧 |
| `stale_decoded_generation` | 解码完成时轮次已经变了 |
| `pcm_retry` | 音频没前进，视频 pts 会重复，只能重试 |
| `queue_replaced` | 新帧覆盖了还没发出去的旧帧 |

如果 `pcm_retry` 很多，说明音频时钟被卡住（通常是合成或事件循环阻塞）；如果 `queue_replaced` 很多，说明渲染比 25 fps 快，传输端在丢帧——这时候该看码率和网络，而不是模型。

## 8. 两个模型怎么共用一块显卡

### 8.1 先理解"预热"到底做了什么

`warmup()` = `_activate()` + `offload()`。它把权重搬上卡、跑一次 CUDA 初始化、再搬回 CPU。收益有三个：

- CUDA context 建好，第一次推理不用再等 1–2 s；
- cuDNN/cuBLAS 的 kernel 选型与编译完成；
- 显存分配器预热，减少首次分配的碎片。

代价是启动多等 30–65 s（本机 voice 65.2 s、portrait 29.8 s）。所以它默认由 `start.ps1` 打开，但**不阻塞健康检查**——服务先可用，模型在后台热。

### 8.2 三种锁

| 锁 | 何时用 | 为什么 |
|---|---|---|
| 全局独占 | 准备人物、整句渲染 | 这些操作会长时间占满显存 |
| 会话共享 | 流式对话 | 允许同一会话的 voice/portrait 同时算 |
| 串行计算 | 显存不够时 | 一次只让一个模型上卡，靠时间换空间 |

会话租约的细节：先拿全局独占锁（排除准备任务），写 `lease.json`（session + 32 字节 token + pid 签名），再把锁降级成共享。这样本会话的两路 worker 能同时持有共享锁，而别的会话或准备任务进不来。

每次推理前都要重新校验 owner：token 用 `compare_digest`，pid 的 `/proc` 签名必须匹配。进程重启后 PID 可能复用，签名能识别出来。

### 8.3 准入测试不是拍脑袋

`.runtime/stream-admission.json` 由 `scripts/profile_streaming.py` 实测生成：同一会话里让 voice 和 portrait 同时跑几组长短句，采样空闲显存，记录双模型峰值增量。只有满足下面全部条件才算通过：

- 采样期间空闲显存从未低于 1 GiB；
- 记录里的 GPU UUID 与当前卡一致；
- `code_key` 与当前六个源码文件的哈希一致（防止改了代码还用旧结论）；
- `peak_delta + 1024 MiB ≤ 当前空闲`。

不通过就走串行。这不是保守，而是因为 OOM 的代价是整轮重来，比慢更伤体验。

### 8.4 串行回退的边界

只有"本轮还没有任何一句开始播放"才允许重试。一旦用户已经听到声音，显存不足就直接报错，不做"从头再来"——那会让体验更差。

## 9. 性能剖面

以下数据来自本机（单卡、WSL2、2026-11-07），只用于判断量级，不是承诺值。

### 9.1 整句渲染：6.02 s 音频 → 150 帧 / 27.6 s（约 4.6 倍慢于实时）

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 逐帧渲染（stitch + warp + decode） | 12.47 s | 45% |
| 动作扩散 | 5.42 s | 20% |
| HuBERT 特征 | 3.78 s | 14% |
| 合成（贴回原图） | 2.98 s | 11% |
| 编码 | 1.02 s | 4% |
| 权重上卡/换入 | 1.06 s | 4% |

结论很直接：**逐帧渲染是大头，不是扩散**。所以 `frame_batch` 与 `frame_stride` 的收益比降采样步数更明显；再往下要走 ONNX/TensorRT 或换更轻的解码器。

### 9.2 首响（热启动、整句模式）

| 节点 | 时间 |
|---|---|
| 首字 | ~1.4 s |
| 首句文字 | ~1.6 s |
| 首句声音 | 6.6 s |
| 首句画面 | 21.0 s |
| 首段入队 | 34 s（该轮首段更长） |

### 9.3 浏览器音画同步（整句模式，非流式）

| 指标 | 值 |
|---|---|
| 采样 | 数十次/轮 |
| 最大偏差 | 52 ms |
| P95 | 39 ms |

这说明"以 PCM 为主时钟 + 音频换算视频 pts"的方案是对的：即使生成慢，同步也是稳的。流式模式还需要单独测这一项。

## 10. 调参手册

| 参数 | 默认 | 调大 | 调小 | 建议 |
|---|---|---|---|---|
| `portrait_sampling_steps` | 12 | 动作更稳、更慢 | 更快、可能抖 | 12–20；抖就加，卡就减 |
| `portrait_frame_batch` | 4 | 吞吐更高、显存更高 | 更省显存 | 4–8；OOM 会自动退回逐帧 |
| `portrait_frame_stride` | 2 | — | 1 = 全帧计算（最清晰最慢） | 2 是速度/画质折中；3 会明显发飘 |
| `first_clause_chars` | 14 | 首段更长、更自然 | 首段更短、开口更早 | 10–16；0 关闭 |
| `dialogue_streaming` | true | — | 整段生成，便于对比基线 | 除排错外保持 true |
| `photo_streaming` | false | — | — | 验收通过前不要开 |
| `dialogue.max_tokens` | 128 | 回答更长 | 更短 | 配合"最多两句"提示词 |

想回到"原始画质基线"：`steps=50, batch=1, stride=1`。这三个值每次渲染重新读配置，改完下一句生效，不用重启。

## 11. 排错手册

| 症状 | 先看什么 | 常见原因 | 处理 |
|---|---|---|---|
| 首句 30 s+ | `last_turn_timing.portrait_first_ms` | 渲染慢、显存换入换出 | 降 stride/steps；确认没有别的进程占卡 |
| 首声很慢但画面正常 | `tts_first_ms` | 冷启动、参考音频太长 | 用 `restart-and-warm.ps1`；参考 6 s |
| 声音断续 | `stream_underruns`、`paired_buffer_seconds` | 缓冲不足、合成速率 < 1× | 降步数；或退回整句模式 |
| 画面停顿但声音连续 | `drop_reasons.pcm_retry` | 音频时钟被阻塞 | 看事件循环是否有同步阻塞调用 |
| 音画越走越偏 | `diagnostics=1` 的偏差采样 | pts 重复、时间线被打断 | 检查是否发了重复 RTP 时间戳 |
| 显存不足 | `stream_fallback_reason`、准入文件 | 并行余量 < 1 GiB | 自动串行；或重跑 `profile_streaming.py` |
| 准备失败 | `job.error` | 参考音频太短/太吵、照片多张脸 | 换素材；照片只留一张清晰正脸 |
| 回答被截断 | `max_tokens`、`maxLength` | 128 token 到了上限 | 提示词约束"最多两句"或调大 |
| 浏览器没声音 | 音频元素状态 | 自动播放策略 | 点"开启声音" |
| 前端改了不生效 | `check_frontend` | dist 与源码不一致 | `npm run build` |

## 12. 怎么读这份代码（1 小时路线）

**第 1 步（10 分钟）看入口。** `backend/api/app.py` 的 `create_app()`：人物、任务、会话、`/offer` 四条路由，先建立"studio 只编排"的印象。

**第 2 步（15 分钟）看一轮对话。** `photo_session.py` 的 `message()` → `respond()`。重点看三处：`_invalidate()` 怎么做轮次失效、`_pump_reply()` 怎么边收边切句、`_model()` 为什么用 `shield`。

**第 3 步（10 分钟）看播放。** `photo_timeline.py` 的 `read_audio()` 与 `video_frame()`。理解 `cursor` 与 `pts` 的关系，就理解了整个同步方案。

**第 4 步（15 分钟）看模型。** `portrait/engine.py` 的 `render()` 与 `stream_render()` 对比着看：一个整段、一个分块；再看 `portrait/stream.py` 的稳定前缀。

**第 5 步（10 分钟）看并发。** `gpu.py` 的 `SessionGpuLease` 与 `session_operation()`。理解"独占降级成共享"这一步，就明白两路并行是怎么被允许的。

**第 6 步（5 分钟）看协议。** `stream_jobs.py` 的 `iter_worker_events()`，四种校验就是全部约定。

### 一条真实请求的调用链

```
浏览器 send({type:"chat_text"})
  └─ PhotoSession.message
      └─ respond
          ├─ ReplyFeed + SentenceSplitter
          ├─ _pump_reply → Dialogue.stream → llama-server (SSE)
          │     └─ ReplyStream 逐字符解析 → emotion / text / done
          ├─ VoiceClient.stream → voice:9880 /tts/stream
          │     └─ VoiceEngine.stream_synthesize → SpeechNormalizer
          ├─ PortraitClient.stream → portrait:9881 /render/stream
          │     └─ PhotoEngine.stream_render → MotionStream → 25 帧块
          ├─ PhotoTimeline.update_stream(append_audio / append_video)
          └─ PhotoAudioTrack.recv → read_audio(960)
                └─ PhotoVideoTrack._produce → video_frame(cursor/48000)
```

## 13. 术语表

| 术语 | 含义 |
|---|---|
| generation | 一轮对话的编号，打断即 +1；所有迟到结果用它判断是否过期 |
| clip | 时间线里的一个播放单元，整句是 `SentenceClip`，流式是 `StreamingSentence` |
| 稳定前缀 | 动作序列里不会再被 overlap/平滑修正的头部 |
| spool | 把 PCM 分块原子写到磁盘的音频缓冲 |
| 租约（lease） | 跨进程的 GPU 使用权记录，带 session、token、pid 签名 |
| 准入（admission） | 用实测峰值判断能否双模型并行 |
| offload | 把权重搬回 CPU 并清空显存缓存 |
| 配对缓冲 | 已就绪且音画都覆盖到的可播时长 |
| NDJSON | 每行一个 JSON 的流式响应，内部 worker 之间使用 |

---

如果只记一句话：**这个项目的核心不是模型，而是"用 PCM 当主时钟、用 generation 当真理、用租约当交通规则"**。模型可以换，这三条不能丢。
