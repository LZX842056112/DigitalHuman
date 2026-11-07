# 2026-11-07 速度问题现场记录

## 1. 新建人物失败：`RuntimeError: 照片服务：Internal Server Error`

`backend/portrait/engine.py` 在流式改造中删除了 `import math`，但 `_frames()` 的待机运镜分支仍调用 `math.sin`。新建人物走 `prepare → _write_idle → _frames(idle=True)`，因此必然抛 `NameError: name 'math' is not defined`，`/prepare` 返回 500，人物停在 `failed`。

已恢复 `math` 导入。同类回归（“删除的导入仍在使用”）已用脚本对 32 个改动文件全量比对，无其他命中。

## 2. 首轮回答慢的主因是冷启动，不是生成

16:12 那次真实对话（`buffered`，人物已缓存）：回答 4.4 s、首句声音 189.1 s、首句画面 69.8 s、首段入队 263.5 s。`voice.log` 显示合成在 16:15:45 才开始，模型构造与前端初始化占了约 180 s。

同一套代码在暖机对照里的服务端中位数（20261105-082835，文本输入）：

| 版本 | 模型加载 | 回答 | 首个声音块 | 首句声音 | 首个画面块 | 首句画面 | 首段就绪 | 整轮 |
|---|---|---|---|---|---|---|---|---|
| baseline（整句） | 每次 | 2.6 s | — | 6.4 s | — | 26.2 s | 36.6 s | 45.7 s |
| candidate（流式） | 每次 | 2.9 s | 3.0 s | 9.8 s | 12.2 s | 27.3 s | 18.6 s | 41.0 s |

结论：暖机后生成阶段本身是几十秒量级，265 s 里约 200 s 是每个进程只该付一次的模型构造。

## 3. 修复：worker 启动即构造模型

- 新增 `backend/warmup.py`：`DIGITALHUMAN_WARMUP=1` 时在后台线程里持服务锁执行一次预热，异常只记录不致命。
- `VoiceEngine.warmup()` = `_load()` + `offload()`；`PhotoEngine.warmup()` = `_activate()` + `offload()`。构造结果常驻，权重仍回到 CPU，避免与对话服务抢显存，单次请求只付约 0.4–1.0 s 的上传。
- 两个 worker 的 `lifespan` 在 `worker_lifespan` 之前启动预热；`scripts/services.py` 为子进程注入 `DIGITALHUMAN_WARMUP=1`。健康检查不要求 `loaded=true`，启动判定不受影响。
- `scripts/restart-and-warm.ps1`：停旧服务 → 启动 → 等 studio 健康 → 等两个 worker 预热完成 → 重建非 `ready` 人物 → 输出 `ready_avatars`。

预期：首轮由 265 s 降到暖机量级（短回答约 40 s，`streaming_parallel` 下首段约 19 s）。流式开关仍为 `false`，验收门禁未变。

## 4. 画面渲染是「实时对话」的真正瓶颈（2026-11-07 17:20 实测）

通过本地 `/render` 接口对同一个人物、同一段 6.0 秒真实语音（`reference.wav`，150 帧）做两次独立测量，结果一致：

| 阶段 | 耗时 | 占整段 | 折算 |
|---|---|---|---|
| 逐帧渲染 `frame_gpu_ms` | 12.47 s | 45% | 83 ms/帧 |
| 动作扩散（50 步） | 5.42 s | 20% | — |
| 音频特征 HuBERT | 3.78 s | 14% | — |
| 画面合成 | 2.98 s | 11% | 20 ms/帧 |
| 编码 | 1.02 s | 4% | 7 ms/帧 |
| 权重上传 + 卸载 | 1.06 s | 4% | — |
| **合计** | **27.6 s** | — | **5.4 fps，约 4.6 倍实时** |

也就是说：**生成 1 秒画面平均要 4.6 秒**。只要这个比值大于 1，流式播放一定会在长句上断流；所以「实时对话」的前提不是打开流式开关，而是先把渲染压到实时以内。

`max_size` 对渲染无效：它只被 `stream_pipeline_*.py` 使用，本项目走的是自建适配层，解码器输出分辨率由模型固定。

### 本轮改动（质量可回退）

| 改动 | 位置 | 效果 |
|---|---|---|
| 动作扩散步数可配置，默认 50 → 12 | `configs/app.yaml: runtime.portrait_sampling_steps` | 动作扩散约省 4 s；动作细节变化，可一行回退 |
| 逐帧渲染批处理，默认 1 → 4 | `runtime.portrait_frame_batch` | 一次前向出 4 帧；失败自动逐帧回退 |
| 每 N 帧只算一张再重复写出 | `runtime.portrait_frame_stride`（默认 2，已启用） | stride=2 时帧计算减半，25 fps 不变但动作略粗 |
| 合成器改为预分配 float32 就地运算 | `PhotoCompositor` | 去掉每帧两次大数组分配，数值逐像素等价 |
| 首段在逗号处切短 | `runtime.first_clause_chars`（默认 14） | 长首句也能先播一小段，不增删任何字 |
| 计时新增 `stitch_ms` | `PhotoEngine._frames` | 把动作拼接从 `frame_gpu_ms` 里拆出来，便于定位 |

渲染参数由 `PhotoEngine.runtime_quality()` 在**每次渲染时**读取 `configs/app.yaml`（按 mtime 缓存），所以调参不需要重启服务，改完下一句即生效。

### 复测方法

```bat
cd D:\Projects\3d\DigitalHuman
python scripts\render-bench.py --presets "50,1,1;12,4,1;12,4,2"
```

脚本用 6 秒固定语音依次渲染每种组合，记录整段耗时、实时倍率、逐阶段耗时和产物路径，结束后自动还原配置。`presets` 每项为 `步数,批大小,帧步长`。

### 仍待解决

- 动作扩散步数取 12 是速度优先的默认值；若动作抖动明显，把 `portrait_sampling_steps` 调回 20–30。
- 真要做到 1 倍实时以内，除本轮改动外还需要：解码器 ONNX/TensorRT、或换更轻的画面模型；这两项都会改动模型运行时，需要单独评估。

## 5. 对话模型改为 SSE 流式、按句切分（2026-11-07 18:30）

### 改动

| 改动 | 位置 | 说明 |
|---|---|---|
| 对话请求带 `stream: true` | `backend/dialogue/client.py` | 仍用 `json_object` + schema，`emotion` 放在 `text` 前面，第一个 token 就能确定语气 |
| 边生成边解析 JSON | `Dialogue.stream()` / `ReplyStream` | 逐字符状态机，`\"`、`\uXXXX`、被拆到两个分块里的中文都能还原；服务端忽略 `stream` 返回整段 JSON 时自动降级 |
| 边生成边切句 | `SentenceSplitter` / `ReplyFeed` | `。！？!?…` 收句；首句在 `first_clause_chars` 内遇到逗号就立刻切出，不等整句结束 |
| 句子一到就开工 | `PhotoSession._pump_reply` | 回答流和 TTS/画面流水线并行；`photo_streaming` 为 `false` 时也生效 |
| 逐字上屏 | `reply_partial` 事件 + `frontend/src/photo-main.tsx` | 前端把同一轮的部分文字替换在同一条气泡里，`transcript` 仍只发一次完整回答 |
| 帧步长 | `configs/app.yaml: runtime.portrait_frame_stride` | 1 → 2，每两帧只算一张再写出，25 fps 与总帧数不变 |

接口约定不变：`/tts`、`/render`、`/tts/stream`、`/render/stream`（仍是内部 NDJSON）和会话 WebSocket 都没有改；关闭开关只需把 `runtime.dialogue_streaming` 写成 `false`。

### 实测（真实 llama-server，127.0.0.1:8081，CPU `-ngl 0`）

| 问题 | 首段就绪 | 整段完成 | 提前 |
|---|---|---|---|
| 房间里的光线和气氛 | 1.33 s | 1.78 s | 0.45 s |
| 周末下雨适合做什么 | 1.27 s | 1.61 s | 0.34 s |
| 为什么规律作息有好处（无逗号可分） | 1.81 s | 1.81 s | 0 s |

首段指已进入播放队列的文字。本机 prefill 约 1.1–1.5 s、解码很快，所以短回答的提前量只有 0.3–0.5 s；回答越长、服务端解码越慢，收益越大。收益同时体现在 TTS/渲染与剩余解码重叠，不再等整段回答。

### 验证

```bat
python -m pytest tests\test_dialogue_stream.py tests\test_sentence_split.py tests\test_photo_session.py tests\test_stream_session.py
```

新增用例覆盖：任意分块边界（含单字符）下的中文/转义还原、服务端忽略流式的降级、半截流报错、逗号处提前切首段、切分不丢字，以及“模型还没写完就已经开始合成第一句”。
