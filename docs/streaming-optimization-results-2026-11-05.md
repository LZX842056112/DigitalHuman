# 数字人流式优化验收记录

本轮以 `28b1151` 为基线。**实现已完成，真实验收未全部通过**：`runtime.photo_streaming` 保持 `false`，原整句播放路线继续生效。未提交或推送 Git，未自动确认人物声音，未修改宿主电源方案。

最近一次完整状态见 `outputs/streaming-optimization/20261105-082835/gates.json`（2026-11-07 用 `evaluate_gates.py` 生成）。结论为 `not_passed`。

## 已完成的实现

- 使用 CosyVoice3 原生流式合成。每句恢复初始块长，保留句内状态和尾部刷新。
- 使用固定前缀增益和有状态限峰。声音持续重采样，不按块重新调整 RMS。
- 画面按 25 帧发布封口的无损块，尾块保留实际帧数。动作、重叠、平滑和眨眼状态跨块保留。
- 固定照片特征只上传一次。warp 与 decoder 之间的张量保留在 GPU。遮罩和背景系数预先计算。
- 会话持有 GPU 使用权。其他会话和构建仍受互斥保护。缺少有效准入记录时使用串行分块。
- 声音与画面使用同一采样位置。缓冲判断同时检查两种生成速度。欠载时暂停两种内容游标，并记录事件。
- 取消会拒绝旧代数据。客户端等待原生生产者退出，再关闭读取器和删除临时块。
- 保留原会话、WebRTC、原生播放器及整句接口。新增流式接口和诊断指标。
- 帧数计算统一改为整数运算。此前 `math.ceil(4.48 s × 25)` 的浮点误差会让 112 帧被判成 113，真实暖机采样曾因此失败；修复后回归通过。

## 已完成的验证

- 后端全量：**290 passed、2 skipped、1 warning**（`.runtime/logs/streaming-all-tests-integer-frames.log`）。前端生产构建、9 项原生音频测试、5 项 Windows 专项检查通过。
- 一次独立代码复核发现 6 项问题，已通过失败回归复现并集中修复；修复后全量后端检查通过。
- 真实固定关键点比较：12 帧的优化前后 RGB 像素全部一致，最大差值为 0。
- 真实显存准入（当前源码键）：5 个固定回答、2835 个样本，原生任务活动区间重叠 33.17 秒，最小空闲显存 3514 MiB，保守峰值增量 4504 MiB。证据 `20261105-082835/memory-082854/report.json`。
- 单次 WebRTC 冒烟：并行分块、首个非静音音频接收约 23.56 秒、缓冲停顿 0。该单次结果不是性能基线。
- A/B 暖机采样：每版文字与语音各 20 个预热样本、共 80 个样本，失败 0。12 次冷启动（每版文字/语音各 3 次）全部完成。
- 十轮连续运行：媒体 600.003 秒、接收 15000 帧、平均 24.99985 FPS、60 个十秒窗口均 25 FPS、`stream_underruns = 0`、错误列表为空；播放中打断后半秒静音检查与同连接恢复均通过。系统显存采样峰值 4128 MiB。报告 `20261105-082835/soak/report.json`。发送端另有 3 次队列替换记录（1 次旧代解码帧、2 次 PCM 重试），接收帧数与窗口帧率不受影响。

## 性能对比结果（关键未通过项）

`20261105-082835/warm-summary.json`，顺序为 baseline → candidate → candidate → baseline，同一人物、同一问题组、同一组带 SHA-256 的预录 WAV、同一 Performance 电源方案。数值为从控制发送到接收首个非静音音频帧的毫秒数。

| 输入 | 基线中位数 | 候选中位数 | 差值 | 基线 P95 | 候选 P95 |
|------|-----------:|-----------:|-----:|---------:|---------:|
| 文字 | 34 713.5 | 34 783.3 | **+69.8 ms（+0.20%）** | 70 769.1 | 50 028.1 |
| 语音 | 33 519.4 | 33 699.6 | **+180.2 ms（+0.54%）** | 75 618.2 | 62 656.7 |

P95 明显改善，但计划要求「首声接收中位数下降」：两组都**没有下降**。差异仅 0.2% / 0.5%，属噪声级，按计划必须追加同组采样后再判定，不能据此宣称提速，也不能把 P95 改善当作通过。`warm-extra` 阶段已写好，将再采一整轮 ABBA 共 80 个样本，合并为每版每种输入 40 个样本后重新评估。

## 其他未通过 / 未测项

- **真实打断验收失败**：`20261105-082835/interruptions/report.json` 报 `KeyError: 'generation'`，原因是验收脚本读取了 `/api/sessions/{id}/metrics` 并不存在的顶层 `generation` 字段，并非产品代码故障。已改为读取会话原子标记 `data/sessions/{id}/generation.json` 验证两次打断的代际递增，并允许重复运行；脚本语法已校验，**等待真实重跑**。
- **浏览器音画差 ≤150 ms 未测**：没有真实 Chrome 的 `estimatedPlayoutTimestamp` 统计。按计划记为「未测」，不能判为通过；连续运行的 aiortc 报告自身也不能替代浏览器结论。`save_browser_report.py` 已备好，供采集后生成 `browser/report.json`。
- **收尾资源检查未运行**：线程、读取器、临时块、显存释放及宿主磁盘 10 GiB 余量尚未实测。`final_cleanup.py` 已备好。
- 生成中打断、播放中连续两次打断、同一连接恢复：单元测试覆盖，真实链路证据未取得（见上）。
- 主观声线、自然度、情感效果和人物相似度需要人工复查所保存的对照音视频。自动测试不确认人物声音，实体麦克风与全新机器安装不属于本轮已通过项目。

## 测试条件与证据

人物使用当前已就绪、声音已确认的 `a5d729b6669a4cb28adf3b5af0821e22`，未修改声音确认状态。照片保持 870×824，视频保持 25 FPS，模型和动作采样步数保持原值。

电源方案为当前的 Performance，GUID `27fa6203-3987-4dcc-918d-748559d549ec`，本轮未切换。Windows 监视器每 5 秒记录方案，变化时标记测试无效。

两版源码位于独立的 WSL 本地测试目录（`comparison.json` 中的 `baseline` / `candidate`），模型、人物和来源素材仍在原目录，共用原项目的 GPU 互斥锁。两版复用同一组问题和同一组 WAV。冷启动指重启本项目管理的服务，不清理系统缓存。

证据根目录：`outputs/streaming-optimization/20261105-082835/`。

- `baseline-source.tar`、`candidate-source.tar` 和 SHA-256 清单保留实际测试源码；`freeze_comparison.py` 只哈希 `backend/`、`configs/`、`scripts/`、`frontend/src/`，因此本轮对验收脚本的修改不会使冻结源码失效。
- `conditions.json` 保留人物、模型目录清单及输入 WAV 清单的哈希；`input-audio/` 为固定 WAV 组。
- `warm-summary.json`、`cold-summary.json` 保留全部采样；失败样本与早期失败轮次 `20261105-035419/`、`20261105-052635/` 均保留，未删除。
- `memory-082854/report.json` 保留显存样本和原生阶段计时；`soak/` 保留十轮报告与关键帧截图。
- `transitions.json`、`*-windows-duration.json` 保留服务切换与 Windows 计时；`power-monitor.jsonl` 位于 `20261105-035419/`，连续性由 `comparison.json` 的 `power_evidence` 指向。
- `gates.json` 为当前逐项门禁判定（10 项：6 项通过、1 项因噪声级差异被阻塞、1 项失败、1 项未测、1 项未运行）。

## 本轮阻塞与续跑方法

本会话中 `wsl.exe` 被 WorkBuddy 安全策略列入程序黑名单，Bash 与 PowerShell 均被拒绝，因此**无法启动真实模型、无法重跑打断/浏览器/追加采样/收尾检查**。这不是项目代码错误；在安全中心放行后，或由用户在普通 PowerShell 中按下列步骤执行即可：

```powershell
cd D:\Projects\3d\DigitalHuman
powershell -ExecutionPolicy Bypass -File .\.superpowers\sdd\2026-11-04-streaming-photo-dialogue\resume-acceptance.ps1 -BrowserWaitMinutes 30
```

该脚本按序执行：`warm-extra`（追加同组采样）→ 切回流式候选版 → 重跑打断验收 → 提示采集 Chrome 统计并等待 → 停止冻结服务 → `final_cleanup.py` → `evaluate_gates.py` → 恢复原服务。每个阶段记录电源方案前后值与耗时，变化即判无效；失败即停止后续阶段。它不改配置、不改人物、不提交 Git。

浏览器统计需人工在 `http://localhost:7860/?diagnostics=1` 用真实 Chrome 播放一次回答，把 `video.dataset.rtcSynchronization` 另存为 JSON 文件后运行：

```powershell
wsl.exe -d DigitalHuman --cd /mnt/d/Projects/3d/DigitalHuman -- /opt/digitalhuman/portrait/bin/python .superpowers/sdd/2026-11-04-streaming-photo-dialogue/save_browser_report.py --delivery-mode streaming_parallel --json /mnt/d/Projects/3d/DigitalHuman/outputs/streaming-optimization/20261105-082835/browser/raw-input.json
```

缺少浏览器数据时脚本保持 `not_measured`，不会误判为通过。只有 `gates.json` 的 `overall` 变为 `passed`、且人工复核过对照音视频后，才由用户决定是否把 `runtime.photo_streaming` 改为 `true`。若追加采样后中位数仍不下降，按计划保留原播放路线并在此记录原因。

本轮不承诺未经测量的「秒回」，不自动确认人物声音，也不自动提交或推送 Git。

## 2026-11-07 启动与真实测试尝试

按用户要求尝试「启动项目和浏览器，真实测试」，结果记录在 `outputs/streaming-optimization/20261105-082835/launch-attempt-20261107.json`。

- 浏览器已启动：WorkBuddy 内置浏览器与真实 Chrome 都打开了 `http://localhost:7860/?diagnostics=1`。
- 服务未启动：绕过环境代理直连探测，7860 / 9880 / 9881 / 8081 四个端口全部 `connection refused`；经代理时一律返回 502，不能当作服务状态。
- 原因：`scripts/start.ps1` 的最终路径是 `wsl.exe -d DigitalHuman -u root --exec bash ...`，而 `wsl.exe` 位于安全中心程序黑名单。普通调用与申请沙箱授权的调用均被 `command safety deny rule` 拒绝；上一会话的 Bash 调用明确提示该拦截不能由当前命令批准或绕过。因此**没有启动任何模型，也没有进行真实推理或浏览器音画同步测量**。
- 未尝试等价绕过、未修改配置、未提交 Git。

解除黑名单后可两条路任选：仅启动原项目并在界面手工试聊用 `.\scripts\start.ps1`；执行完整剩余验收用上文 `resume-acceptance.ps1`（它会自行管理冻结的两版服务，无需先手工启动）。**旧记录称内置浏览器缺 `estimatedPlayoutTimestamp`，2026-11-07 实测不成立**，详见下节。

## 2026-11-07 16:12 真实浏览器对话测试

用户在普通 PowerShell 启动了服务。四个端口健康检查均为 200，`/api/health` 返回 `ready_avatars: 1`、`offline_inference: true`，服务根为原项目，开关仍为 `false`。完整证据：`outputs/streaming-optimization/20261105-082835/real-browser-20261107/report.json`。

- 会话 `a10960094eea47178116929e4b229b1e`，人物 `a5d729b6669a4cb28adf3b5af0821e22`（ready、声音已确认）。输入「请用一句话问候观众。」，回答「你好，欢迎观看！」，整轮 `completed`。
- 服务端计时：回答生成 4 367 ms、首句声音 189 112 ms、首句画面 69 798 ms、**首段进入播放队列 263 474 ms**、整轮 265 487 ms。运行方式为 `buffered`（整句准备）。这是冷启动单次值，不是性能基线；它直观解释了流式优化要缩短的串行等待。
- 浏览器诊断面板（内置 WorkBuddy 浏览器）：同步采样 821 次、最大音画差 **52.0 ms**、P95 39.0 ms、解码 10 499 帧、接收端丢帧 0、25 FPS、缓冲停顿 0 次。服务端 9 479 帧，1 次 `queue_replaced` 帧替换，`stream_cleanup_errors` 为空。
- **重要修正**：内置浏览器本次能采集 `estimatedPlayoutTimestamp`。旧验收记录中「内嵌浏览器缺少该统计字段」的结论在本机当前环境不成立，后续流式版可直接用它采集，不必依赖 Chrome 扩展。
- **口径限制**：这批数据来自 `buffered` 路线，**不能**用于本计划的 `browser_av_sync` 门禁，因此没有写入 `browser/report.json`，门禁仍为 `not_measured`。已加固工具：`save_browser_report.py` 新增 `--delivery-mode`，`evaluate_gates.py` 只接受 `streaming_*` 的浏览器报告，其它一律记为未测。

### 本次新增的可逆测试与续跑工具

- `switch-to-streaming-test.ps1`：临时把开关写成 `true` → 停止并重启服务 → **立即把磁盘配置恢复为 `false`**（服务启动时已读入配置）。用于只测一次流式版，不改动生产路线。缺 `.runtime/stream-admission.json` 时会退回 `streaming_serial`。
- `restore-buffered.ps1`：测完重启服务，完全回到整句准备路线。
- `freeze` 前置检查：WSL 重启会清空 `/tmp`，`comparison.json` 指向的 `20261105-082835/baseline|candidate` 冻结树**当前已不存在**。`resume-acceptance.ps1` 现在会先检查并给出重建命令（`freeze_comparison.py` → `run_comparison.py profile`）。重新冻结后，`evaluate_gates.py` 会从 `previous_evidence` 回退读取已完成的 warm/cold/soak/interruptions 报告，并在 `gates.json` 的 `adopted_from_previous_evidence` 记录来源；浏览器与收尾报告**不允许回退**。
- 三个 PowerShell 脚本均为纯 ASCII（PowerShell 5.1 按 ANSI 解析无 BOM 脚本，中文会乱码），已通过语法校验。
