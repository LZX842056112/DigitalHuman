# 照片对话数字人

用一张照片进行中文对话。人物可显示口型、眨眼、表情和轻微头部动作。照片的服装、背景和比例会保留。视频只提供参考声音。

所有推理在本机运行。安装时需要联网。默认使用 Silent 电源方案。

## 启动

在项目目录打开 PowerShell，运行：

```powershell
.\scripts\start.ps1
```

打开 http://localhost:7860。选择人物，然后点击“准备人物”。试听参考录音，点击“确认声音并继续准备”。准备完成后，启动对话。可输入文字，也可按住麦克风按钮说话。原生音频播放器位于画面下方。若浏览器阻止自动播放，点击播放器的播放键。点击“打断”可停止当前回答。

若准备时服务退出，重启会将中断任务标为失败。再次点击“准备人物”可重试，并复用有效缓存。

停止服务：

```powershell
.\scripts\stop.ps1
```

首次安装，或补齐依赖：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
.\scripts\doctor.ps1
```

`-EnvironmentOnly` 只安装环境和源码。`-SkipModels` 使用已有权重。首次启用 WSL 可能需要管理员权限和重启。安装失败后可重跑。模型下载支持续传，并检查大小与 SHA。源码写入使用临时文件。已有源码修改不会被覆盖。

环境为 WSL2 Ubuntu 22.04、Python 3.10、Torch 2.3.0+cu121。照片和声音各使用一个环境。已在 RTX 3070 Ti Laptop 8 GiB 上运行真实模型。首次安装约需下载 9.6 GiB 权重；还需为环境、缓存及输出留出空间。脚本保留至少 10 GiB 可用空间。首次加载较慢。每句话先生成声音和完整画面，再开始播放。

## 素材

在界面导入一张 PNG/JPG 照片和一个带声音的视频。使用清晰、无遮挡、只有一张主要人脸的照片。视频需有至少六秒清晰原声，参考音轨最长十分钟。准备时按实际解码采样检查上限，超限会提示更换较短视频，不会静默裁剪。已有且校验有效的参考录音会继续复用。视频处理只解码音轨，不抽帧。

照片与声音视频可分别替换。更换照片只使外观缓存失效。更换视频只使声音参考失效。重启后可重新加载缓存。删除人物会删除项目中的人物副本，不会删除外部原文件或共享模型。

已有三维人物在第一次准备时转换为照片人物，默认使用其第一张身体照片。旧人物 JSON 会保存为 `avatar.legacy.json`。旧素材、输出和模型权重仍保留在磁盘上。

也可通过脚本导入：

```powershell
.\scripts\build_avatar.ps1 -Photo 'C:\path\photo.png' -VoiceVideo 'C:\path\voice.mp4' -Name '人物名称'
```

准备已有的人物：` .\scripts\build_avatar.ps1 -AvatarId '人物编号' `。脚本遇到声音确认步骤时返回。请在界面完成确认。

## 代码结构

```text
backend/api/             人物、素材、任务和会话接口
backend/photo_profiles.py 素材身份、旧资料转换和资产校验
backend/media.py         仅音轨解析与参考录音
backend/dialogue/        Qwen 中文回答和 faster-whisper 识别
backend/voice/           CosyVoice 3 情感克隆声音
backend/portrait/        Ditto 照片处理、动画和外观缓存
backend/transport/       句子队列、共同播放时间轴和 WebRTC
frontend/src/            React 界面及原生音频播放器
configs/                 参数、依赖版本和模型哈希清单
scripts/                 安装、下载、检查、准备和服务管理
models/ third_party/     本地权重和固定版本源码
 data/avatars/           人物素材、照片缓存和参考声音
 data/sessions/          当前回答的临时 WAV/MKV
outputs/                 试听、视频和验收证据
.runtime/                WSL、日志、安装锁定依赖和缓存
```

对话流程为：文字或语音识别 → Qwen 回答和情感 → CosyVoice 声音 → Ditto 画面 → 同步播放。情感类型为自然、开心、难过和生气。声音与画面使用同一类型。两个模型串行使用显卡，空闲模型移回内存。待播队列最多两句。播放以已消耗的 PCM 采样位置选择画面。打断会清空队列，并使旧模型结果失效。

主要接口为 `/api/avatars/import`、`/api/avatars/{id}/photo`、`/voice-video`、`/voice`、`/build`、`/api/jobs/{id}` 和 `/api/sessions`。三维头像候选、相机和动作接口已删除。

## 验证

```powershell
wsl -d DigitalHuman -- bash /mnt/d/Projects/3d/DigitalHuman/scripts/run_python.sh /mnt/d/Projects/3d/DigitalHuman -m pytest -q
cd frontend
npm.cmd run build
node --experimental-strip-types --test tests/native-audio-output.test.ts
```

真实十轮对话和十分钟测试使用 `scripts.acceptance`。该命令会实际调用模型。最新结果见 [验收记录](docs/acceptance.md)。自动测试不能确认人物相似度、声音身份和主观情感效果。仍需实际试看和试听。

首声延迟可用 `scripts.benchmark_latency --avatar-id 人物编号` 测量。该命令会重启本项目管理的服务，分别采集文字和预录语音输入的三次冷启动与二十次预热请求；不结束其他程序。结果保存在 `outputs/latency/`，记录首个非静音音频帧的接收耗时、整轮耗时及后台阶段计时。这些数据不代表实体扬声器已经出声，也不替代持续帧率和浏览器音画同步验收。

比较两版实现时，候选版增加 `--audio-fixtures 原采样目录/input-audio`，复用相同语音文件。工具会校验问题顺序、文件路径与 SHA-256；不要重新生成另一批输入来代替配对比较。
