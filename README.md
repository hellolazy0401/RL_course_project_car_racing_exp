# CarRacing 强化学习课程项目

**使用 Anaconda：优先查看 [Conda 运行指南](docs/Conda运行指南.md)。环境名为 `rl_carracing`；激活后用 `python` 执行下面的命令，不再使用 `.venv\Scripts\python.exe`。**

**建议题目：视觉连续驾驶中策略更新与价值估计的对照研究——A2C、PPO 与 ICML 2025 VPG-repeat。**

本工程从像素学习连续驾驶，支持 `--algo a2c|ppo|vpg50|grpo` 切换。环境与测试赛道可统一；GRPO 使用同赛道完整轨迹分组、仅策略网络，采样与预算边界不同，详见 [GRPO 运行指南](docs/GRPO运行指南.md)。`vpg50` 是本工程的命令行简称，论文称其为 **VPG-repeat-k**，不是作者命名的新算法。

先读 [算法选型与实验设计](docs/算法选型与实验设计.md)，再按本页操作。硬件与预算见 [配置与时间成本](docs/配置与时间成本.md)。本目录原有的课程清单和 GIF 均保留。已有 `.venv` 只适用于本机；复制项目到新电脑后重新安装。

## 1. 你最终能得到什么

- 四个算法的可读 PyTorch 源码；A2C 为同步 GAE 变体，PPO 为 PPO-Clip，VPG-repeat 保留独立 critic 多步更新，GRPO 为连续控制适配版。
- 每次训练保存配置、软件版本、源码 SHA256、模型权重、优化器状态、TensorBoard、逐回合和逐次更新 CSV。
- 最终等预算模型和验证集最佳模型分别测试，输出逐赛道得分、完赛率、覆盖率、≥900 比例。
- 多训练种子的均值、标准差、bootstrap 区间、配对差值；按决策数、原始帧数、训练时间绘图。
- 固定测试赛道的完整 MP4 和可离线打开的 HTML 报告。

**短跑结果只证明软件流程可执行，不能作为算法排名或收敛结论。**正式实验需要执行下面的 `main` 命令；100 万决策也只是预先规定的预算，不保证任何算法达到 900 分。

## 2. Windows 从零安装

推荐 Windows 10/11、64 位 CPython 3.11，GPU 使用 NVIDIA 驱动和 PyTorch CUDA 12.1 wheel；无需另装系统 CUDA Toolkit。不要用 MSYS2 的 `python.exe`。本机 PATH 的默认 Python 正是 MSYS2，建议显式指定解释器。

新机器首先安装 Python 和 C++ 编译工具（安装命令可能触发系统权限提示）：

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

重新打开 PowerShell，进入项目：

```powershell
cd D:\Desktop\RL_project
py -3.11 --version
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Python "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
```

如果 Python 安装在其他位置，把 `-Python` 改成实际路径。本机已发现可用解释器，可以直接运行以下替代命令，不必重装 Python：

```powershell
cd D:\Desktop\RL_project
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Python "D:\Anaconda_envs\envs\lidar_vis\python.exe"
```

脚本会创建独立 `.venv`、安装依赖并运行测试，不修改原有 Conda 环境。无 GPU 时给安装脚本加 `-Device cpu`，后续命令加 `--device cpu`。CPU 适合查错，正式图像训练优先用 GPU。

如果需要逐条执行，等价步骤是：

```powershell
cd D:\Desktop\RL_project
& "D:\Anaconda_envs\envs\lidar_vis\python.exe" -m venv .venv
$env:PATH = "$PWD\.venv\Scripts;" + $env:PATH
.\.venv\Scripts\python.exe -m pip install pip==24.3.1 setuptools==75.6.0 wheel==0.45.1 swig==4.3.0
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`--no-build-isolation` 和加入 `.venv\Scripts` 的 PATH 很重要：Box2D 编译过程需要调用同一环境的 SWIG。报 `Microsoft Visual C++ 14.0 or greater is required` 时安装上面的 C++ 工作负载；安装后重开终端。不要把旧环境中 PyTorch 1.8 的可用性当作本工程已安装成功。

## 3. Linux / 云服务器从零安装

选已有 Python 3.11 的 Ubuntu GPU 镜像最省事。若用 Ubuntu 24.04 原生 Python 3.12，先单独安装 Python 3.11 或选对应镜像，不要直接把版本号改掉假定同样可复现。

```bash
sudo apt-get update
sudo apt-get install -y build-essential swig ffmpeg libgl1 libglib2.0-0 python3.11-venv
cd /your/path/RL_project
bash scripts/setup_linux.sh
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m racing.suite --preset smoke --device cuda
```

Ubuntu 默认源没有 `python3.11-venv` 时使用预装 Python 3.11 的镜像/Conda 环境，再执行脚本。设置为无界面 SDL，训练不需要桌面和 X server。不要把 Windows 的 `.venv` 上传到 Linux。以下命令中的 `.\.venv\Scripts\python.exe` 在 Linux 替换为已激活环境中的 `python`。

## 4. 先跑完整管线自检

```powershell
cd D:\Desktop\RL_project
.\.venv\Scripts\python.exe -m racing.suite --preset smoke --device cuda
```

它顺序训练 3 个算法，每个 512 决策，包含真实的 50 次 critic 更新、验证、保存、重新加载、最终测试、录制视频和出图。`smoke` 的回合与网络仍是真实 CarRacing；只有训练预算、并行数和验证回合数变小。报告位于 `runs\smoke\report\index.html`，双击打开。不要在该报告上写“某算法最优”。

同名输出目录存在时程序会拒绝覆盖；需要重跑可用 `--out runs_recheck`。批处理恢复已完成条目用 `--skip-completed`，它不会自动续训不完整目录。

## 5. 测速与试验预算

测速应采用与正式训练相同的 4 环境、128 rollout、128 batch；不要用 smoke 的缩小批次外推最终时间。

```powershell
foreach ($algo in @('a2c','ppo','vpg50')) {
  .\.venv\Scripts\python.exe -m racing.train --algo $algo --steps 4096 --num-envs 4 --rollout 128 --batch-size 128 --eval-every 4096 --eval-episodes 1 --out runs\benchmark
  if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $algo" }
}
.\.venv\Scripts\python.exe -m racing.benchmark --root runs\benchmark --steps 1000000 --seeds 3 --hourly-price 0
```

`hardware_estimate.json` 会记录实际吞吐、峰值张量显存与投影。云端将 `--hourly-price` 改为下单时每小时含税价格，货币自行统一；脚本不查询也不承诺租赁报价。30% 附加项只是安装、评估与波动的预算预留。

随后试跑每种算法 10 万决策、1 个训练种子：

```powershell
.\.venv\Scripts\python.exe -m racing.suite --preset pilot --device cuda
.\.venv\Scripts\python.exe -m racing.benchmark --root runs\pilot --steps 1000000 --seeds 3
```

pilot 用来判断是否学到东西、是否需要调整学习率或延长预算。调参只看验证集；pilot 自动产生的测试分数只用于管线检查，不用于选择超参数。严谨的最终测试可用 `--seed-start 3000000` 另建未看过的测试集合；务必对所有方法统一更换。

## 6. 正式实验：一条命令完整跑完

```powershell
.\.venv\Scripts\python.exe -m racing.suite --preset main --device cuda
```

默认 **3 算法 × 3 训练种子 × 100 万决策**，最多约 3600 万训练原始帧；每个模型每 5 万决策在同样 10 条验证赛道评估。每次训练完成后，`last.pt` 与 `best.pt` 分别在 50 条相同测试赛道上评估，各录前三条视频，然后自动汇总。

资源允许时，优先扩展到 5 个训练种子；如果 pilot 显示 100 万仍明显没学会，再统一增加到 300 万，不能只给喜欢的算法更多预算。

```powershell
.\.venv\Scripts\python.exe -m racing.suite --preset main --seeds 0 1 2 3 4 --steps 3000000 --out runs_extended --device cuda
```

训练时不必一直开浏览器，但系统不能休眠。插电、保持散热。查看实时曲线：

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir runs\main --port 6006
```

在浏览器打开 `http://localhost:6006`。TensorBoard 命令会占据当前终端，训练用另一终端。

## 7. 单独切换算法、评估和恢复

```powershell
.\.venv\Scripts\python.exe -m racing.train --algo ppo --seed 0 --steps 1000000 --out runs_single
.\.venv\Scripts\python.exe -m racing.train --algo a2c --seed 0 --steps 1000000 --out runs_single
.\.venv\Scripts\python.exe -m racing.train --algo vpg50 --seed 0 --steps 1000000 --value-steps 50 --out runs_single
.\.venv\Scripts\python.exe -m racing.evaluate --checkpoint runs_single\ppo_s0\last.pt --device cuda --episodes 50 --videos 3
.\.venv\Scripts\python.exe -m racing.evaluate --checkpoint runs_single\ppo_s0\best.pt --device cuda --episodes 50 --videos 3 --out runs_single\ppo_s0\test_best
```

CPU 推理也可：`--device cpu`。所有测试默认输出目录不覆盖已有文件；更改输出目录可重复验证。

每次验证保存 `last.pt`；正常完成保存 `finished.json`。Ctrl+C 会尽力保存 `interrupted.pt`，直接关机/杀进程只能使用最近 `last.pt`。续训必须用新目录/名字，且传回原来的超参数：

```powershell
.\.venv\Scripts\python.exe -m racing.train --algo ppo --seed 0 --steps 2000000 --resume runs_single\ppo_s0\last.pt --out runs_resumed --name ppo_s0
```

`--steps` 是累计目标，不是追加步数。恢复权重、优化器、随机数状态和累计计数；Box2D 物理状态不序列化，恢复时会开始新的训练回合，因此不是逐位连续复现。新目录只记录续训段的验证曲线，不能把该段 AUC 当作从零训练 AUC。需要严格从零对比时完整重跑；相同 seed 跨系统/GPU 也不保证逐位一致。

## 8. 最有解释力的消融实验

固定策略一次更新，仅改变价值更新次数 `K∈{1,10,50}`。本工程 A2C 与 K=1 的 VPG 分支数学上相同，可作为一致性检查；不是两个独立创新算法。

```powershell
foreach ($k in @(1,10,50)) {
  foreach ($seed in @(0,1,2)) {
    $name = "vpg_K${k}_s${seed}"
    .\.venv\Scripts\python.exe -m racing.train --algo vpg50 --value-steps $k --seed $seed --steps 1000000 --out runs_ablation --name $name
    if ($LASTEXITCODE -ne 0) { throw "Train failed: $name" }
    .\.venv\Scripts\python.exe -m racing.evaluate --checkpoint "runs_ablation\$name\last.pt" --device cuda --episodes 50 --videos 1
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $name" }
  }
}
.\.venv\Scripts\python.exe -m racing.report --root runs_ablation
```

这是额外预算；可以复用相同协议下的主实验 A2C / VPG K=50 原始结果用于论文表格，但不要复制成多个训练种子。只做 K=10 的补充试验是节省预算的办法。

可选泛化测试（改变赛道/背景颜色，不能与正常测试混在一张均值表）：

```powershell
.\.venv\Scripts\python.exe -m racing.evaluate --checkpoint runs\main\ppo_s0\last.pt --device cuda --episodes 50 --videos 3 --domain-randomize
```

对其余算法/种子重复后执行：

```powershell
.\.venv\Scripts\python.exe -m racing.report --root runs\main --selection test_ood
```

## 9. 自动报告和视频

```powershell
.\.venv\Scripts\python.exe -m racing.report --root runs\main
.\.venv\Scripts\python.exe -m racing.report --root runs\main --selection test_best
Start-Process .\runs\main\report\index.html
```

主要输出：

| 路径（每次训练/报告内） | 内容 |
|---|---|
| `config.json` / `provenance.json` | 参数、依赖版本、源码哈希、GPU 信息 |
| `updates.csv` | 损失、GAE 目标拟合误差、解释方差、KL、clip fraction、更新次数、吞吐、显存 |
| `train_episodes.csv` | 每个训练回合原始得分和驾驶指标 |
| `validation.csv` / `validation_episodes.csv` | 验证曲线和逐赛道记录 |
| `last.pt` / `best.pt` | 等预算终点 / 验证最优模型 |
| `test/episodes.csv` / `test/summary.json` | 固定终点模型的详细测试数据 |
| `test/videos/track_2000000.mp4` | 固定第一条测试赛道；不是最佳赛道 |
| `report/comparison.csv` | 算法汇总表 |
| `report/per_run.csv` | 每个训练种子、验证 AUC、首次验证 ≥900 的步数 |
| `report/paired_differences.csv` | 相同训练种子配对的得分差和区间 |
| `report/all_test_episodes.csv` | 所有测试回合，可继续用 Excel 分析 |
| `report/*.png` / `report/index.html` | 报告插图与离线视频展示页 |

同一报告根目录只能放同一组超参数、相同测试协议的实验；改变学习率等参数用不同根目录。默认主表拒绝混合不同训练预算。最好报告完整种子集，不能只交表现最好的种子。

视频按每个原始环境帧录制，播放帧率 50；最大回合约 20 秒。可直接将同赛道的三个 MP4 放进演示文稿，HTML 中可以同时播放。画面差异来自不同策略，不是不同地图或播放倍速。

合成三算法同屏视频（按真实时间对齐；提前结束的画面显示 ended）：

```powershell
.\.venv\Scripts\python.exe -m racing.video_grid --inputs runs\main\a2c_s0\test\videos\track_2000000.mp4 runs\main\ppo_s0\test\videos\track_2000000.mp4 runs\main\vpg50_s0\test\videos\track_2000000.mp4 --labels A2C PPO VPG-repeat-50 --out runs\main\comparison_track_2000000.mp4
```

## 10. 提交与归档

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 requirements-resolved-local.txt
.\.venv\Scripts\python.exe scripts\package_source.py
```

提交源码包、正式报告、原始 CSV、至少一个模型与视频。不要打包 `.venv`。如果压缩包已存在，换文件名保留旧版。课程报告结构建议：任务与 MDP → 算法及论文 → 统一协议 → 配置与预算 → 多种子效果/时间 → K 消融 → 失败案例与局限。先填写实验事实，再下结论。
