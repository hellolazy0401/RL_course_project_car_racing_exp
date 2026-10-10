# PPO KL 对照实验：Anaconda Prompt 操作指南

本轮目标：检验限制策略更新是否缓解训练退化。环境版本、完成判定、奖励、时限、观测和动作保持原样。视频中的长时间离路、掉头逆行和覆盖不足 95% 不能通过修正完成判定解决；环境修复应另做实验，不与本轮混合。

## 1. 进入现有环境与项目

在 Anaconda Prompt 依次执行，无需重建环境或重新安装依赖：

```bat
conda activate rl_carracing
cd /d D:\Desktop\RL_project
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

应显示 CUDA: True 和 RTX 3060 Laptop GPU。PyTorch 的 CUDA 0 不等于 Windows 任务管理器的 GPU 0。

## 2. 执行两组对照训练

```bat
python -m racing.kl_experiment --stage train --device cuda
```

这一条会依次运行 control、kl 两组，不会同时争用 GPU：

|项目|control|kl|
|---|---|---|
|起始检查点|同一个 runs_tuning_lr1e4_1000k/ppo_s0/best.pt|同左|
|起始累计决策数|800032|800032|
|新增决策数|200000|200000|
|最终累计决策数|1000032|1000032|
|actor 学习率|继承 0.0001|继承 0.0001|
|每批最大 PPO epoch|继承 10|继承 10|
|KL 提前停止|关闭|阈值 0.02|
|价值网络更新预算|原计划|原计划|
|验证集|种子 1000000–1000019，共 20 条|同左|

两组恢复原模型、Adam 优化器及随机数状态；环境从相同种子流重新开始。每组先评估并保存起始模型，之后约每 10 万决策评估一次，最后再评估。输出目录分别为 runs_ppo_kl_ablation/control 和 runs_ppo_kl_ablation/kl。原始训练文件不被覆盖。

如果新的 20 条验证赛道结果没有超过分支起点，best.pt 会保留起点模型。这是有效结果，意味着新增训练尚未改善验证成绩。last.pt 始终是预算结束模型。

预计两组加评估约 5–7 小时，受设备负载影响。不要同时启动其他 GPU 训练。已单独完成 512 决策/组的短自检，不能把自检成绩当作算法效果结论。

如希望分两次运行，使用以下命令替代上面那一条，不要两套都运行：

```bat
python -m racing.kl_experiment --stage train --arm control --device cuda
python -m racing.kl_experiment --stage train --arm kl --device cuda
```

脚本拒绝覆盖已有组目录。例如 control 已完成，只执行 --arm kl。如果整组需要重做，给所有后续命令加同一个新目录参数，例如 --out runs_ppo_kl_ablation_v2。中断不等同于完成；不要把已有未完成目录当作可跳过的结果。严格续训功能仍由 racing.train --resume 提供，续训须保持该分支的参数（尤其 KL 阈值）一致。

## 3. 打开对比报告

训练命令结束会自动生成报告，也可以手动刷新：

```bat
python -m racing.kl_experiment --stage report
start "" "runs_ppo_kl_ablation\report\index.html"
```

重点检查：

1. 两组起始验证回报是否一致，应显示 True。
2. 两组 last 的验证回报、覆盖率与完成率，KL 组是否减少退化。
3. best 是否改善，还是停留在 800032 步起点。
4. kl_stop_fraction 是触发提前停止的更新批次比例；actor_updates_mean 是每批实际策略优化步数。较低 KL 不是成功标准，最终要看驾驶成绩。
5. segment_train_hours 仅为新增训练时间；wall_hours 包含本次评估，不会把父检查点历史训练时间误算成本次成本。

报告数据：runs_ppo_kl_ablation/report/comparison.csv；曲线：comparison.png。此阶段比较的是验证集，只有一个训练随机种子，不支持统计显著性结论。

## 4. 生成两组最佳模型的验证视频

```bat
python -m racing.kl_experiment --stage videos --device cuda
```

每组记录相同的前 5 条验证赛道，约需数分钟。输出：

```text
runs_ppo_kl_ablation/control/validation_video_best/videos/
runs_ppo_kl_ablation/kl/validation_video_best/videos/
```

视频可观察弯道、离路恢复和抖动。若两组 best 都是相同起点，视频相同属于预期结果。此命令只是诊断，不是独立测试。

## 5. 方案确定后，再做独立测试

先根据验证集完成调参，再执行本步。不要每看一次测试结果就继续据此调参。

```bat
python -m racing.kl_experiment --stage test --device cuda
start "" "runs_ppo_kl_ablation\report\index.html"
```

对两组的 best 和 last，共 4 个模型，各运行 50 条新赛道，使用相同的种子 3000000–3000049；每个模型保存前 5 条视频。按此前测试速度，全部约 1.5–2 小时。新赛道避免重复依赖已经检查过的 2000000–2000049。

输出目录分别是每个组下的 test_best、test_last，内有 summary.json、episodes.csv、videos。报告会自动加入测试指标。不允许覆盖已有评估目录。

## 实现细节与限制

- racing/train.py 新增 --target-kl、--branch-from、--eval-seed-start。旧命令不传 --target-kl 时策略更新方式保持原样。
- 每次 actor 小批量更新后，用整批 rollout 计算近似 KL；超过阈值后停止本批剩余 actor 更新。触发阈值的那一步已经执行，不回滚，所以 0.02 不是绝对上界。
- critic 网络独立，其更新次数保持不变，尽量将实验差异集中在策略更新限制。
- --branch-from 只允许 target_kl 与父训练优化设置不同，其他关键优化参数仍须一致。脚本自动继承这些参数，避免手工填写错误。
- 下一轮若 KL 组没有改善，不要直接宣称 KL 无用；检查是否几乎每批第一步就越限，以及验证成绩是否仅受少量赛道影响。先确定原因，再决定是否单独减小学习率。
- 本轮未修复环境圈完成检测。当前所有新旧成绩仍处于相同环境协议。若之后修复，应对所有算法统一环境版本，并将修复后结果单独标注。

## 可选代码自检

```bat
python -m pytest tests -q
```

包含默认更新兼容性、KL 提前停止、critic 更新保持、旧检查点兼容、分支配置限制、两组参数一致性与原有环境/GAE 检查。短实验输出在 runs_ppo_kl_selfcheck，仅用于工程自检。

2026-10-09 本机验证：13 项测试通过；两组 CUDA 各运行 512 决策并完成起点/终点评估、检查点保存及报告生成。起点回报均为 893.3554817；对照组 actor/critic 更新次数为 40/40，KL 组为 1/40；整批更新后 KL 分别为 0.48218 和 0.03639。KL 组第 1 次更新即越过 0.02，之后停止，符合“越限后停止而非强制回滚”的实现。只有一条赛道、一个更新批次，不能以此判断算法优劣。
