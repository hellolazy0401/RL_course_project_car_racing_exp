# GRPO-C：多策略连续控制适配

> **2026-10-15 更新：本文的方法说明已更新为 v2。已有 drive 初始化的 2 万及 10 万决策复核均未稳定学会驾驶，不建议原配置继续扩预算。下一项建议见[GRPO-C 停滞修复与重跑](GRPO-C停滞修复与重跑.md)第 9 节：只改为 standard 初始化的 2 万决策对照，效果尚未验证。v2 不能从 v1 检查点续训；原检查点仍可评估。下文长跑命令为流程参考，须先通过短实验筛查。**

论文：[Khanda 等，《Extending Group Relative Policy Optimization to Continuous Control: A Theoretical Framework for Robotic Reinforcement Learning》，arXiv:2507.19555v1，2025](https://arxiv.org/abs/2507.19555)。本文把连续控制 GRPO 写成一个框架，包含轨迹策略聚类、状态聚类优势、组自适应裁剪、时间平滑、组间多样性及参考策略更新。论文的初步实验只在 HalfCheetah-v4 使用单一随机种子；目前不能据此断言方法在 CarRacing 上有效，也不能把 arXiv 预印本称为已经发表在顶会或顶刊。

本项目把 CLI 名称定为 `grpo-c`，与原有 `grpo` 共存。当前检查点记录 `grpo_c_variant=pixel_motion_global_loo_mixture_v2`。这是论文框架的独立、可执行 CarRacing 实例，未声称逐行复现作者代码；v2 修复了 v1 在像素任务中的状态基线、聚类和参考策略实现问题。

## 方法具体如何落地

|论文组件|本项目实现|
|---|---|
|多个策略|每个环境持有一套独立的高斯 CNN 策略与 Adam；一轮每个策略在同一赛道种子跑一局|
|轨迹特征|平均每决策回报、高斯策略熵、连续动作方差、相对混合参考分布的近似 KL|
|轨迹聚类|标准化四维特征后执行确定性 k-means；默认最多 K=2；将单成员组并入邻近组|
|状态特征|当前帧压缩为 8×8，加上当前帧与最早堆叠帧的 8×8 差值，保留外观和运动；标准化时设方差下限|
|状态聚类|在所有策略的轨迹中执行 DBSCAN；最大簇占比超过 90% 或噪声超过 50% 时，回退为最多 16 个 k-means 状态簇|
|状态优势|按 `gamma` 算每步 return-to-go；基线只使用其他轨迹中同状态簇、同 16 决策时间段的数据；无同簇参考则使用同时段其他轨迹，无对应轨迹则用零；最后在各策略组内标准化|
|自适应裁剪|`clip_g = clip_base × max(1, std_g/(std_all+1e-8))`；`std_g` 是该组未标准化优势的标准差|
|时间平滑|对同一局相邻观测的策略高斯均值加 L2 距离惩罚|
|组间多样性|以各组初始观测上的平均高斯均值为代理中心，先做 `m / sqrt(norm(m)^2 + 1)` 平滑归一化，再对中心内积超过阈值的部分加惩罚，避免原余弦相似度在近零输出处产生过大梯度|
|参考策略|每个策略保留对应的冻结参考副本，仅对同一成员做参数 EMA；参考密度是各成员动作概率密度的精确加权混合，权重向回报较好的半数成员平滑移动；当前策略到混合分布的 KL 用每动作维度 5 个 Gauss–Hermite 节点近似，共 125 点|
|动作初始化|默认 `--grpo-c-init drive`：转向约 0、油门约 0.5、刹车约 0.02；所有维度继续由网络学习。`standard` 保留原初始化，用于消融|

运动像素特征、排除本轨迹且匹配时间的基线、聚类回退、平滑动作中心和低刹车初始化都是针对 CarRacing 的工程选择，需要在课程报告中披露。参考策略不再跨独立网络平均权重；混合概率密度计算本身是精确的，但 KL 为数值求积近似，极小负值可能来自近似误差。状态聚类与混合参考计算都会增加开销。默认超参数只是启动实验的设置，没有经过充分调参验证。

初始化的收益必须与训练的收益区分：v2 会保存 `initial.pt` 和 0 步验证结果，只有训练后的持续改善才能算学习进展。`--grpo-c-state-fallback-clusters 16` 指定回退簇数；设为 `0` 可关闭回退以做消融。所有基线都只能使用本轮已经采集的轨迹，不使用测试结果。

算法不训练价值网络。`grpo-c` 与之前的 `grpo` 不同：旧实现只有单一共享策略、同赛道多条轨迹及整局组相对回报；新实现训练独立策略，使用每步状态优势和论文提出的聚类、裁剪及正则项。

验证时会在完全相同的赛道上评估所有成员，并选择验证均分最高的成员保存为可直接录视频的 `best.pt` 或 `last.pt`。这使其比单策略 PPO 有更多选择机会；课程报告需要写明，不能把一次验证选出的成员当作未经选择的固定策略。`member_validation.csv` 保留全部成员数据。新旧版本检查点均能通过 `racing.evaluate` 评估，但 v1 检查点不能通过 v2 的 `--resume` 继续训练。

## 安装环境

沿用已有 Conda 环境，无新增依赖。Anaconda Prompt 执行：

```bat
conda activate rl_carracing
cd /d D:\Desktop\RL_project
python -m pytest tests -q
```

本机 PyTorch 仅识别一块 CUDA 显卡，编号 `cuda:0`，名称为 NVIDIA GeForce RTX 3060 Laptop GPU。Windows 任务管理器的 GPU 0/1 编号与 CUDA 编号不必相同；这里的 `--device cuda` 使用该 NVIDIA 显卡。

## 短流程自检

```bat
python -m racing.train --algo grpo-c --steps 500 --num-envs 2 --grpo-c-clusters 2 --grpo-c-epochs 1 --actor-lr 1e-4 --device cuda --completion-mode lap95 --eval-every 500 --eval-episodes 1 --out runs_grpo_c_smoke
```

这会评估两个初始策略，各在同一赛道采集一条完整轨迹，执行多策略更新，再评估终点。两个成员最多形成一个具有有效相对比较的策略组，单成员组会自动合并。完整组可能略超 500 决策，查看 `finished.json` 中实际的 `steps` 和 `budget_overshoot`。短跑只能验证管线。

历史 v1 两策略自检曾测得：500 决策训练约 59 秒，约 8.4 决策/秒；两次全成员验证共 4 局约 220 秒；录制 1 局完整原始帧视频耗时约 21 分钟。**这些数字只描述旧 v1 当次运行，不是 v2 的耗时测量。**不同运行的渲染和机器负载可能差异很大，请根据 v2 短跑的 `updates.csv`、`finished.json` 及实际视频录制耗时安排预算。

## 2 万决策筛查之后，扩到 10 万决策试验

已有低刹车初始化短实验未通过稳定学习筛查，目前不要直接执行下面的长跑命令。下一项对照见[修复与重跑指南](GRPO-C停滞修复与重跑.md)第 9 节。当前代码仍默认低刹车初始化和最多 16 个回退状态簇；初始化对照需要显式传入 `--grpo-c-init standard`。`racing.suite` 也支持显式 `--actor-lr`；省略时 GRPO-C 使用 `1e-4`。

```bat
python -m racing.train --algo grpo-c --seed 0 --steps 100000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-state-eps 0.65 --grpo-c-state-min-samples 4 --grpo-c-epochs 2 --grpo-c-beta 0.01 --grpo-c-smooth 0.01 --grpo-c-diversity 0.01 --grpo-c-diversity-threshold 0.2 --grpo-c-reference-tau 0.05 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 25000 --eval-episodes 5 --out runs_grpo_c_lap95_pilot
```

`--num-envs 4` 表示 4 个独立策略；`--grpo-c-clusters 2` 把四条轨迹分成最多两组。训练赛道种子来自 0–999999；验证赛道固定从 1000000 开始。每轮所有策略在同一赛道起点采样。`--completion-mode original` 可切换回原环境规则，须换新的输出目录。

重点看：`updates.csv` 中 `occupied_policy_clusters`、`merged_singleton_groups`、`state_clusters`、`state_dbscan_largest_fraction`、`state_dbscan_noise_fraction`、`state_fallback`、`state_peer_fraction`、`advantage_std`、`clip_mean`、`approx_kl`、`reference_kl`，以及 `member_validation.csv` 中各成员成绩。一组可能是单成员组自动合并的合理结果；DBSCAN 经常触发回退则需结合最终簇数和同簇参考匹配率分析。更新幅度或簇数本身都不能证明模型学会驾驶，必须结合 0 步到后续验证的回报、覆盖率和轨迹变化。

## 视频、测试与报告

```bat
python -m racing.evaluate --checkpoint runs_grpo_c_lap95_pilot\grpo-c_s0\best.pt --device cuda --episodes 5 --seed-start 1000000 --videos 1 --out runs_grpo_c_lap95_pilot\grpo-c_s0\validation_video_best
python -m racing.evaluate --checkpoint runs_grpo_c_lap95_pilot\grpo-c_s0\last.pt --device cuda --episodes 5 --seed-start 1000000 --videos 1 --out runs_grpo_c_lap95_pilot\grpo-c_s0\validation_video_last
```

确定参数后，统一使用未参与调参的赛道做测试：

```bat
python -m racing.evaluate --checkpoint runs_grpo_c_lap95_pilot\grpo-c_s0\last.pt --device cuda --episodes 50 --seed-start 3000000 --videos 0 --out runs_grpo_c_lap95_pilot\grpo-c_s0\test
python -m racing.evaluate --checkpoint runs_grpo_c_lap95_pilot\grpo-c_s0\best.pt --device cuda --episodes 50 --seed-start 3000000 --videos 1 --out runs_grpo_c_lap95_pilot\grpo-c_s0\test_best
python -m racing.report --root runs_grpo_c_lap95_pilot --allow-unequal-budgets
python -m racing.report --root runs_grpo_c_lap95_pilot --selection test_best
start "" "runs_grpo_c_lap95_pilot\report\index.html"
```

报告允许展示 GRPO-C 完整回合造成的预算超出，并列出实际步数；它不能把不同实际决策预算称为严格等样本比较。与 PPO、A2C、GRPO 比较时需要相同完赛模式、测试赛道和尽可能接近的实际预算，也需要多个独立训练种子。价值网络指标对 GRPO-C 为不适用。

## 后续从零跑 100 万决策

```bat
python -m racing.train --algo grpo-c --seed 0 --steps 1000000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-state-eps 0.65 --grpo-c-state-min-samples 4 --grpo-c-epochs 2 --grpo-c-beta 0.01 --grpo-c-smooth 0.01 --grpo-c-diversity 0.01 --grpo-c-diversity-threshold 0.2 --grpo-c-reference-tau 0.05 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 100000 --eval-episodes 10 --out runs_grpo_c_lap95_main
```

建议在 10 万试验确认确有学习、计算时间可接受后再运行。最后会跑完整个策略组，实际步数可能超出目标，检查点和 `finished.json` 都记录实际值。历史实验目录不会被覆盖。

若从 **v2 pilot** 的 `last.pt` 续训到累计 100 万，可用相同训练参数，把 `--steps` 设为 `1000000`，加 `--resume runs_grpo_c_lap95_pilot\grpo-c_s0\last.pt`，并将 `--out` 换成一个新目录。版本、初始化、回退簇数、关键聚类、正则化和环境参数必须一致。v1 续训会被拒绝，但旧模型仍可评估。续训会重新评估所有成员并建立新目录的 best；环境回合本身重新开始。

来源代码：[grpo_c.py](../racing/grpo_c.py)、[grpo_c_reference.py](../racing/grpo_c_reference.py)、[train.py](../racing/train.py)、[evaluate.py](../racing/evaluate.py)。
