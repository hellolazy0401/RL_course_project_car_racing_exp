# GRPO-C：多策略连续控制适配

论文：[Khanda 等，《Extending Group Relative Policy Optimization to Continuous Control: A Theoretical Framework for Robotic Reinforcement Learning》，arXiv:2507.19555v1，2025](https://arxiv.org/abs/2507.19555)。本文把连续控制 GRPO 写成一个框架，包含轨迹策略聚类、状态聚类优势、组自适应裁剪、时间平滑、组间多样性及参考策略更新。论文的初步实验只在 HalfCheetah-v4 使用单一随机种子；目前不能据此断言方法在 CarRacing 上有效，也不能把 arXiv 预印本称为已经发表在顶会或顶刊。

本项目把 CLI 名称定为 `grpo-c`，与原有 `grpo` 共存。检查点会记录 `grpo_c_variant=pixel8_dbscan_multipolicy_v1`。这是论文框架的独立、可执行 CarRacing 实例，未声称逐行复现作者代码。

## 方法具体如何落地

|论文组件|本项目实现|
|---|---|
|多个策略|每个环境持有一套独立的高斯 CNN 策略与 Adam；一轮每个策略在同一赛道种子跑一局|
|轨迹特征|平均每决策回报、高斯策略熵、连续动作方差、相对参考策略的解析 KL|
|轨迹聚类|标准化四维特征后执行确定性 k-means；默认 K=2|
|状态聚类|对 4 帧 64×64 灰度观测取帧均值并压缩为 8×8 特征，在每个策略组内执行 DBSCAN；噪声点使用组平均基线|
|状态优势|按 `gamma` 从整局奖励算每步 return-to-go，减去所在状态簇的平均值，之后在各策略组内标准化|
|自适应裁剪|`clip_g = clip_base × max(1, std_g/(std_all+1e-8))`；`std_g` 是该组未标准化优势的标准差|
|时间平滑|对同一局相邻观测的策略高斯均值加 L2 距离惩罚|
|组间多样性|各组策略在赛道初始观测的平均高斯均值作为可微代理中心，对高余弦相似度加惩罚|
|参考策略|初始化为首个策略；每轮将表现较好的一半策略做参数均值，再以系数 `tau` 平滑更新参考；计算当前策略到参考高斯的解析 KL|

8×8 像素特征、可微动作中心和参数空间参考混合都是在论文未指定 CarRacing 实现细节处做的工程选择。状态 DBSCAN 会产生额外 CPU 开销。默认超参数只是启动实验的设置，没有经过调参验证。

算法不训练价值网络。`grpo-c` 与之前的 `grpo` 不同：旧实现只有单一共享策略、同赛道多条轨迹及整局组相对回报；新实现训练独立策略，使用每步状态优势和论文提出的聚类、裁剪及正则项。

验证时会在完全相同的赛道上评估所有成员，并选择验证均分最高的成员保存为可直接录视频的 `best.pt` 或 `last.pt`。这使其比单策略 PPO 有更多选择机会；课程报告需要写明，不能把一次验证选出的成员当作未经选择的固定策略。`member_validation.csv` 保留全部成员数据。

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

这会评估两个初始策略，各在同一赛道采集一条完整轨迹，执行一次多策略更新，再评估终点。完整组可能略超 500 决策，查看 `finished.json` 中实际的 `steps` 和 `budget_overshoot`。短跑只能验证管线。

本机一次两策略自检实测：500 决策训练约 59 秒，约 8.4 决策/秒；两次全成员验证共 4 局约 220 秒；录制 1 局完整原始帧视频耗时约 21 分钟。这些只是当次观测值，不能直接外推到 4 策略和长时间训练。尤其视频录制较慢，建议只给选定模型和少量代表性赛道录视频。10 万决策可能需要数小时，100 万可能需要数十小时，应先观察 pilot 的 `updates.csv` 和 `finished.json` 再安排正式实验。

## 建议先跑 10 万决策试验

```bat
python -m racing.train --algo grpo-c --seed 0 --steps 100000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-state-eps 0.65 --grpo-c-state-min-samples 4 --grpo-c-epochs 2 --grpo-c-beta 0.01 --grpo-c-smooth 0.01 --grpo-c-diversity 0.01 --grpo-c-diversity-threshold 0.2 --grpo-c-reference-tau 0.05 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 25000 --eval-episodes 5 --out runs_grpo_c_lap95_pilot
```

`--num-envs 4` 表示 4 个独立策略；`--grpo-c-clusters 2` 把四条轨迹分成最多两组。训练赛道种子来自 0–999999；验证赛道固定从 1000000 开始。每轮所有策略在同一赛道起点采样。`--completion-mode original` 可切换回原环境规则，须换新的输出目录。

重点看：`updates.csv` 中 `occupied_policy_clusters`、`state_clusters`、`state_noise_fraction`、`advantage_std`、`clip_mean`、`approx_kl`、`reference_kl`，以及 `member_validation.csv` 中各成员成绩。如果经常只有一个占用的策略组，或绝大多数状态落入 DBSCAN 噪声，需要检查轨迹特征或状态半径；不能只凭总体回报判断聚类在发挥作用。

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

若从 pilot 的 `last.pt` 续训到累计 100 万，可用相同训练参数，把 `--steps` 设为 `1000000`，加 `--resume runs_grpo_c_lap95_pilot\grpo-c_s0\last.pt`，并将 `--out` 换成一个新目录。关键聚类、正则化和环境参数必须一致。续训会重新评估所有成员并建立新目录的 best；环境回合本身重新开始。

来源代码：[grpo_c.py](../racing/grpo_c.py)、[train.py](../racing/train.py)、[evaluate.py](../racing/evaluate.py)。
