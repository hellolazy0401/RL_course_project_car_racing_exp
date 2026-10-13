# GRPO：同赛道分组的连续控制适配版

已加入 `python -m racing.train --algo grpo`，也可在 suite 中用 `--algos grpo`。支持 original/lap95 完赛模式、CUDA、检查点续训、评估、视频、TensorBoard 和报告。依赖不变，无需重装环境。

## 算法定义与边界

来源：[DeepSeekMath，Shao 等，2024](https://arxiv.org/abs/2402.03300)，第 4.1 节提出 GRPO：同一问题采样一组答案，用组内奖励作相对优势，不需要价值模型。本实现将“同一问题”映射为“同一赛道种子与初始状态”，将“答案”映射为完整驾驶回合。它是从 LLM 方法向 CarRacing 的独立适配，不是原论文赛车基准复现，也不是 2025/2026 年的新算法；课程项目的 ICML 2025 VPG-repeat 可继续保留。

1. 每组 G 个环境使用同一赛道种子 reset；各自独立采样连续动作，直到正常终止或 1000 原始帧时限结束。结束的成员暂停，不自动重置加入下一局。
2. R_i 是整局原始奖励之和；A_i=(R_i-组内均值)/(组内总体标准差+1e-8)。该轨迹所有决策使用同一个优势。奖励相同的组得到零优势。
3. 各轨迹先对自己的决策取平均，再对轨迹取平均。因此短局不会因样本较少而权重更小。使用存储的 tanh 前动作及旧 logprob 计算裁剪策略目标。
4. 每次更新默认采集 2 组、每组 4 局；用微批次累积完整目标梯度，每个 GRPO epoch 做一次 actor 优化。默认 4 个 epoch，即每组批数据通常 4 次优化，不能与 PPO 的小批量更新次数直接等同。
5. 不创建、不调用、不训练 critic；不用 GAE，也不对回合终点进行价值 bootstrap。这里优化的是最多 1000 帧的完整回合成绩，gamma/gae-lambda/reward-scale/critic-lr/ppo-epochs/rollout/value-steps/normalize-adv 这些共享解析器字段不参与 GRPO 计算。不要用它们调 GRPO。
6. `--grpo-beta` 控制可选的参考策略 KL 惩罚，默认 0。若大于 0，冻结训练初始化时的策略，用当前策略与该固定参考策略的解析高斯 KL（current || reference），不是 PPO 的 KL 提前停止，也不是每轮重置的旧策略。可逆 tanh/缩放映射在数学上保留 KL。解析 KL 替代论文中的采样估计，是连续高斯策略适配选择。

本项目从随机初始化训练，缺少原语言模型场景中的监督微调起点，默认关闭对随机初始策略的参考惩罚（beta=0）。这是明确的无参考 KL 变体，不能声称与原论文超参数完全相同。可试 beta=0.01 作单独消融，但尚无证据表明它适合本任务。

同一赛道重复 G 次会降低单位预算内不同赛道的数量；整局奖励的信用分配比 GAE 更粗。方法新加入不代表能解决之前的驾驶退化，也不能预先保证优于 PPO。

## 1. 环境准备

在 Anaconda Prompt：

```bat
conda activate rl_carracing
cd /d D:\Desktop\RL_project
```

## 2. 可选流程自检

```bat
python -m racing.suite --preset smoke --algos grpo --completion-mode lap95 --device cuda --out runs_grpo_smoke
```

套件自检使用自己的小预算，生成测试视频和报告，不证明训练效果。已有目录不可覆盖，重跑时换 --out。

## 3. 建议先跑 10 万决策试验

```bat
python -m racing.train --algo grpo --seed 0 --steps 100000 --device cuda --num-envs 4 --grpo-group-size 4 --grpo-groups-per-update 2 --grpo-epochs 4 --grpo-beta 0 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 25000 --eval-episodes 10 --out runs_grpo_lap95_pilot
```

默认不启用 PPO target-kl。`--num-envs` 必须等于 `--grpo-group-size`，且至少为 2。组间随机抽取训练赛道种子 0–999999；验证仍使用 1000000 起的固定赛道。

关键日志：groups.csv 给出每组赛道、实际决策数、回报均值及组内标准差；train_episodes.csv 给出各组成员的回合成绩；updates.csv 包含策略损失、相对于采样策略的 approx_kl、参考 KL、组奖励标准差、零优势比例及吞吐。若大量组的回报完全一致，组相对奖励不能提供有效区分信号；不要仅增加 epoch。

## 4. 验证视频

```bat
python -m racing.evaluate --checkpoint runs_grpo_lap95_pilot\grpo_s0\best.pt --device cuda --episodes 5 --seed-start 1000000 --videos 5 --out runs_grpo_lap95_pilot\grpo_s0\validation_video_best
python -m racing.evaluate --checkpoint runs_grpo_lap95_pilot\grpo_s0\last.pt --device cuda --episodes 5 --seed-start 1000000 --videos 5 --out runs_grpo_lap95_pilot\grpo_s0\validation_video_last
```

评估自动加载仅策略网络，并继承检查点中的完赛模式。best 可以停留在起点，故建议同时查看 last。

## 5. 如需正式从零训练 100 万决策

```bat
python -m racing.train --algo grpo --seed 0 --steps 1000000 --device cuda --num-envs 4 --grpo-group-size 4 --grpo-groups-per-update 2 --grpo-epochs 4 --grpo-beta 0 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 100000 --eval-episodes 20 --out runs_grpo_lap95
```

为了保留完整轨迹，最后一组会跑完，实际预算可能最多超出 G×250−1 个决策（G=4 时最多 999）。finished.json 记录 requested_steps、steps 和 budget_overshoot。原始帧数也记录实际值。不能把实际 1000500 步写成精确 1000000 步。

本地时间应先用 pilot 的实际吞吐估算，GRPO 没有 critic 不等于整项实验必然更快。完整组需要等待最慢成员结束，环境仿真也仍耗时。

## 6. 独立测试与报告

方案确定后执行：

```bat
python -m racing.evaluate --checkpoint runs_grpo_lap95\grpo_s0\last.pt --device cuda --episodes 50 --seed-start 3000000 --videos 5 --out runs_grpo_lap95\grpo_s0\test
python -m racing.evaluate --checkpoint runs_grpo_lap95\grpo_s0\best.pt --device cuda --episodes 50 --seed-start 3000000 --videos 5 --out runs_grpo_lap95\grpo_s0\test_best
python -m racing.report --root runs_grpo_lap95 --allow-unequal-budgets
python -m racing.report --root runs_grpo_lap95 --selection test_best
start "" "runs_grpo_lap95\report\index.html"
```

若要测试 pilot，同步将上面的 runs_grpo_lap95 换为 runs_grpo_lap95_pilot。相同种子、相同完赛模式和相同测试赛道才便于比较。报告列出实际检查点步数范围；显式 --allow-unequal-budgets 允许展示 GRPO 的完整组预算差异，但不能称为严格等预算比较。价值误差/解释方差图对 GRPO 留空，而不是伪造一个零误差。

## 7. 从 pilot 续训（可选，和第 5 步二选一）

```bat
python -m racing.train --algo grpo --seed 0 --steps 1000000 --device cuda --num-envs 4 --grpo-group-size 4 --grpo-groups-per-update 2 --grpo-epochs 4 --grpo-beta 0 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 100000 --eval-episodes 20 --resume runs_grpo_lap95_pilot\grpo_s0\last.pt --out runs_grpo_lap95_resumed
```

累计目标为 100 万，不是追加 100 万。续训恢复策略、优化器、随机数状态及可选固定参考网络；关键优化/分组参数必须一致。新目录重新评估起点并选择 best。中断时未完成的组不保留轨迹，恢复后重新采样；不保证位级无缝续跑。原来的 PPO/A2C/VPG 检查点不能直接当作 GRPO 严格续训检查点。

## 源码与验证

- racing/grpo.py：采集同赛道完整回合、组相对优势、等轨迹权重、无 critic 更新、参考 KL、检查点与训练日志。
- racing/train.py：共享 --algo grpo 命令行入口。
- racing/envs.py：显式种子 reset、暂停结束成员，避免自动重置污染组。
- racing/model.py 与 evaluate.py：仅策略网络及兼容旧检查点评估。
- suite.py、report.py、scripts/diagnose_driving.py：批量运行、预算标注、诊断兼容。

25 项测试已通过，包括组标准化、零方差组、等轨迹权重、微批次梯度一致性、参考网络冻结、同种子真实环境重置、提前结束成员不再 step，以及所有原有测试。正式长训练尚未启动。

2026-10-13 本机端到端自检：CUDA、lap95、G=2、每批 1 组、2 个 epoch、beta=0.01，完成 1000 决策训练，评估并生成 MP4/HTML；再从检查点续训到 1500 决策，参考策略随检查点正确保留。短跑组奖励标准差约 1.67–1.79，每次更新 critic_updates=0。原始自检目录为 runs_grpo_selfcheck 与 runs_grpo_resume_selfcheck，均不是有效驾驶成绩结论。
