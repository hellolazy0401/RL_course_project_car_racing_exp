# GRPO-C 停滞诊断、v2 修复与重新实验

本文对应 `grpo_c_variant=pixel_motion_global_loo_mixture_v2`。源码已经修改在 `D:\Desktop\RL_project`，无需重建或重新安装现有 `rl_carracing` Conda 环境。重新训练时使用新的输出目录。原有 `runs_grpo_c_lap95_main` 是 v1 实验数据，不覆盖，也不接着训练。

**本次结论：已修复明确的实现问题，但稳定学习尚未解决。**已完成 100,116 决策的复核，最终选中模型在十条验证赛道上均分 −35.59、覆盖率 6.44%、完赛率 0%。曾出现更高分，但视频仍有明显草地绕圈行为。下列命令用于复现和诊断，不是已经验证能完赛的训练配方；目前不建议直接重跑 100 万决策。

**最新短实验已完成：**`runs_grpo_c_v2_pilot` 再次复现涨分后退化，不建议原配置续训。第 3 节的 `drive` 命令现在仅作历史复现。下一步建议执行第 9 节的 `standard` 初始化单因素对照，保持 2 万决策预算；其效果尚未验证。

## 1. 原实验实际发生了什么

检查对象：`runs_grpo_c_lap95_main\grpo-c_s0`。这次配置已经是 `actor_lr=1e-4`，因此不能把停滞解释为仍使用了 `3e-4`。

|检查项|实测结果|解释|
|---|---|---|
|已采集决策|中断检查点为 400,043；最终完整验证对应 300,043|没有实际跑满请求的 1,000,000 决策|
|选中成员验证均分|0、100,371、200,371、300,043 步都是 −28.677346|最佳验证分数没有提升|
|选中成员验证覆盖率|上述各阶段都是 7.1323%|驾驶进展没有明显改善|
|最佳检查点|`best.pt` 对应 0 步|当前最佳模型仍是初始化模型|
|策略更新|401 次更新平均 KL 约 0.001974，最高约 0.014954|参数有更新，但更新没有转化为有效驾驶|
|训练覆盖率|前 10 万决策平均约 7.42%；30–40 万约 7.89%|随机采样表现只有很小改善|
|状态簇数|平均 2.012，策略组数固定为 2|几乎每个策略组只有一个大状态簇|
|探索尺度|40 万步四个策略的 `logstd` 仍接近初始化 −0.5，最大偏差约 0.013|没有发现标准差塌缩|

对 400,043 步检查点的 member 0 在验证赛道 `1000000` 重新运行，得到覆盖率 6.9767%、回报 −30.23256，最终在 1000 原始帧时限结束。平均油门为 0.516、刹车为 0.466；member 1 的平均油门为 0.577、刹车为 0.278，覆盖率也为 6.9767%。初始网络输出的油门和刹车都接近 0.5。

原 8×8 像素特征在这条实际轨迹上的相邻状态距离中位数为 0.0314，而 `eps=0.65`。DBSCAN 可以通过连续相邻状态连接出一个大簇，即使两个远隔时刻的画面差别较大。这使“相似状态的相对回报”退化成了很粗的比较。

我们还做了一个反事实诊断：冻结模型，只把执行时的刹车强制设为 0。member 0 的覆盖率从 6.98% 降到 2.66%，member 1 从 6.98% 降到 3.99%；初始模型取消刹车后 274 帧驶出边界。这说明同时踩油门和刹车是需要处理的初始化与探索问题，但取消刹车本身不能让策略学会转弯。此诊断不计入正式算法成绩。

## 2. 本次代码修复

1. **状态基线使用所有策略的轨迹，并排除被评分轨迹。**同一条轨迹不再把自己的回报放进自己的比较基线。先在全体轨迹中聚类，再在同一状态簇、同一时间段中寻找其他轨迹作为参考，最后在策略组内标准化优势。
2. **匹配相近的回合时间。**按 16 个决策划分时间段，避免在固定时限下，把开头尚有很长未来的 return-to-go 与结尾剩余很少时间的值直接混合。没有同状态簇的其他轨迹时，退回相同时间段的其他轨迹；没有任何对应轨迹时使用零基线。
3. **保留当前画面和运动变化。**状态特征使用当前帧的 8×8 表示，以及当前帧与最早堆叠帧的差值。此前对四帧求平均会丢失运动信息。特征经过带方差下限的标准化，再按维度数的平方根缩放。虽然 `--grpo-c-state-eps 0.65` 的数值沿用旧设置，其距离空间已经改变，不能把 v1、v2 的半径视为相同物理含义。
4. **处理 DBSCAN 退化。**最大状态簇占比超过 90%，或噪声点超过 50% 时，改用最多 16 个 k-means 状态簇。参数为 `--grpo-c-state-fallback-clusters 16`；设为 `0` 可关闭回退，用于消融实验。16 只限制回退算法的簇数，正常 DBSCAN 的簇数可以超过 16。完全相同的观测仍可能只占用一个簇，不人为按时间编号制造视觉差异。
5. **合并只有一条轨迹的策略组。**防止单成员组缺少相对比较对象。`--grpo-c-clusters 2` 是请求的最多组数，实际可能合并为一组；日志记录合并次数。
6. **参考策略改为动作分布混合。**`ReferenceMixture` 为每个独立策略保存自己的冻结参考副本，分别做平滑更新，再混合它们的动作概率密度。不同随机初始化网络的隐藏单元没有一一对应关系，因此不再跨成员平均网络权重。参考权重向本轮回报较好的半数成员平滑移动。当前策略到混合分布的 KL 使用确定性高斯求积近似。
7. **平滑多样性代理的归一化。**原来直接使用余弦相似度，在接近零的动作均值附近可能产生过大的梯度。现在先做 `m / sqrt(norm(m)^2 + 1)`，再对组中心内积超过阈值的部分加惩罚。这是平滑的代理目标，与原余弦相似度不完全等价，需要在报告中说明；现有证据不足以认定多样性项是旧训练停滞的主要原因。
8. **增加低初始刹车配置。**默认 `--grpo-c-init drive` 将初始确定性刹车设为约 0.02；油门约 0.5，转向约 0。仍保留全部三个连续动作维度，刹车由网络学习，没有强制关闭。此先验可能促进车辆移动，也可能让未学会转弯的车辆更快驶出边界；并不保证更高初始得分。`--grpo-c-init standard` 保留原来油门和刹车都约为 0.5 的初始化，适合做消融。实际两赛道验证中 v2 初始均分约 −57.42，低于旧模型在相同两赛道的 −31.25；第 8 节给出后续训练变化。

上述内容是项目对论文框架的像素连续控制适配和稳定性修复，并非作者官方实现。特别是运动像素特征、时间分组、排除本轨迹的基线、聚类回退和动作初始化都需要在课程报告中披露。它们是否改善长期学习必须以新实验为准。

此外，`racing.suite` 支持显式指定 `--actor-lr`；省略时 GRPO-C 使用 `1e-4`，其他算法保持 `3e-4`。使用 `--skip-completed` 时会检查已有学习率和 GRPO-C 版本，拒绝把旧实验当成新设置的结果直接跳过。当前单独训练命令的停滞不能归因于这个套件问题。

这次首先修复已发现的实现问题，再整体检查学习曲线。尚未逐项做消融，不能把效果变化定量归因于其中某一个改动。

## 3. 已完成的 drive 短实验命令（历史复现）

下面的实验已经完成并分析，不需要重复执行，也不建议把它直接延长至 100 万决策。需要复现历史结果时，在 **Anaconda Prompt** 中执行；下一项诊断实验见第 9 节。

```bat
conda activate rl_carracing
cd /d D:\Desktop\RL_project
python -m racing.train --algo grpo-c --seed 0 --steps 20000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-state-eps 0.65 --grpo-c-state-min-samples 4 --grpo-c-state-fallback-clusters 16 --grpo-c-init drive --grpo-c-epochs 2 --grpo-c-beta 0.01 --grpo-c-smooth 0.01 --grpo-c-diversity 0.01 --grpo-c-diversity-threshold 0.2 --grpo-c-reference-tau 0.05 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 5000 --eval-episodes 3 --out runs_grpo_c_v2_pilot
```

这一步训练 4 个独立策略，每轮在同一条赛道上各跑完一局后更新；总决策预算约为 20,000，不是每个策略各 20,000。因为采集完整轨迹组，实际预算可能略超目标。

训练开始先评估所有初始成员，之后每约 5,000 决策在固定的 3 条验证赛道上评估。训练输出保存在 `runs_grpo_c_v2_pilot\grpo-c_s0`，可检查：

|文件|用途|
|---|---|
|`config.json`|确认版本为 `pixel_motion_global_loo_mixture_v2`、初始化为 `drive`|
|`initial.pt`|保存 0 步模型，衡量初始化本身的表现|
|`best.pt`|截至当前，验证集分数最高的成员检查点|
|`last.pt`|最近完成验证的成员及可继续训练的整套状态|
|`validation.csv`|每次验证选中成员的回报、覆盖率、完赛率|
|`member_validation.csv`|所有成员的验证成绩，避免只看被选中的成员|
|`updates.csv`|策略更新、状态聚类、回退与参考策略诊断|
|`train_episodes.csv`|训练回合成绩及策略分组|
|`finished.json`|完整结束后记录实际预算和训练耗时|

`--completion-mode original` 可切换回原完赛规则，须同时换一个新的输出目录，例如 `runs_grpo_c_v2_original_pilot`。只学会覆盖约 7% 时尚未接近完赛阈值，切换完赛规则不能解决当前的基础驾驶问题。

同名训练目录已存在时程序会拒绝覆盖。如果再次试跑，把 `--out` 改为 `runs_grpo_c_v2_pilot_r2`，并相应替换后续命令中的目录名。

## 4. 如何判断是否真的在学习

先看 `validation.csv` 中 **0 步到训练后**的回报和覆盖率变化，再看四个成员是否都有改善趋势。即使某种新初始化得分比 v1 高，也不等于训练取得了进步；本次实测的新初始化得分反而较低。第 3 节建议的短跑只有一个训练种子和三条验证赛道，只能用于筛查；不能据此宣布算法优于 PPO。

同时检查 `updates.csv`：

- `state_dbscan_largest_fraction`、`state_dbscan_noise_fraction` 显示 DBSCAN 的原始退化情况；`state_fallback` 表示是否启动回退。
- `state_clusters` 是最终状态簇数；`state_peer_fraction` 显示有多少状态找到了同簇、同时段的其他轨迹作为参考。簇多本身也不代表基线有效。
- `merged_singleton_groups` 与 `occupied_policy_clusters` 显示策略分组是否合并。
- `approx_kl`、`reference_kl`、`clip_fraction` 用来诊断更新幅度，不能当作学习效果。混合分布 KL 是数值近似，极小负值可能来自求积误差。

只有回报和覆盖率在多个验证点上超过各自初始水平、动作与轨迹也有相应改善，再考虑把同样配置扩到 100,000 决策和更多训练种子。不要仅因损失下降或 KL 非零就直接投入 1,000,000 决策。

## 5. 分别查看初始、最后和最佳模型

先使用验证赛道，避免在调参阶段反复查看最终测试集。以下三个命令不录视频，输出目录互不重复。

```bat
python -m racing.evaluate --checkpoint runs_grpo_c_v2_pilot\grpo-c_s0\initial.pt --device cuda --episodes 3 --seed-start 1000000 --videos 0 --out runs_grpo_c_v2_pilot\grpo-c_s0\validation_initial
python -m racing.evaluate --checkpoint runs_grpo_c_v2_pilot\grpo-c_s0\last.pt --device cuda --episodes 3 --seed-start 1000000 --videos 0 --out runs_grpo_c_v2_pilot\grpo-c_s0\validation_last
python -m racing.evaluate --checkpoint runs_grpo_c_v2_pilot\grpo-c_s0\best.pt --device cuda --episodes 3 --seed-start 1000000 --videos 0 --out runs_grpo_c_v2_pilot\grpo-c_s0\validation_best
```

每个目录会有 `summary.json` 和 `episodes.csv`。`best.pt` 可能仍是 0 步模型，检查摘要中的 `checkpoint_steps`。选最好成员和最好验证时刻会带来选择优势，比较其他算法时必须披露。

如需一段驾驶视频，再单独运行：

```bat
python -m racing.evaluate --checkpoint runs_grpo_c_v2_pilot\grpo-c_s0\best.pt --device cuda --episodes 1 --seed-start 1000000 --videos 1 --out runs_grpo_c_v2_pilot\grpo-c_s0\validation_video_best
```

视频保存在 `validation_video_best\videos\track_1000000.mp4`，采用原始帧率。录制会额外耗时。每次输出目录必须是新目录，重复执行可将末尾改为 `validation_video_best_r2`。

## 6. 配置确定后生成测试结果、报告和折线图

这一步用于一次完整短实验的结果归档。若继续根据短跑结果调整实现或参数，请把这些赛道视为开发集，并为最终论文结论预留一段未查看过的赛道种子。

```bat
python -m racing.evaluate --checkpoint runs_grpo_c_v2_pilot\grpo-c_s0\last.pt --device cuda --episodes 10 --seed-start 3000000 --videos 0 --out runs_grpo_c_v2_pilot\grpo-c_s0\test
python -m racing.evaluate --checkpoint runs_grpo_c_v2_pilot\grpo-c_s0\best.pt --device cuda --episodes 10 --seed-start 3000000 --videos 0 --out runs_grpo_c_v2_pilot\grpo-c_s0\test_best
python -m racing.report --root runs_grpo_c_v2_pilot --allow-unequal-budgets
python -m racing.report --root runs_grpo_c_v2_pilot --selection test_best
start "" "runs_grpo_c_v2_pilot\report\index.html"
start "" "runs_grpo_c_v2_pilot\report_test_best\index.html"
```

希望报告内直接包含一段测试视频时，在**首次执行**相应测试命令前，把该行 `--videos 0` 改为 `--videos 1`。已经生成过测试目录则使用新的输出目录单独录制，不覆盖原结果。

报告目录包含 `index.html`、`learning_curve.png`、`frame_curve.png`、`time_curve.png`、`diagnostics.png`、`test_seeds.png`、`per_run.csv`、`all_test_episodes.csv`、`validation_curves.csv` 等。前三张曲线分别以策略决策数、原始帧数和训练秒数为横轴。GRPO-C 没有价值网络，对应价值指标为空是正常情况。

这里只做一个训练种子，报告不会给出有效的跨训练种子置信区间。完整组采样可能超出请求步数，`--allow-unequal-budgets` 只是允许透明展示实际预算，不能把不相等预算称为严格公平比较。v1 和 v2 的超参数及实现不同，放在不同报告根目录中。

## 7. 旧模型与续训兼容性

**v2 拒绝从 v1 检查点续训。**优势估计和参考策略结构均已改变，不能把旧优化器状态接入后当作原实验的自然延长。请使用第 3 节命令从零开始，保留 v1 数据用于失败分析。

旧模型仍可执行评估，例如：

```bat
python -m racing.evaluate --checkpoint runs_grpo_c_lap95_main\grpo-c_s0\interrupted.pt --device cuda --episodes 3 --seed-start 1000000 --videos 0 --out runs_grpo_c_lap95_main\grpo-c_s0\v1_validation_after_fix
```

该命令只使用原检查点的策略参数和环境配置。它不会把旧模型转换成 v2。v2 自己的续训需要相同版本和训练参数，并使用新的输出目录；`--steps` 表示累计预算。

续训目录的 `best.pt` 只表示**该段续训中**评估过的最佳检查点，不会自动继承原目录更早的最佳成绩。比较整个训练过程时，要同时保留各段目录的 `best.pt` 和验证日志。真正 0 步的 `initial.pt` 位于最初从零训练的目录；续训起点的评估不能当作初始化成绩。

## 8. 本次修复后的实测结果

最终完整检查通过 **45 个测试，耗时 74.14 秒**，命令为 `python -m pytest tests -q`。其中包含新增的 9 项套件配置检查；最后一次测试与诊断训练并行，耗时受机器负载影响。测试通过说明这些检查覆盖的计算和兼容性行为符合预期，不代表模型已经收敛。

先从零运行 `runs_grpo_c_v2_check\grpo-c_s0`，请求 12,000 决策，实际完成 12,401；然后在 `runs_grpo_c_v2_continue_check\grpo-c_s0` 续训，请求累计 30,000 决策，实际完成 30,972。两段使用同一训练种子 0、4 个独立策略及相同训练配置。验证只使用固定赛道 `1000000` 和 `1000001`，所以是初步诊断，不是最终多种子测试。

|累计决策数|选中成员|两赛道验证均分|平均覆盖率|完赛率|记录目录|
|---:|---:|---:|---:|---:|---|
|0|3|−57.42|7.20%|0%|`runs_grpo_c_v2_check`|
|6,697|2|89.17|18.92%|0%|`runs_grpo_c_v2_check`|
|12,401|2|−58.17|6.55%|0%|`runs_grpo_c_v2_check`|
|20,404|2|−57.87|6.55%|0%|`runs_grpo_c_v2_continue_check`|
|30,972|1|72.63|17.26%|0%|`runs_grpo_c_v2_continue_check`|

6,697 步曾明显超过初始化，但 12,401 和 20,404 步又退化。30,972 步选中成员切换为 member 1，再次获得正回报。它表明 v2 已出现可观测的驾驶改善，也表明训练仍不稳定：不能只展示 89.17 的最高点，也不能把不同成员之间的择优曲线描述成同一个策略的单调进步。所有验证仍未完赛。

原 v1 的 −28.68 是十条赛道均值，与这里两条赛道不是同一评估协议。在相同两条赛道上，v1 初始均分为 −31.25、覆盖率 6.88%；v2 初始均分为 −57.42、覆盖率 7.20%。因此低刹车初始化没有直接提高这里的初始回报，后续高分也不能全归因于初始化。

两段的平均最终状态簇数分别约为 14.15、12.45；同状态簇同时段的其他轨迹匹配率分别约为 80.94%、79.43%。这两段均未触发 DBSCAN 回退，说明目前看到的多个状态簇来自更新后的特征和正常 DBSCAN，而不是强制设定了 16 个簇。在 30,972 步最后一轮中，最终状态簇数为 18、匹配率为 70.8%，亦可验证 16 不是正常 DBSCAN 的簇数上限。

累计训练时间为 440.81 秒，包含采样、聚类和优化，排除验证；两段验证额外累计约 279.50 秒。两段会话墙钟时间合计约 721.22 秒，即约 12.0 分钟。这是本次机器状态下的短跑测量，不能直接保证其他规模或录视频时的速度。

第一段的 `best.pt` 位于 6,697 步；第二段的 `best.pt` 位于 30,972 步。后者是续训段内最佳，回报 72.63 仍低于第一段保留的 89.17。0 步检查点仍应从 `runs_grpo_c_v2_check\grpo-c_s0\initial.pt` 读取。

### 扩大验证与驾驶行为检查

将初始、6,697 步和 100,116 步的三个模型固定后，额外使用 `racing.evaluate --device cpu --episodes 10 --seed-start 1000000 --videos 0` 评估。这里使用 CPU，而上表训练过程中的验证使用 CUDA；浮点细小差异可能随物理仿真累积，所以十赛道结果单独列出。三者使用同一评估设备、相同十条赛道和相同环境规则。

|固定模型|十赛道均分|平均覆盖率|完赛率|
|---|---:|---:|---:|
|v2 初始模型，member 3|−54.59|7.48%|0%|
|v2 6,697 步模型，member 2|77.33|17.73%|0%|
|v2 100,116 步模型，member 3|−35.59|6.44%|0%|

前两个结果目录分别为第一段训练目录中的 `validation10_initial`、`validation10_best_6697`；最终结果在 `runs_grpo_c_v2_100k_check/grpo-c_s0/validation10_last`。这仍是开发阶段的验证，不能当作最终未见测试集成绩；成员是先按两条验证赛道选出的，没有在十条赛道上重新选成员。它与原命令每次在十条赛道上挑选四个成员的选择协议有差异，不能直接称为严格公平的 v1/v2 优劣比较。

最终模型的十赛道覆盖率低于新初始化，虽然回报略有提高；早期最佳模型的优势没有保持住。因此这轮复核不能支持“稳定学习问题已解决”的结论。

另外录制了 6,697 步模型在赛道 `1000000` 的视频，并在原有 1000 帧限制内记录轨迹。该局回报为 42.86、覆盖率 14.29%，但 **78.9% 的帧四个车轮全部离开路面**。转向始终为负，约在 −0.0202 至 −0.0085 之间，车辆主要在草地上绕大圈，尚未根据道路形状有效转弯。平均油门 0.488、刹车 0.0179，同时大于 0.2 的比例为 0。可见初始油门刹车冲突已缓解，但沿赛道驾驶仍未学会；回报变高不等于控制任务已经解决。

已有产物（均相对项目根目录）：

- 视频：`runs_grpo_c_v2_check/grpo-c_s0/validation_video_best/videos/track_1000000.mp4`
- 轨迹图：`runs_grpo_c_v2_check/grpo-c_s0/driving_diagnosis_best_6697/path_1000000.png`
- 动作与覆盖率曲线：同一诊断目录下的 `trace_1000000.png`
- 视频关键帧：同一诊断目录下的 `frames_1000000.jpg`
- 原始逐帧数据：同一诊断目录下的 `telemetry_1000000.csv`

重新生成时请使用新的输出目录。轨迹诊断命令中的 `--limit 1000` 与正式环境时限一致，不延长回合：

```bat
python scripts\diagnose_driving.py --run runs_grpo_c_v2_check\grpo-c_s0 --device cpu --limit 1000 --seed-start 1000000 --episodes 1 --video-subdir validation_video_best --out-name driving_diagnosis_best_6697_r2
```

### 已完成：累计 100,116 决策检查

从累计 30,972 决策续训到请求的 100,000 决策，实际完成 **100,116 决策、400,377 原始帧**，输出目录为 `runs_grpo_c_v2_100k_check\grpo-c_s0`。训练配置保持一致，仍使用上述两条验证赛道，每约 20,000 决策验证一次。核心采样、策略、参考分布和环境源码的 SHA-256 与三段运行的记录均一致。

|累计决策数|选中成员|两赛道验证均分|平均覆盖率|完赛率|
|---:|---:|---:|---:|---:|
|40,342|2|28.72|12.87%|0%|
|60,100|2|−59.55|7.53%|0%|
|80,863|2|−36.18|6.38%|0%|
|100,116|3|17.96|11.80%|0%|

最终四个成员的两赛道均分约为 −77.08、−67.26、−58.02、17.96。全程最高点仍为第一段 6,697 步的 member 2。第三段目录的 `best.pt` 对应续训起点 30,972 步的 member 1；该目录的 `last.pt` 才是 100,116 步的 member 3。不要把这两个文件混为同一时刻。

全程有 108 轮完整轨迹组更新，每个策略进行了 216 次优化器更新；不是四个策略各跑了 100,116 次决策。最终状态簇数范围为 6–28，按更新平均为 12.45；同簇同时段的其他轨迹匹配率平均 78.72%，没有触发聚类回退。状态簇不再整体退化，但这并不保证基线已经足够准确。

累计训练耗时约 **31.13 分钟**，三段训练过程中的验证累计约 **10.76 分钟**，会话墙钟时间合计约 **41.92 分钟**，不包括另外进行的十赛道复评和视频诊断。期间并行执行过 CPU 验证和测试，因此这是本次复核成本，不能当作独占机器时的速度基准。

完整曲线与原始汇总已生成在 `runs_grpo_c_v2_check/diagnosis_100k/`：

- `validation_learning_curves.png` / `.svg`：四个成员及每次选中成员的回报、覆盖率曲线，无平滑。
- `selected_validation.csv` / `member_validation.csv`：合并后的数据，续训边界不重复计数。
- `metadata.json`：验证协议、源目录和合并规则。
- `run_summary.json`：训练成本、聚类统计和核心源码一致性核对。

重新生成曲线时使用新的输出目录：

```bat
python scripts\plot_grpo_c_diagnosis.py --runs runs_grpo_c_v2_check\grpo-c_s0 runs_grpo_c_v2_continue_check\grpo-c_s0 runs_grpo_c_v2_100k_check\grpo-c_s0 --out runs_grpo_c_v2_check\diagnosis_100k_r2
```

脚本会检查训练参数、验证赛道和续训关系，拒绝把不同实验当作一条连续学习曲线。

### 本次三段复核的完整复现命令

下面会重新消耗约一次本节实验的预算。现有结果已经生成，不需要为了查看图表再次训练。为避免覆盖现有文件，使用新的 `runs_grpo_c_v2_reproduce_stage*` 目录；仍在前面激活的 Conda 环境及项目目录中运行。其余未列出的训练参数采用当前默认值，与已完成复核的 `config.json` 一致。

```bat
python -m racing.train --algo grpo-c --seed 0 --steps 12000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-init drive --grpo-c-epochs 2 --grpo-c-beta 0.01 --grpo-c-smooth 0.01 --grpo-c-diversity 0.01 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 6000 --eval-episodes 2 --out runs_grpo_c_v2_reproduce_stage1
python -m racing.train --algo grpo-c --seed 0 --steps 30000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-init drive --grpo-c-epochs 2 --grpo-c-beta 0.01 --grpo-c-smooth 0.01 --grpo-c-diversity 0.01 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 10000 --eval-episodes 2 --resume runs_grpo_c_v2_reproduce_stage1\grpo-c_s0\last.pt --out runs_grpo_c_v2_reproduce_stage2
python -m racing.train --algo grpo-c --seed 0 --steps 100000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-init drive --grpo-c-epochs 2 --grpo-c-beta 0.01 --grpo-c-smooth 0.01 --grpo-c-diversity 0.01 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 20000 --eval-episodes 2 --resume runs_grpo_c_v2_reproduce_stage2\grpo-c_s0\last.pt --out runs_grpo_c_v2_reproduce_stage3
python scripts\plot_grpo_c_diagnosis.py --runs runs_grpo_c_v2_reproduce_stage1\grpo-c_s0 runs_grpo_c_v2_reproduce_stage2\grpo-c_s0 runs_grpo_c_v2_reproduce_stage3\grpo-c_s0 --out runs_grpo_c_v2_reproduce_plots
python -m racing.evaluate --checkpoint runs_grpo_c_v2_reproduce_stage1\grpo-c_s0\initial.pt --device cpu --episodes 10 --seed-start 1000000 --videos 0 --out runs_grpo_c_v2_reproduce_stage1\grpo-c_s0\validation10_initial
python -m racing.evaluate --checkpoint runs_grpo_c_v2_reproduce_stage1\grpo-c_s0\best.pt --device cpu --episodes 10 --seed-start 1000000 --videos 0 --out runs_grpo_c_v2_reproduce_stage1\grpo-c_s0\validation10_best
python -m racing.evaluate --checkpoint runs_grpo_c_v2_reproduce_stage3\grpo-c_s0\last.pt --device cpu --episodes 10 --seed-start 1000000 --videos 0 --out runs_grpo_c_v2_reproduce_stage3\grpo-c_s0\validation10_last
```

### 仍未解决的问题与后续实验边界

当前每个策略每轮只使用一条完整轨迹。四个独立策略之间的比较、视觉状态聚类和组内标准化都来自很小的样本，这可能带来较大的梯度方差；像素状态特征也不等于经过训练的驾驶状态表示。它们是待验证的后续方向，现有数据尚不能把训练退化归因于其中单独一项。

后续算法结构的候选改动是“每个策略积累多条独立完整轨迹后再更新”，并与当前 v2 保持相同总决策预算。这需要真正修改采样和优势计算，当前尚未实现。最新建议先利用已有开关完成第 9 节的初始化对照，单独隔离未经验证的低刹车先验，再决定是否投入结构改造。只有多个训练种子上持续超过初始化，且轨迹显示能够沿道路转弯，才应扩大预算或宣称算法有效。

## 9. 最新 2 万决策实验的判断与下一条命令

检查目录：`runs_grpo_c_v2_pilot/grpo-c_s0`。实际完成 20,404 决策、88 条训练轨迹，各策略各 22 条轨迹、约 5,000 决策。

|决策数|三赛道验证均分|覆盖率|完赛率|
|---:|---:|---:|---:|
|0|−58.87|7.05%|0%|
|5,697|62.66|16.27%|0%|
|10,602|−58.54|7.05%|0%|
|15,001|−57.95|6.40%|0%|
|20,404|−59.28|6.40%|0%|

**决定：不按当前配置继续训练。**本次 22 轮更新中的损失、KL、参考 KL、状态簇数、同伴匹配率、训练回报/覆盖率、油门及刹车数值，与先前检查的相同步数完全一致；88 条轨迹的赛道种子和回报也一致。共同验证赛道在 0 和 20,404 步的四成员逐局结果一致。这是相同训练过程的复现，仅验证次数和赛道数量不同。先前该过程已经延长至 10 万决策，仍没有稳定学习，不需要重复消耗这段预算。

本次平均更新 KL 约 0.000534，最大约 0.001195，没有观察到大幅更新失控；状态簇为 8–28，同伴匹配率平均约 80.3%，亦没有全部退化成单一状态簇。这些指标只能排查部分问题，不能证明视觉状态表示和优势估计已经足够有效。

对这次 `best.pt`（5,697 步）重新录制了赛道 `1000000` 的视频，并在原 1000 帧限制下追踪车辆：覆盖率 15.61%，但 78.1% 的帧四轮全部离开路面，主要在草地绕大圈；平均油门 0.485、刹车 0.0181。因此这个正分模型也尚未学会沿道路驾驶。新产物：

- 曲线及合并数据：`runs_grpo_c_v2_pilot/assessment/`
- 视频：`runs_grpo_c_v2_pilot/grpo-c_s0/assessment_best_video/videos/track_1000000.mp4`
- 轨迹、动作曲线、关键帧及逐帧数据：`runs_grpo_c_v2_pilot/grpo-c_s0/assessment_driving_best/`

**下一项只改初始化：`drive` 改为 `standard`。**目的在于检验未学会转弯时，低刹车先验是否让车辆过早加速、减少有效道路样本。这是待验证的假设，不能预先称为解决方案；`standard` 仍使用全部 v2 状态基线和混合参考策略修复，并不是回退到旧 v1。

其他训练参数、验证频率、三条验证赛道和训练种子保持一致。使用新目录，从零开始，**不加 `--resume`**：

```bat
conda activate rl_carracing
cd /d D:\Desktop\RL_project
python -m racing.train --algo grpo-c --seed 0 --steps 20000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-state-eps 0.65 --grpo-c-state-min-samples 4 --grpo-c-state-fallback-clusters 16 --grpo-c-init standard --grpo-c-epochs 2 --grpo-c-beta 0.01 --grpo-c-smooth 0.01 --grpo-c-diversity 0.01 --grpo-c-diversity-threshold 0.2 --grpo-c-reference-tau 0.05 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 5000 --eval-episodes 3 --out runs_grpo_c_v2_standard_pilot
```

本次没有修改算法源码，也没有启动这项新的训练。`standard` 开关已存在。完成后首先比较它自己的 0 步和最终成绩，再与当前 `drive` 曲线比较。初始刹车本身可能改变初始回报，因此不能仅凭 `standard` 的初始分数高于 `drive` 就声称训练改善。关注最后若干验证点的回报和覆盖率是否持续提高，同时检查是否沿道路行驶。若再次只有早期峰值，或最终与初始化相当，不扩大预算，再进行每策略多轨迹采样的结构实验。

当前 `--batch-size` 只控制一条轨迹的计算分块，所有分块累积梯度后才更新；把 128 改成 256 或 512 并不会增加独立轨迹数。也不建议同时修改学习率、聚类参数、正则项和完赛规则，否则无法判断改变是否有效。

这次总耗时约 25.53 分钟，其中训练 10.03 分钟、验证 15.46 分钟，验证占约 60.6%。对照命令保留相同验证频率以便比较；这些时间是本次实测，不是下一次运行时长的保证。

相关代码：[grpo_c.py](../racing/grpo_c.py)、[grpo_c_reference.py](../racing/grpo_c_reference.py)。完整多算法实验产物流程见[算法与完赛逻辑对照](完整实验流程_算法与完赛逻辑对照.md)。

## 10. 分数平台的进一步定位与多轨迹累积实现

最新 standard 实验在 0、5,000、10,000、15,000 决策的三赛道验证均分均为 −32.7273，覆盖率均为 6.7273%。目录没有 finished.json；训练日志到 17,000 步，last.pt 是 15,000 步，另有 interrupted.pt，不能称为完成了 20,000 步。

对 initial.pt 和 last.pt 的同一成员（member 0），固定赛道 1000000、CPU、原 1000 帧限制复核：

|指标|初始|15,000 步|
|---|---:|---:|
|回报|−30.2326|−30.2326|
|覆盖率|6.9767%|6.9767%|
|平均转向|0.00159|−0.11607|
|平均油门|0.49888|0.54318|
|平均刹车|0.49954|0.42451|
|平均速度（环境单位）|6.2395|7.0839|
|最后一次新增路块的决策序号|168|167|

权重差的 L2 范数为 0.21357，明确不是模型没更新。该赛道访问路块数量相同、帧数相同，路块奖励与时间惩罚自然给出相同回报；动作已经改变，但没有转化为更多道路覆盖。这是分数完全相同的直接解释。油门和刹车持续同时施加、轨迹后段不再增加路块，是行为层面的症状；不能仅凭这一个赛道认定唯一训练根因。原始复核在 runs_grpo_c_v2_standard_pilot/grpo-c_s0/plateau_diagnosis.json。

原每轮四个策略各一条轨迹：17,000 总决策仅约每策略 4,250 决策、17 条轨迹。视觉状态基线和组内标准化样本少，跨策略比较可能有较大方差；这是待验证的估计问题。drive 版本虽减少油门刹车冲突，却仍在草地绕圈，因此单改初始化不能视为解决方案。

本次增加 --grpo-c-episodes-per-update，默认 1 保留原实验，设置 4 开启每策略四条完整轨迹后更新：

- 采样期间策略固定，每轮各成员仍在同一赛道比较，不把不同赛道混成状态基线。
- 各回合分别计算折扣回报与优势，再合并训练；平滑约束屏蔽回合重置边界。
- 策略分组特征、参考策略的成绩权重使用多个回合的均值。
- 每个策略按合并后的总决策数归一化梯度；微批大小只控制显存。
- 日志增加 episodes_per_policy 与 sampled_episodes；记录所有回合的种子和成员。
- 续训禁止改变该参数；旧检查点缺失该字段按 1 解释。实验矩阵跳过检查和曲线合并检查也区分该参数。

建议先以现有 drive 实验为基准，只改轨迹批量，使用新目录从零开始：

```bat
python -m racing.train --algo grpo-c --seed 0 --steps 20000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-init drive --grpo-c-episodes-per-update 4 --grpo-c-epochs 2 --grpo-c-beta 0.01 --grpo-c-smooth 0.01 --grpo-c-diversity 0.01 --actor-lr 1e-4 --batch-size 128 --completion-mode lap95 --eval-every 5000 --eval-episodes 3 --out runs_grpo_c_multitrack_pilot
```

同一总采样预算下，多轨迹批量会减少优化器更新次数；这是方差与更新频率之间的实验取舍，不能提前保证涨分。评估会落在完整批量结束处，最终预算也可能超出请求值，应比较实际决策数。不要为了恢复更新次数同时把 epochs 改成四倍，以免失去单因素对照。先看相对自身初始化的持续改善，再检查驾驶轨迹；短跑通过仅证明实现能运行，不代表收敛。暂不启动长训练。

本次验证：全量测试 46 项通过，随后新增的配置防混用检查连同 suite 测试 10 项通过。真实环境短跑使用 2 个策略、每策略 2 条轨迹，完成 1,000 决策和 4,000 帧，正常生成检查点及 finished.json；更新 KL 为 0.000319，数值有限。结果保存在 runs_grpo_c_multitrack_smoke/grpo-c_s0。该短跑只验证流程，不用于评价学习效果。

## 11. PPO 初始化 + GRPO-C 完赛微调（2026-10-16）

目标改为提高真实完赛率。用户已明确接受 PPO 预训练再由 GRPO-C 微调。本实验不是从零训练 GRPO-C，不能把预训练带来的能力算作 GRPO-C 自身收益。

初始化取 runs_lap95_lr1e4/ppo_s0/best.pt（PPO 第 800,256 决策），历史 20 条验证赛道完赛率 55%、覆盖率 91.17%。实际微调起点会重新验证；历史 20 条结果不能直接与本轮 5 条结果比较。

新增功能：

- --grpo-c-warm-start：严格检查环境配置，只迁移 actor 和动作标准差，不迁移 PPO critic、优化器和计数器。保存源文件 SHA-256、预训练步数与时间至 warm_start.json，并随续训检查点保留。迁移后不再施加 drive 初始化偏置。
- --grpo-c-selection completion：先比较完赛率，再比较覆盖率，最后比较原始回报；用于成员选择和 best.pt 保存。旧默认仍为 return。
- --grpo-c-completion-bonus 200：只在训练中对真实完赛额外加 200；达到 95% 覆盖但未满足终止规则、超时和出界均不奖励。训练日志同时保留原始回报与训练目标回报。
- 评估不加完赛奖励，不改变物理、1000 帧时限、图像输入或 lap95 规则。lap95 要求覆盖率严格大于 95% 且离开起点后正向重入，或者访问全部路块。

首次微调采用 1e-5 学习率、2 条轨迹/策略/更新、两轮更新、0.1 剪切与参考 KL 权重、0.01 参考更新率。关闭平滑和组间方向多样性惩罚，避免强迫已有道路驾驶策略产生无关的动作差异。保留 GRPO-C 的多策略采样、轨迹分组、状态相对优势、剪切目标和混合参考策略；没有引入 critic。

```bat
python -m racing.train --algo grpo-c --seed 0 --steps 12000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-episodes-per-update 2 --grpo-c-epochs 2 --grpo-c-warm-start runs_lap95_lr1e4/ppo_s0/best.pt --grpo-c-selection completion --grpo-c-completion-bonus 200 --grpo-c-beta 0.1 --grpo-c-smooth 0 --grpo-c-diversity 0 --grpo-c-reference-tau 0.01 --actor-lr 1e-5 --clip 0.1 --batch-size 128 --completion-mode lap95 --eval-every 6000 --eval-episodes 5 --out runs_grpo_c_ppo_finish_pilot
```

--steps 是新增 GRPO-C 决策预算，不含 PPO 预训练；完整轨迹批量可能使实际步数略超预算。目录已存在时重现请换新的 --out。续训改用 --resume 指向 last.pt，同时删除 --grpo-c-warm-start；保留其他训练设置并提高累计 --steps，源 PPO 文件不再是续训依赖。

scripts/compare_finish.py 可对固定基线与候选在另一组相同赛道做逐赛道配对比较，保存原始回报、覆盖率、完赛结果以及新增/丢失完赛数量。它不参与训练或选模；小样本结果不等于总体成功率。示例：

```bat
python scripts/compare_finish.py --baseline runs_grpo_c_ppo_finish_pilot/grpo-c_s0/initial.pt --candidate runs_grpo_c_ppo_finish_pilot/grpo-c_s0/best.pt --episodes 10 --seed-start 1000100 --device cuda --out runs_grpo_c_ppo_finish_pilot/comparison10
```

若 best.pt 仍为零步初始化，应明确报告“未取得额外微调收益”；不得把继承的 PPO 完赛能力称为 GRPO-C 微调成果。

首轮结果：runs_grpo_c_ppo_finish_pilot 正常完成 13,765 决策，训练约 8.51 分钟、验证约 9.86 分钟、墙钟约 18.38 分钟。固定 5 条验证赛道上，初始化为 40% 完赛率、86.45% 覆盖率；7,968 步选中成员为 20%/89.99%；13,765 步选中成员为 20%/94.44%。因此 best.pt 保持零步初始化，首轮没有取得微调增益。曲线保存在 runs_grpo_c_ppo_finish_pilot/assessment。

第二轮重新从相同 PPO 起点出发，使用更小的学习率 2e-6、每批一次更新，并设置 gamma=1，让训练目标对应不折扣的整回合回报及完赛奖励；这是一组面向目标的调参，不是单因素因果实验。验证扩到 10 条赛道，初始化也重新评估。新增 --grpo-c-parallel-eval 用已有环境进程同时验证四个策略，保持逐策略单观测推理、相同种子、原始奖励及终止规则；结果按成员分别记录，不使用随机探索。

```bat
python -m racing.train --algo grpo-c --seed 0 --steps 6000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-episodes-per-update 2 --grpo-c-epochs 1 --grpo-c-warm-start runs_lap95_lr1e4/ppo_s0/best.pt --grpo-c-selection completion --grpo-c-completion-bonus 200 --grpo-c-beta 0.1 --grpo-c-smooth 0 --grpo-c-diversity 0 --grpo-c-reference-tau 0.01 --grpo-c-parallel-eval --gamma 1 --actor-lr 2e-6 --clip 0.1 --batch-size 128 --completion-mode lap95 --eval-every 6000 --eval-episodes 10 --out runs_grpo_c_ppo_finish_conservative
```

配对比较脚本也支持 --parallel，同一赛道同时运行固定基线与候选；此时两份摘要中的 evaluation_seconds 都标记为共享墙钟时间，不可相加。


第二轮结果：runs_grpo_c_ppo_finish_conservative 完成 7,979 决策，10 条固定赛道起点为 5/10 完赛、92.03% 覆盖率；最终四成员分别为 3/10、3/10、4/10、2/10，按完赛率选中 member 2，仍未超过起点，best.pt 保持初始化。耗时约 10.98 分钟，其中训练约 5.12 分钟、验证约 5.85 分钟；期间同时运行过 CPU 测试，不能用作独占机器速度基准。

第三轮保留视觉特征和动作噪声，只微调最后的动作输出层（每策略 771 个参数）。--grpo-c-trainable head 需要预训练起点或匹配的续训配置；梯度测试确认冻结部分保持逐位不变。每策略每批采四条轨迹，再进行一次 GRPO-C 更新。仍使用独立策略、状态相对优势、剪切目标和参考混合分布，没有使用 critic。

```bat
python -m racing.train --algo grpo-c --seed 0 --steps 6000 --device cuda --num-envs 4 --grpo-c-clusters 2 --grpo-c-episodes-per-update 4 --grpo-c-epochs 1 --grpo-c-warm-start runs_lap95_lr1e4/ppo_s0/best.pt --grpo-c-trainable head --grpo-c-selection completion --grpo-c-completion-bonus 200 --grpo-c-beta 0.1 --grpo-c-smooth 0 --grpo-c-diversity 0 --grpo-c-reference-tau 0.01 --grpo-c-parallel-eval --gamma 1 --actor-lr 1e-5 --clip 0.1 --batch-size 128 --completion-mode lap95 --eval-every 6000 --eval-episodes 10 --out runs_grpo_c_ppo_finish_head
```

第三轮结果：runs_grpo_c_ppo_finish_head 完成 7,975 决策，四成员的 10 赛道完赛次数分别为 2、3、3、4，最终选中 member 3（4/10，覆盖率约 92.1%），仍未超过初始化。对实际检查点逐项核对，四个成员都只改变了 actor.9.weight 与 actor.9.bias，视觉参数及 logstd 与源 PPO 完全相同；结果保存在该目录的 frozen_parameter_check.json。

第四轮缩小到 gas-bias：只优化最终动作层油门均值的一个偏置分量，同一画面下的转向、刹车与动作噪声保持不变。梯度掩码测试已验证多次 Adam 更新后其他参数逐位不变。这是 GRPO-C 对预训练策略的受限微调，不是从零学驾驶。策略比较设为同一组，使成功和失败的轨迹留在同组比较；状态聚类、相对优势与参考混合分布仍保留。这里的 0.01 学习率只用于油门偏置，不能当作全网络训练的建议。

```bat
python -m racing.train --algo grpo-c --seed 0 --steps 6000 --device cuda --num-envs 4 --grpo-c-clusters 1 --grpo-c-episodes-per-update 2 --grpo-c-epochs 1 --grpo-c-warm-start runs_lap95_lr1e4/ppo_s0/best.pt --grpo-c-trainable gas-bias --grpo-c-selection completion --grpo-c-completion-bonus 200 --grpo-c-beta 0.1 --grpo-c-smooth 0 --grpo-c-diversity 0 --grpo-c-reference-tau 0.01 --grpo-c-parallel-eval --gamma 1 --actor-lr 0.01 --clip 0.1 --batch-size 128 --completion-mode lap95 --eval-every 6000 --eval-episodes 10 --out runs_grpo_c_ppo_finish_gas
```


第四轮最终四成员完赛次数为 2、3、2、3 /10，未超过 5/10 初始化。第三轮 head 的独立 10 赛道比较为基线和候选均 2/10，但覆盖率从 89.02% 降为 79.82%；验证集上的输出层回溯 alpha=0、0.1、0.25、0.5、1 分别为 5、3、2、3、4 /10，选择 alpha=0 即原始 PPO，不能算作 GRPO-C 改善。

另用 runs_tuning_lr1e4_1000k/ppo_s0/best.pt 作为原 original 协议教师，显式转到 lap95。runs_grpo_c_ppo_original_gas_oneupdate 完成 1,992 决策后，验证 5 赛道从 1/5 到 2/5；但固定候选在独立 20 赛道（1000200–1000219）上从基线 8/20 降到 4/20，新增完赛 1 条、丢失 5 条。因此不采用这次验证集提升作为有效结论，不能向用户宣称完赛率已提高。

## 12. 图像 + 车辆/道路状态实验（2026-10-16）

用户明确接受新增状态输入，以提高真实完赛率为目标。此协议单独标注为 pixel_road_state_global_loo_mixture_v3，不与纯图像 GRPO-C 视作相同条件。PPO 仍提供预训练能力，GRPO-C 只承担后续微调。

保留原四帧图像，追加元数据平面编码 16 个特征：前向/侧向车速、角速度、横向偏移、道路切向相对车体的两个分量、前方第 3/8/16 个路点在车体坐标的横向/前向位置、覆盖率、已用帧比例、前轮转角、车轮在路面比例。特征使用模拟器车辆和完整赛道几何，属于额外信息；归一化截断到 [-1,1]，以 uint8 元数据编码，分辨率 1/127。原图像不被覆盖。

策略均值为原 PPO 图像网络输出 + 0.25*tanh(新增线性状态分支)。新增分支共 51 个参数，初始权重和偏置均为零，初始化动作与原 PPO 完全一致；road-head 模式冻结原图像网络和动作噪声。没有手写转向控制器。状态相对优势的聚类改用这 16 个状态特征；多策略相对比较、剪切目标、参考混合分布仍保留，不使用 critic。

这一轮训练目标为原始奖励 + 真正完赛奖励 200 - 每原始帧 0.1*离开道路的车轮比例。草坪惩罚只用于训练，不修改环境原始奖励或评估得分。完赛仍需严格超过 95% 且正向重入起点，或访问所有路块；1000 原始帧时限不变。

启动命令（使用项目 .venv 的 Python）：

```bat
python -m racing.train --algo grpo-c --seed 0 --steps 12000 --device cuda --num-envs 4 --grpo-c-clusters 1 --grpo-c-episodes-per-update 4 --grpo-c-epochs 1 --grpo-c-warm-start runs_lap95_lr1e4/ppo_s0/best.pt --grpo-c-road-state --grpo-c-road-scale 0.25 --grpo-c-trainable road-head --grpo-c-offroad-penalty 0.1 --grpo-c-selection completion --grpo-c-completion-bonus 200 --grpo-c-beta 0.1 --grpo-c-smooth 0 --grpo-c-diversity 0 --grpo-c-reference-tau 0.01 --grpo-c-parallel-eval --gamma 1 --actor-lr 0.003 --clip 0.1 --batch-size 128 --completion-mode lap95 --eval-every 12000 --eval-episodes 10 --out runs_grpo_c_road_state_pilot
```

验证覆盖环境观测扩展的原始像素/奖励/终止一致性、1000 帧限制、零修正动作精确一致、状态迁移显式授权和物理参数防错、仅新增状态分支更新。初次相关测试 20 项通过。只有独立赛道上固定候选的配对评估才用于判断泛化改善，训练奖励和验证集选模均不能代替这项检查。


首轮状态输入实验完成 15,819 决策（完整批次导致比请求预算多 3,819），63,244 原始帧；训练 431.02 秒，验证 477.51 秒，总墙钟 909.02 秒。10 条固定验证赛道起点 5/10 完赛、92.0281% 覆盖率、822.45 原始回报；最终成员分别 3/10、4/10、4/10、5/10，选择 member 3，覆盖率 93.9250%、原始回报 840.89。best.pt 是本轮训练后的 member 3（完赛率持平，覆盖率作为第二排序项胜出），并非完赛率已提高。实际权重核查证明仅 road_head.weight/bias 变化，结果保存到 frozen_parameter_check.json。完整测试 62 项通过。

固定 best.pt 在此前未使用的 1000300–1000319 共 20 条赛道上，与同目录 initial.pt 配对比较。两者均使用新增状态协议，initial.pt 的零状态分支与原 PPO 行为一致；不在这 20 条赛道上重新挑成员或调参数。结果保存到 comparison20/comparison.json。


独立 20 赛道最终结果：基线 9/20（45%）、候选 9/20（45%），新增完赛 3 条、丢失 3 条、共同完赛 6 条、共同未完赛 8 条。平均覆盖率 87.8594% → 89.0033%，原始回报 781.48 → 793.18。结论是覆盖率小幅改善但完赛率没有提高，不能把这一轮称为目标达成，也不能仅凭此小样本断言二者总体能力相同。只有 4 次 GRPO-C 更新、1 个训练种子，尚不能判断此表示已经收敛或无效。

本轮训练和评估均已结束，没有后台训练进程。后续应以训练/验证赛道上的多次更新和更多训练种子检查稳定性；本次 20 条对照赛道已经用于报告，后续调参不得把它们作为选模依据，最终结论应使用另一组未见赛道。源 PPO 和所有候选检查点均保留。简要结果见 runs_grpo_c_road_state_pilot/assessment/result.json。
