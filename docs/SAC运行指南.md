# SAC — Soft Actor-Critic 实验步骤

新增 `--algo sac`，复用项目训练、评估、视频及报告流程，不增加依赖。算法源码为 `racing/sac.py`。以下命令在项目根目录和已激活的 `rl_carracing` 环境执行；使用项目虚拟环境时，将 `python` 替换成 `.\.venv\Scripts\python.exe`。

## 1. 算法原理与实现约定

SAC 是最大熵框架下的离策略 Actor-Critic。本工程采用 SAC-v2：状态相关均值/标准差的高斯策略、双 Q、双目标 Q、经验回放、重参数化梯度和自动熵温度 α；没有独立 V 网络。

```text
a' ~ π(·|s')
y = reward_scale × r + γ(1-terminated)[min(Q̄1(s',a'),Q̄2(s',a')) - α logπ(a'|s')]
LQ = mean[(Q1(s,a)-y)² + (Q2(s,a)-y)²]
Lπ = mean[α logπ(a|s) - min(Q1(s,a),Q2(s,a))]
Llogα = -mean[logα × stop_gradient(logπ(a|s) + target_entropy)]
目标参数 ← (1-τ)目标参数 + τ在线参数
```

策略经 tanh 映射到归一化动作，再映射为转向 `[-1,1]`、油门/刹车 `[0,1]`。熵包含 tanh 雅可比修正，定义在归一化动作空间，目标默认 `-3`；物理动作缩放常数不混入该熵目标。评估使用 tanh 后的策略均值，不加探索噪声。

输入为 4 帧 64×64 灰度图，动作重复 4，奖励和赛道协议沿用项目设置；未增加道路状态或奖励塑形。actor 和两个 Q 各有独立 CNN。时间上限截断继续 bootstrap，真正终止才切断价值；经验池存最终画面，而非自动重置画面。

参考：[SAC 原论文](https://arxiv.org/abs/1801.01290)、[SAC Algorithms and Applications（自动温度）](https://arxiv.org/abs/1812.05905)、[Spinning Up SAC 公式](https://spinningup.openai.com/en/latest/algorithms/sac.html)。本工程是像素环境实现，不是论文成绩复现。

## 2. 环境检查

```powershell
cd D:\Desktop\RL_project
conda activate rl_carracing
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest tests/test_sac.py -q
```

无 CUDA 时把后续 `--device cuda` 改为 `--device cpu`。正式训练前关闭其他 GPU 训练任务。

## 3. 完整流程自检

```powershell
python -m racing.suite --preset smoke --algos sac --device cuda --completion-mode lap95 --out runs_sac_smoke_lap95
```

自检训练 512 决策、2 环境，专门设置预热 128 决策、经验池 1024 条，确保发生真实梯度更新。之后验证、保存、重载 last/best、测试、录视频及出报告。查看 `runs_sac_smoke_lap95\smoke\report\index.html`，确认 `updates.csv` 的 `total_updates > 0`、更新损失有限、α 为正。预热阶段无更新的损失为 NaN，表示不适用。

pilot/main 使用正常参数；自检成绩不能作为算法排名或收敛结论。同名输出目录拒绝覆盖，重跑请换 `--out`。

## 4. 测速与单种子试跑

SAC 默认随机探索 5000 决策，测速必须超过该阶段：

```powershell
python -m racing.train --algo sac --steps 10000 --num-envs 4 --batch-size 128 --eval-every 10000 --eval-episodes 1 --completion-mode lap95 --device cuda --out runs_sac_benchmark
python -m racing.benchmark --root runs_sac_benchmark --steps 1000000 --seeds 3
python -m racing.suite --preset pilot --algos sac --completion-mode lap95 --device cuda --out runs_sac_pilot_lap95
```

pilot 是 1 个种子、10 万决策。只根据验证集判断学习趋势和选择超参数，自动生成的 pilot 测试数据仅检查流程。测速平均值包含无更新预热，会高估持续训练吞吐；另看预热结束后的 `decision_sps` 并留时间余量。

## 5. 正式三种子实验与算法对照

先固定完赛规则。下面采用 `lap95`；如沿用原生规则，所有方法统一改为 `original`，另设输出目录。

仅运行 SAC：

```powershell
python -m racing.suite --preset main --algos sac --seeds 0 1 2 --steps 1000000 --completion-mode lap95 --device cuda --out runs_sac_main_lap95
```

或从头运行四算法对照：

```powershell
python -m racing.suite --preset main --algos a2c ppo vpg50 sac --seeds 0 1 2 --steps 1000000 --completion-mode lap95 --device cuda --out runs_four_main_lap95
```

两种方案按需选一，不必重复跑。suite 默认仍为原来的三个算法，必须显式加入 `sac`。各任务顺序执行，避免 GPU 竞争。

每方法/种子训练 100 万决策，每 5 万决策在同样 10 条验证赛道评估；last/best 各测试 50 条赛道并录前三条视频。训练赛道种子小于 1,000,000，验证从 1,000,000 开始，测试从 2,000,000 开始。

主表用 `last.pt` 比较等交互预算；辅表用 `best.pt` 比较验证选模结果。另报告原始帧数、优化次数、耗时、显存及参数量：等样本不等于等计算成本。100 万决策不保证收敛，增加预算时应对所有比较算法统一增加。不同完赛规则、输入协议或奖励塑形的旧实验不能混入同一主表。

## 6. 参数和资源

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--sac-buffer-size` | 50000 | CPU uint8 经验池，按转移计数 |
| `--sac-learning-starts` | 5000 | 全部环境合计的随机探索决策数 |
| `--sac-gradient-steps` | 1 | 每个向量环境步后的更新次数 |
| `--num-envs` | 4 | 每个向量步增加的决策数 |
| `--batch-size` | 128 | 有放回采样的 minibatch 大小 |
| `--sac-tau` | 0.005 | 目标 Q 软更新比例 |
| `--sac-alpha` | 0.2 | 初始温度，关闭自动温度时为固定值 |
| `--sac-alpha-lr` | 0.0003 | log α 学习率 |
| `--sac-target-entropy` | -3 | 归一化三维动作目标熵 |
| `--actor-lr` / `--critic-lr` | 0.0003 | 策略 / 双 Q 学习率 |
| `--gamma` / `--reward-scale` | 0.99 / 0.1 | 折扣 / 仅训练奖励缩放 |

默认更新/新转移比为 `1/4=0.25`；设 `--sac-gradient-steps 4` 后为 1，计算成本也增加。更改环境数时必须说明这一比例。`--rollout` 仅控制日志块长度；SAC 不等整个 rollout 结束才更新。PPO 的 epoch/clip、GAE、value-steps、entropy-coef 和梯度裁剪参数不参与 SAC 更新。

两份图像栈需要约 `50000×2×4×64×64` 字节，即 **1.53 GiB** RAM，另有模型和进程开销。经验池不放显存。填满后的 `last.pt` 也超过 1.53 GiB；原子替换时新旧文件共存，恢复时加载态与新池短暂共存。请留数 GiB 空闲 RAM 和磁盘。内存不足可单独用 `--sac-buffer-size 20000` 运行，并记录参数改变。

## 7. 评估、报告和续训

```powershell
python -m racing.evaluate --checkpoint runs_sac_main_lap95\main\sac_s0\last.pt --episodes 50 --videos 3 --device cuda --out runs_sac_main_lap95\main\sac_s0\test_recheck
python -m racing.report --root runs_sac_main_lap95\main
python -m racing.report --root runs_sac_main_lap95\main --selection test_best
python -m tensorboard.main --logdir runs_sac_main_lap95\main
```

suite 已生成默认 test/test_best，重复评估须另存。`test_recheck` 不替换主表数据。报告新增 `sac_diagnostics.png`（Q 损失、α、熵）；Q 的 Bellman 损失与其他方法的 GAE `value_mse` 含义不同。

续训示例，参数对应上面的 main：

```powershell
python -m racing.train --algo sac --seed 0 --steps 2000000 --num-envs 4 --batch-size 128 --completion-mode lap95 --device cuda --resume runs_sac_main_lap95\main\sac_s0\last.pt --out runs_sac_resumed_lap95
```

`--steps` 是累计目标，须大于已完成数且可被环境数整除。若原实验改了学习率或 SAC 参数，续训需带回相同值。恢复 actor、双 Q、目标 Q、温度、全部优化器、经验池和随机数状态。Box2D 物理状态不恢复，从新回合开始，不能承诺逐位连续。

`interrupted.pt` 也支持恢复；`best.pt` 为节省空间仅保存评估所需状态，不支持续训。续训用新目录，其曲线不当作从零训练曲线。

## 8. 可选消融和报告填写

可额外比较自动温度与固定 α=0.2（`--no-sac-auto-alpha --sac-alpha 0.2`），或更新/新转移比 0.25 与 1。其他配置、预算和种子保持一致，各配置用独立目录，不得合并成同一方法的不同随机种子。

报告填写训练种子数、平均原始回报及种子间标准差、完赛率、覆盖率、≥900 比例、验证曲线、训练时间、更新次数和失败案例。正式结果产生之前不预填成绩或优劣结论。

## 9. 本次实现验证（2026-10-17）

- 全部测试：75 passed（其中新增 SAC 测试 13 项）。
- 本机 PyTorch 2.5.1+cu121，CUDA 可用。
- `runs_sac_selfcheck`：原生 `original` 完赛规则，512 决策、2048 原始帧、193 次梯度更新；last/best 均完成重新加载、单赛道评估及 MP4 生成，HTML 和 SAC 诊断图已生成。
- `runs_sac_resume_selfcheck`：从上述 last.pt 恢复至累计 516 决策、195 次更新，成功保存新检查点。
- 尚未执行 SAC 的 10 万决策 pilot 或百万决策多种子正式训练；本次自检不提供算法性能结论。
