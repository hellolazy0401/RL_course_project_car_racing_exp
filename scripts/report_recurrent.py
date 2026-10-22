"""Produce a standalone Chinese HTML experiment record and auditable CSV/plots."""
import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import shutil
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    args = p.parse_args()
    root = Path(args.root).resolve()
    results = read(root / "results.json")
    selection = read(root / "selection.json")
    delivery = root / "delivery"
    delivery.mkdir(exist_ok=True)
    for record in ("protocol.json", "实验决策记录.txt", "verification.json"):
        if (root / record).exists():
            shutil.copy2(root / record, delivery / record)
    if (root / "diagnostics" / "validation_failure").exists():
        shutil.copytree(root / "diagnostics" / "validation_failure", delivery / "diagnostics",
                        dirs_exist_ok=True)
    extra_links = " · ".join(f'<a href="{name}">{label}</a>' for name, label in (
        ("protocol.json", "完整实验协议"), ("实验决策记录.txt", "实验决策记录"),
        ("verification.json", "代码验证记录")) if (delivery / name).exists())
    diagnostic_html = ("""<details><summary>确认验证中的漏过起点案例（保留失败分析）</summary>
<p>原始实验赛道 1100010 覆盖率约 99%，但未触碰起点感应路块；进入第二圈后超时。</p>
<img src="diagnostics/telemetry.png"><a href="diagnostics/telemetry.csv">逐决策遥测</a> ·
<a href="diagnostics/gate_trace.csv">逐物理帧起点接触记录</a> ·
<a href="diagnostics/gate_trace_summary.json">起点接触核验</a></details>"""
                       if (delivery / "diagnostics" / "telemetry.png").exists() else "")
    chosen = selection["winner"]["name"]
    selected = next(r for r in results if r["name"] == chosen)
    alternate_html = ""
    alternate = next((r for r in results if r["name"] == "state_direct_finish_s0"), None)
    if alternate is not None and alternate["name"] != chosen:
        checkpoint = Path(alternate["checkpoint"])
        if not checkpoint.exists():
            # Rebuild a report from portable records with remapped paths.
            checkpoint = root / "training" / alternate["name"] / checkpoint.name
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != alternate["checkpoint_sha256"]:
            raise ValueError("Finish-assist candidate changed after evaluation")
        shutil.copy2(checkpoint, delivery / "finish_assist_candidate.pt")
        confirmed = next(r for r in selection["candidates"] if r["name"] == alternate["name"])
        (delivery / "finish_assist_candidate_card.json").write_text(json.dumps(dict(
            role="Secondary candidate; not selected by pre-test validation rule",
            source=str(checkpoint), sha256=digest, additional_training_decisions=0,
            confirmation_completion=confirmed["completion_rate"],
            test_completion=alternate["completion_rate"],
            test_mean_finish_seconds=alternate["mean_finish_seconds"],
            note="Retained without replacing the frozen primary model; finite validation and test sets can rank candidates differently.",
            usage="python -m racing.evaluate --checkpoint delivery/finish_assist_candidate.pt --episodes 100 --seed-start 5000000 --out finish_assist_new_evaluation"
        ), ensure_ascii=False, indent=2), encoding="utf-8")
        alternate_html = (
            f'<p><b>另附终点辅助候选：</b>测试完赛率 {alternate["completion_rate"]:.1%}，'
            f'成功平均 {alternate["mean_finish_seconds"]:.2f} 秒；确认验证完赛率为 {confirmed["completion_rate"]:.1%}。'
            '确认验证与最终测试的排序可能不同，因此保留测试前选定的主模型，并提供该候选供独立复核。'
            '<a href="finish_assist_candidate.pt">辅助候选模型</a> · '
            '<a href="finish_assist_candidate_card.json">候选说明</a>。</p>')
    (delivery / "开始使用.txt").write_text(
        "Recurrent PPO 赛车实验交付包\n\n"
        "1. 打开 delivery/实验记录.html 查看实验、比较、限制与录像。\n"
        "2. delivery/best_model.pt 是测试前按确认验证选定的主模型。\n"
        "   finish_assist_candidate.pt 是保留的终点辅助候选；详见候选说明。\n"
        "3. delivery/code 是完整代码；records 是配置、日志、逐回合数据和检查点。\n"
        "   阶段模型位置与指标见 delivery/milestones.csv；步数 0 表示初始化。\n\n"
        "在本机已有的 rl_carracing 环境中，解压后从解压根目录执行：\n"
        "  conda activate rl_carracing\n"
        "  cd delivery\\code\n"
        "  python -m racing.evaluate --checkpoint ../best_model.pt --episodes 100 "
        "--seed-start 5000000 --videos 3 --device cpu --out ../../new_evaluation\n"
        "辅助候选：把 --checkpoint 改为 ../finish_assist_candidate.pt，使用另一个 --out。\n"
        "完整重训（仍在 delivery/code 目录，需等待全部训练与评估）：\n"
        "  python -m racing.reproduce_recurrent --records ../../records "
        "--out ../../reproduced_experiment --finalize\n\n"
        "输出目录必须尚不存在。详细参数和续训方法见 code/docs/Recurrent_PPO运行说明.txt。\n"
        "本次高完赛率模型使用仿真器状态，推理时必须逐步传递 LSTM 记忆，"
        "新赛道清零；racing.evaluate 已实现。模型只从可信来源加载。\n",
        encoding="utf-8")
    if (root / "test" / "previous_sac_best" / "summary.json").exists():
        results.append(dict(name="previous_sac_best", **read(root / "test" / "previous_sac_best" / "summary.json")))
    records = []
    for result in results:
        records.append(dict(name=result["name"], episodes=result["episodes"],
            completed=round(result["completion_rate"] * result["episodes"]),
            completion_rate=result["completion_rate"], mean_finish_seconds=result["mean_finish_seconds"],
            penalized_seconds=result["penalized_seconds"], mean_return=result["mean_return"],
            mean_coverage=result["mean_coverage"], selected=result["name"] == chosen))
    with (delivery / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    baseline_rows = rows(root / "test" / "controller_only" / "episodes.csv")
    selected_rows = rows(root / "test" / chosen / "episodes.csv")
    failed_rows = [r for r in selected_rows if r["completed"] != "True"]
    failure_text = "；".join(
        f'{r["track_seed"]}（覆盖率 {float(r["coverage"]):.2%}，'
        f'{"超时" if r["truncated"] == "True" else "终止失败"}）'
        for r in failed_rows) or "本批测试无失败"
    second_seed = next((r for r in results if r["name"] == "state_direct_s1"), None)
    stability_text = ("另一直接控制训练种子在本批测试中没有完赛。"
                      if second_seed is not None and second_seed["completion_rate"] == 0 else "")
    base_by_seed = {r["track_seed"]: r for r in baseline_rows}
    pairs = [(r, base_by_seed[r["track_seed"]]) for r in selected_rows
             if r["completed"] == "True" and base_by_seed[r["track_seed"]]["completed"] == "True"]
    deltas = [(float(a["raw_frames"]) - float(b["raw_frames"])) / 50 for a, b in pairs]
    paired = dict(common_success_tracks=len(pairs),
                  mean_model_minus_controller_seconds=float(np.mean(deltas)) if deltas else None,
                  model_only_success=sum(a["completed"] == "True" and base_by_seed[a["track_seed"]]["completed"] != "True" for a in selected_rows),
                  controller_only_success=sum(a["completed"] != "True" and base_by_seed[a["track_seed"]]["completed"] == "True" for a in selected_rows))
    (delivery / "paired_comparison.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [r["name"] for r in records]
    colors = ["#16a085" if r["selected"] else "#5375a7" for r in records]
    axes[0].barh(labels, [r["completion_rate"]*100 for r in records], color=colors)
    axes[0].set_xlim(0, 105); axes[0].set_xlabel("Completion rate (%)")
    axes[1].barh(labels, [r["mean_finish_seconds"] or 0 for r in records], color=colors)
    axes[1].set_xlabel("Mean seconds on successful tracks only")
    axes[0].set_title("Fresh held-out tracks"); axes[1].set_title("Zero means no completed lap")
    fig.tight_layout(); fig.savefig(delivery / "comparison.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    training = []
    for run in sorted((root / "training").glob("*")):
        if not (run / "config.json").exists():
            continue
        segments = [run] + sorted(run.glob("segment_*"))
        curve = {}
        for segment in segments:
            if (segment / "validation.csv").exists():
                curve.update({int(r["steps"]): r for r in rows(segment / "validation.csv")})
        if not curve:
            continue
        ordered = [curve[s] for s in sorted(curve)]
        axes[0].plot([int(r["steps"]) for r in ordered], [float(r["completion_rate"])*100 for r in ordered], label=run.name)
        axes[1].plot([int(r["steps"]) for r in ordered], [float(r["mean_coverage"])*100 for r in ordered], label=run.name)
        from racing.recurrent_ppo import selection_key
        best = max([read(segment / "best_metrics.json") for segment in segments
                    if (segment / "best_metrics.json").exists()], key=selection_key)
        finishes = [read(segment / "finished.json") for segment in segments if (segment / "finished.json").exists()]
        finished = max(finishes, key=lambda item: item["steps"]) if finishes else {}
        training.append(dict(name=run.name, decisions=finished.get("steps"),
                             training_seconds=finished.get("train_seconds"), best_validation=best))
    for ax, title in zip(axes, ("Validation completion", "Validation coverage")):
        ax.set_title(title); ax.set_xlabel("Training decisions"); ax.set_ylabel("%"); ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(delivery / "validation.png", dpi=180); plt.close(fig)
    (delivery / "training_summary.json").write_text(json.dumps(training, ensure_ascii=False, indent=2), encoding="utf-8")
    table = ""
    for r in records:
        finish = f'{r["mean_finish_seconds"]:.2f} 秒' if r["mean_finish_seconds"] is not None else "无完赛"
        table += (f'<tr><td>{html.escape(r["name"])}{" ★" if r["selected"] else ""}</td>'
                  f'<td>{r["completed"]}/{r["episodes"]}</td><td>{r["completion_rate"]:.1%}</td>'
                  f'<td>{finish}</td><td>{r["mean_coverage"]:.1%}</td><td>{r["mean_return"]:.1f}</td></tr>')
    config = selection["winner"]["config"]
    ci = selected["completion_wilson95"]
    milestone_files = sorted((root / "training").rglob("milestone_*.pt"))
    milestones = "".join(f'<li>{p.relative_to(root / "training").as_posix()}</li>' for p in milestone_files)
    milestone_records = []
    for checkpoint in milestone_files:
        step = int(checkpoint.stem.split("_")[-1])
        validations = rows(checkpoint.parent / "validation.csv")
        metrics = next(r for r in validations if int(r["steps"]) == step)
        milestone_records.append(dict(
            archive_path="records/" + checkpoint.relative_to(root).as_posix(),
            decisions=step, completion_rate=metrics["completion_rate"],
            mean_finish_seconds=metrics["mean_finish_seconds"],
            validation_episodes=metrics["episodes"],
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    if milestone_records:
        with (delivery / "milestones.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(milestone_records[0]))
            writer.writeheader(); writer.writerows(milestone_records)
    training_table = "".join(
        f'<tr><td>{r["name"]}</td><td>{r["decisions"]}</td>'
        f'<td>{r["best_validation"]["completion_rate"]:.1%}</td>'
        f'<td>{r["best_validation"]["steps"]}</td></tr>' for r in training)
    confirmation_table = "".join(
        f'<tr><td>{r["name"]}</td><td>{r["checkpoint_steps"]}</td><td>{r["completion_rate"]:.1%}</td>'
        f'<td>{r["mean_finish_seconds"] if r["mean_finish_seconds"] is not None else "无完赛"}</td></tr>'
        for r in selection["candidates"])
    video_dir = root / "test" / chosen / "videos"
    videos = ""
    if video_dir.exists():
        shutil.copytree(video_dir, delivery / "videos", dirs_exist_ok=True)
        videos = "".join(f'<video controls preload="metadata" width="420" src="videos/{p.name}"></video>'
                         for p in sorted(video_dir.glob("*.mp4")))
    text = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Recurrent PPO 赛车实验记录</title><style>
body{{font:16px/1.7 system-ui,sans-serif;color:#203040;max-width:1120px;margin:36px auto;padding:0 24px;background:#fafcff}}
h1,h2{{line-height:1.3}} .card{{padding:20px;background:#eaf5f2;border-radius:12px}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #dce4ec}}
img{{width:100%;margin-top:20px}}code,pre{{background:#edf1f6}}pre{{padding:16px;overflow:auto}}a{{color:#1c6994}}
</style><h1>Recurrent PPO 赛车实验记录</h1>
<p>报告生成日期：{datetime.now().astimezone().strftime("%Y-%m-%d")}。目标：先提高完赛率，再缩短完赛时间。★ 为测试前已选定的交付模型。</p>
<div class="card"><b>交付模型：{chosen}</b>，训练检查点 {selected["checkpoint_steps"]:,} 决策。
在 {selected["episodes"]} 条未参与调参的测试赛道上，完赛率 <b>{selected["completion_rate"]:.1%}</b>，
成功完赛平均 <b>{selected["mean_finish_seconds"]:.2f} 秒</b>。
完赛率 Wilson 95% 区间：{ci[0]:.1%}–{ci[1]:.1%}。</div>{alternate_html}
<h2>实验设计与边界</h2><p>Box2D 赛车物理保持原样，lap95 完赛门槛保持原样，最多 1,000 帧（20 秒），每决策重复 4 帧。
只有真正终止且 lap_finished 为真才计为完赛，不把 95% 覆盖率、超时或出界计为完赛。
状态输入分支读取赛道中心线、车辆状态和由几何计算的控制建议，属于有额外信息的工程方案；不能把它相对纯像素 SAC 的改善全部归因于 Recurrent PPO。
direct 分支由网络直接给出转向与有符号加减速，油门和刹车互斥；residual 分支叠加预瞄控制器；
adaptive 分支学习转向修正和 ±25% 的局部目标速度调整。</p>
<p>训练赛道随机来自 0–999999；训练中验证 1000000 起的 10 或 30 条固定赛道；
候选模型在 1100000–1100099 的 100 条确认验证赛道复核。选择规则为完赛率优先，其次成功完赛平均时间，再看覆盖率、原始回报。
选择结果先写入 selection.json，再运行 4000000–4000099 的最终测试。测试后没有据测试成绩重新选择模型。</p>
<p>这是有限预算的探索实验，候选的训练预算、验证集大小与种子数见原始配置；不宣称严格等预算算法排名。
成功回合平均时间有条件选择效应，另附共同成功赛道的配对时间和失败按 40 秒计的惩罚用时。</p>
<p>像素试验在保存 65,536 决策检查点后有计划地恢复，以启用经过等价性验证的绘图优化。
至少另有 1,024 个已写入更新日志的决策发生在保存点之后、未进入恢复模型；中断时未完成的 rollout 无法精确计数。
intentional_restart.json 保留了额外采样记录。
状态直接控制 seed 0 也在 98,304 决策后续训。两次恢复均重新开始环境回合。</p>
<h2>最终测试</h2><table><tr><th>模型</th><th>完赛数</th><th>完赛率</th><th>成功平均用时</th><th>覆盖率</th><th>原始回报</th></tr>{table}</table>
<img src="comparison.png"><p>controller_only 为验证选定的 v2 控制器（倍率 2.2），controller_v1 为初始控制器（倍率 2.2），均无学习。
模型与 v2 无学习控制器共同成功 {paired["common_success_tracks"]} 条赛道；
平均用时差（模型减控制器）：{paired["mean_model_minus_controller_seconds"]} 秒。
仅模型成功 {paired["model_only_success"]} 条，仅控制器成功 {paired["controller_only_success"]} 条。</p>
<p>预选主模型失败赛道：{failure_text}。高覆盖率本身不算完赛，保留这些失败记录。
{stability_text}当前交付证明的是这份已选权重对新赛道的表现，尚未证明训练过程对随机种子稳定。</p>
<h2>训练过程</h2><table><tr><th>实验</th><th>累计决策</th><th>训练中最佳完赛率</th><th>最佳步数</th></tr>{training_table}</table>
<img src="validation.png"><p>原始 2026-10-20 实验的调参过程：初步状态直接控制实验未能在 98,304 决策内完赛。
预瞄控制器速度筛选依次尝试 0.8–2.6 的倍率；过慢会超时，过快会失控，依据验证结果固定 2.2。
转向网格进一步比较 12 组增益、阻尼和预瞄参数，选出 v2；v2 的速度倍率再次验证后仍采用 2.2。
之后使用 Recurrent PPO 学习控制修正、局部目标速度，尝试独立转向/纵向输出头、不同隐藏层宽度及随机种子，并延长直接控制训练。
所有失败试验、初始基线和阶段性检查点都保留在原始实验目录。</p>
<h2>100 条确认验证赛道上的选择依据</h2>
<table><tr><th>候选</th><th>检查点步数</th><th>完赛率</th><th>成功平均用时（秒）</th></tr>{confirmation_table}</table>
<p>state_direct_finish_s0 复用 state_direct_s0 的同一份学习权重，只增加终点附近的物理驾驶辅助，没有额外训练。
辅助只改变控制命令，仍须实际经过起点触发完赛；它在确认验证集上的表现决定是否采用。
未能超过初始化的训练运行，在确认和最终测试中使用训练后的 last.pt，并保留初始化 best.pt 作为失败实验的证据。
不同训练种子的收敛差异必须与已选模型对新赛道的泛化性能分开理解。</p>
<h2>算法实现</h2><p>编码器 → LSTM → 策略与价值头；连续 32 步序列做截断反向传播，按序列打乱，不打散序列内部时间顺序。
LSTM 状态跨决策和 rollout 保留，回合开始逐环境清零。超时使用最终观察及其对应隐藏状态自举；
真正终止不自举，两类边界都截断 GAE。使用 PPO-Clip、优势标准化、梯度裁剪与目标 KL 提前停止。
模型保留优化器、随机数状态和累计计数；环境物理状态不序列化，续训会启动新回合。</p>
<p>实现参考：<a href="https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html">SB3-Contrib Recurrent PPO 的状态管理约定</a>；
<a href="https://gymnasium.farama.org/api/env/">Gymnasium 终止/截断语义</a>。本项目使用自行实现的 PyTorch 训练器，无新增第三方依赖。</p>
<h2>复现与交付</h2><p><a href="开始使用.txt">解压后使用说明</a> ·
<a href="best_model.pt">最佳模型</a> · <a href="model_card.json">模型配置</a> ·
<a href="comparison.csv">对照表</a> · <a href="paired_comparison.json">配对分析</a> · <a href="training_summary.json">训练摘要</a> · {extra_links}</p>
<pre>conda activate rl_carracing
cd /d D:\\Desktop\\RL_project
python -m racing.evaluate --checkpoint runs_recurrent_ppo_20261020/delivery/best_model.pt --device cpu --episodes 100 --seed-start 5000000 --videos 3 --out runs_recurrent_ppo_new_test
python -m pytest -q</pre>
<p>完整复现实验矩阵：<code>python -m racing.reproduce_recurrent --records 解压后的records目录 --out 新实验目录 --finalize</code>。
运行说明位于 code/docs/Recurrent_PPO运行说明.txt。</p>
<p>原始实验目录内包含完整代码、配置、版本/源码哈希、逐回合记录、逐更新记录、选择依据、视频和阶段性权重。
模型 SHA256：<code>{selected["checkpoint_sha256"]}</code>。</p>
<details><summary>交付模型参数</summary><pre>{html.escape(json.dumps(config, ensure_ascii=False, indent=2))}</pre></details>
<details><summary>阶段性模型（{len(milestone_files)} 个）</summary>
<p><a href="milestones.csv">阶段模型指标与 SHA256 索引</a>，步数 0 为初始化基线。</p><ul>{milestones}</ul></details>
{diagnostic_html}
<h2>固定测试赛道视频</h2>{videos}</html>"""
    (delivery / "实验记录.html").write_text(text, encoding="utf-8")
    # Portable archive: code, selected model, all checkpoints and numerical
    # records, plots and winner videos, including failed experiments.
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source / "racing", delivery / "code" / "racing",
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    shutil.copytree(source / "tests", delivery / "code" / "tests",
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    shutil.copytree(source / "scripts", delivery / "code" / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__"), dirs_exist_ok=True)
    for name in ("requirements.txt", "requirements-resolved-local.txt", "README.md", "pytest.ini"):
        shutil.copy2(source / name, delivery / "code" / name)
    shutil.copytree(source / "docs", delivery / "code" / "docs", dirs_exist_ok=True)
    manifest = {str(p.relative_to(delivery)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in delivery.rglob("*") if p.is_file() and p.name != "manifest.json"}
    (delivery / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = root / "Recurrent_PPO_delivery.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for f in delivery.rglob("*"):
            if f.is_file():
                z.write(f, "delivery/" + f.relative_to(delivery).as_posix())
        for f in root.rglob("*"):
            if f.is_file() and delivery not in f.parents and "__pycache__" not in f.parts:
                is_source = f.relative_to(root).parts[0].startswith("source")
                if is_source or f.suffix in (".json", ".jsonl", ".csv", ".log", ".txt", ".png", ".pt"):
                    z.write(f, "records/" + f.relative_to(root).as_posix())
    print(json.dumps(dict(report=str(delivery / "实验记录.html"), archive=str(archive),
                         selected=chosen, completion=selected["completion_rate"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
