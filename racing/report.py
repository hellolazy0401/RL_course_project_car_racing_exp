"""Seed-level inference, transparent raw episode exports, and portable HTML."""
import argparse
import html
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bootstrap(values, rng):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (None, None)
    draws = rng.choice(values, (10_000, len(values)), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.quantile(draws, [.025,.975]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--selection", choices=["test","test_best","test_ood"], default="test")
    p.add_argument("--allow-unequal-budgets", action="store_true", help="Explicitly report unequal actual budgets (e.g. complete GRPO groups)")
    args = p.parse_args()
    root = Path(args.root).resolve()
    out = root / ("report" if args.selection=="test" else "report_"+args.selection)
    out.mkdir(exist_ok=True)
    runs, episodes, curves, updates, videos = [], [], [], [], []
    configurations = {}
    for path in sorted(root.glob(f"*/{args.selection}/summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        run = path.parent.parent
        row = {k:v for k,v in s.items() if k not in ("config",)}
        row["run"] = run.name
        row["completion_mode"] = s.get("completion_mode", s.get("env_config", {}).get("completion_mode", "original"))
        row["observation_protocol"] = "image+road-state" if s.get("env_config", {}).get("road_state", False) else "image"
        if s.get("env_config", {}).get("observation_mode") == "state":
            row["observation_protocol"] = "privileged-state"
        row["checkpoint_train_seconds"] = row["train_seconds"]
        finished_path = run / "finished.json"
        finished = json.loads(finished_path.read_text(encoding="utf-8")) if finished_path.exists() else {}
        # Selecting an early best checkpoint still costs the entire training run.
        row["train_seconds"] = finished.get("train_seconds", row["train_seconds"])
        # Group different ablations independently instead of averaging them together.
        row["method"] = s["algo"] + (f"_K{s['config']['value_steps']}" if s["algo"]=="vpg50" else "")
        if s["algo"] == "sac":
            row["method"] += "_" + s["config"].get("sac_arch", "baseline")
        if s["algo"] == "recurrent-ppo":
            c = s["config"]
            row["method"] += f"_{c['rp_observation']}_{c['rp_control']}_h{c['rp_hidden']}"
            row["method"] += "_split" if c["rp_split_head"] else ""
            row["method"] += "_finish" if c.get("rp_finish_assist", False) else ""
        if row["observation_protocol"] == "image+road-state":
            row["method"] += "_image_road_state"
        protocol = {k:v for k,v in s["config"].items() if k not in
                    ("seed", "out", "name", "resume", "device", "threads")}
        prior = configurations.setdefault(row["method"], protocol)
        if prior != protocol:
            raise ValueError(f"Hyperparameters differ within method {row['method']}; use separate report roots")
        row["decisions_per_second"] = finished.get("steps", s["checkpoint_steps"]) / max(row["train_seconds"],1e-9)
        ep = pd.read_csv(path.parent / "episodes.csv")
        ep["method"], ep["train_seed"], ep["run"] = row["method"], s["seed"], run.name
        episodes.append(ep)
        val = pd.read_csv(run/"validation.csv")
        val["method"], val["seed"] = row["method"], s["seed"]
        curves.append(val)
        row["validation_auc"] = float(np.trapz(val.mean_return, val.steps) / (val.steps.iloc[-1]-val.steps.iloc[0])) if len(val)>1 else None
        hit = val[val.mean_return>=900]
        row["first_validation_900_steps"] = None if hit.empty else int(hit.steps.iloc[0])
        up = pd.read_csv(run/"updates.csv")
        # Earlier recurrent runs recorded frame/time counters per update.
        # Recover exact matching validation counters without dropping that data.
        for counter in ("frames", "train_seconds"):
            if counter not in val and counter in up:
                val[counter] = np.interp(val.steps, up.steps, up[counter], left=0)
        up["method"], up["seed"] = row["method"], s["seed"]
        updates.append(up)
        runs.append(row)
        for video in sorted((path.parent/"videos").glob("*.mp4")):
            videos.append((run.name, video.relative_to(root).as_posix()))
    if not runs:
        raise SystemExit("No completed evaluations found. Run racing.evaluate or racing.suite first.")
    frame = pd.DataFrame(runs)
    if frame.observation_protocol.nunique() > 1:
        raise ValueError("Observation protocols differ; use separate report roots")
    if frame.completion_mode.nunique() > 1:
        raise ValueError("Completion modes differ; use separate report roots")
    if frame.duplicated(["method","seed"]).any():
        raise ValueError("Duplicate method/seed runs: use separate roots for different hyperparameter experiments")
    # Refuse silently mixing unequal budgets or differing test tracks in the main table.
    if frame.checkpoint_steps.nunique()>1 and args.selection=="test" and not args.allow_unequal_budgets:
        raise ValueError("Unequal final training budgets. Compare in separate report roots.")
    if frame.seed_start.nunique()>1 or frame.episodes.nunique()>1 or frame.domain_randomize.nunique()>1:
        raise ValueError("Test protocol differs between runs")
    frame.to_csv(out/"per_run.csv",index=False)
    pd.concat(episodes).to_csv(out/"all_test_episodes.csv",index=False)
    rng = np.random.default_rng(2025)
    groups = []
    for name, g in frame.groupby("method"):
        low, high = bootstrap(g.mean_return, rng)
        groups.append(dict(method=name, training_seeds=len(g), mean_return=g.mean_return.mean(),
            checkpoint_steps_min=int(g.checkpoint_steps.min()), checkpoint_steps_max=int(g.checkpoint_steps.max()),
            seed_std=g.mean_return.std(ddof=1) if len(g)>1 else None, ci95_low=low, ci95_high=high,
            completion_rate=g.completion_rate.mean(), mean_coverage=g.mean_coverage.mean(),
            score_ge_900_rate=g.score_ge_900_rate.mean(), train_hours=g.train_seconds.mean()/3600,
            decision_sps=g.decisions_per_second.mean(), validation_auc=g.validation_auc.mean()))
    table = pd.DataFrame(groups)
    table.to_csv(out/"comparison.csv",index=False)
    pairs = []
    for a,b in itertools.combinations(sorted(frame.method.unique()),2):
        merged = frame[frame.method==a].merge(frame[frame.method==b],on="seed",suffixes=("_a","_b"))
        diff = merged.mean_return_a-merged.mean_return_b
        if len(diff):
            low,high=bootstrap(diff,rng)
            pairs.append(dict(method_a=a,method_b=b,paired_seeds=len(diff),
                mean_difference=diff.mean(),ci95_low=low,ci95_high=high))
    pd.DataFrame(pairs).to_csv(out/"paired_differences.csv",index=False)
    cv = pd.concat(curves)
    up = pd.concat(updates)
    cv.to_csv(out/"validation_curves.csv",index=False)
    figures=[]
    for x, filename, label in (("steps","learning_curve","Policy decisions"),
                                ("frames","frame_curve","Actual raw frames"),
                                ("train_seconds","time_curve","Training seconds (evaluation excluded)")):
        fig,ax=plt.subplots(figsize=(8,4.6))
        for name,g in cv.groupby("method"):
            # No extrapolation outside the common observed support of the seeds.
            sequences=[v.sort_values(x) for _,v in g.groupby("seed")]
            lo=max(v[x].min() for v in sequences); hi=min(v[x].max() for v in sequences)
            grid=np.linspace(lo,hi,100)
            ys=np.array([np.interp(grid,v[x],v.mean_return) for v in sequences])
            mean=ys.mean(axis=0)
            ax.plot(grid,mean,label=f"{name} (n={len(ys)})")
            if len(ys)>1:
                ax.fill_between(grid,mean-ys.std(axis=0,ddof=1),mean+ys.std(axis=0,ddof=1),alpha=.15)
        ax.set(xlabel=label,ylabel="Validation raw return",title="Mean and +/-1 SD across training seeds")
        if cv.mean_return.max()-cv.mean_return.min()<20:
            center=float(cv.mean_return.mean())
            ax.set_ylim(center-10,center+10)
        ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/f"{filename}.png",dpi=160); plt.close(fig)
        figures.append(filename)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for name,g in up.groupby("method"):
        for metric,ax in zip(("value_mse","approx_kl","explained_variance"),axes):
            for seed,v in g.groupby("seed"):
                if metric in v and v[metric].notna().any():
                    ax.plot(v.steps,v[metric],alpha=.55,label=f"{name}/{seed}")
            ax.set(xlabel="Decisions",title=metric); ax.grid(alpha=.2)
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6)
        else:
            ax.text(.5,.5,"N/A: metric not recorded",ha="center",va="center",transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(out/"diagnostics.png",dpi=160); plt.close(fig)
    figures.append("diagnostics")
    if "q_loss" in up and up.q_loss.notna().any():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for metric, ax in zip(("q_loss", "alpha", "entropy"), axes):
            for (method, seed), g in up.groupby(["method", "seed"]):
                if g[metric].notna().any():
                    ax.plot(g.steps, g[metric], label=f"{method}/{seed}", alpha=.7)
            ax.set(xlabel="Decisions", title=metric)
            ax.grid(alpha=.2)
            ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(out/"sac_diagnostics.png", dpi=160); plt.close(fig)
        figures.append("sac_diagnostics")
    fig,ax=plt.subplots(figsize=(7,4))
    for i,(name,g) in enumerate(frame.groupby("method")):
        ax.scatter([i]*len(g),g.mean_return,label=name)
    ax.set_xticks(range(frame.method.nunique()), sorted(frame.method.unique()))
    ax.set(ylabel="Mean test return per trained seed",title="Independent training runs; each dot is one seed")
    if frame.mean_return.max()-frame.mean_return.min()<20:
        center=float(frame.mean_return.mean())
        ax.set_ylim(center-10,center+10)
    fig.tight_layout();fig.savefig(out/"test_seeds.png",dpi=160);plt.close(fig);figures.append("test_seeds")
    content='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CarRacing 实验报告</title>
<style>body{max-width:1200px;margin:36px auto;font:16px system-ui;line-height:1.7;padding:0 20px;background:#f4f7fc;color:#14283e}table{border-collapse:collapse;font-size:13px;background:white}td,th{padding:7px;border:1px solid #ccd6e3}img{max-width:100%;background:white}video{width:380px;max-width:100%}.card{display:inline-block;margin:8px;padding:12px;background:white}h1,h2{color:#12385b}</style>
<h1>CarRacing 算法对比</h1>'''
    content+=f"<p>数据目录：{html.escape(str(root))}；模型选择：{args.selection}。</p>"
    if args.allow_unequal_budgets:
        content+="<p><b>已允许不相等的实际预算：表中列出检查点决策数范围，本报告不能称为严格等样本预算比较。</b></p>"
    content+="<p>GRPO 为同赛道完整轨迹分组的连续控制适配版，不使用 critic/GAE；其价值网络指标为空。完整组采样可能略超请求预算，以实际 checkpoint_steps 为准；不同实际预算不合并为等预算最终模型比较。</p>"
    content+="<p><b>短跑仅用于验证管线，不代表收敛表现。</b>表中区间为训练种子均值的 bootstrap 95% 区间；单种子不计算区间，3 个种子的区间也很不稳定，建议最终 5 个。曲线阴影是种子间标准差，不是置信区间。测试赛道重复不等于独立训练重复。</p>"
    content+=table.round(3).to_html(index=False,na_rep="—")
    for name in figures:
        content+=f'<h2>{name}</h2><img src="{name}.png" alt="{name}">'
    content+="<h2>固定测试赛道视频</h2><p>展示预先指定的前几个测试种子，不按成绩挑选；播放器可全屏。每段为原始 50 FPS，动作重复不会加速视频。</p>"
    for name,path in videos:
        content+=f'<div class="card"><b>{html.escape(name)} / {html.escape(Path(path).stem)}</b><br><video controls preload="metadata" src="../{html.escape(path)}"></video></div>'
    content+="<h2>解释边界</h2><p>value_mse 是当前采样批次固定 GAE 目标上的拟合误差，不是无偏真实价值误差。时间曲线排除验证但包含采样和优化；云端账单还需计入安装、验证、测试。颜色随机化测试应另表报告。未达到 900 的阈值时间留空，不记成零。</p></html>"
    content = content.replace("</html>", "<p>SAC 的 q_loss 是双 Q 的 Bellman 回归损失之和，不与 GAE value_mse 混为同一指标。SAC 使用经验回放；等交互预算不代表等优化次数或等计算量。alpha 和 entropy 使用归一化动作空间的熵定义。</p></html>")
    (out/"index.html").write_text(content,encoding="utf-8")
    print(f"Report: {out/'index.html'}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
