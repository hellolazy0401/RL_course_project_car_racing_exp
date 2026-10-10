"""Paired PPO continuation experiment; run from the project root."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from .utils import save_json


INHERITED = ("algo", "seed", "num_envs", "rollout", "batch_size", "threads", "actor_lr",
             "critic_lr", "gamma", "gae_lambda", "reward_scale", "entropy_coef", "value_steps",
             "ppo_epochs", "clip", "max_grad_norm", "critic_grad_norm")


def training_command(parent, config, end_steps, out, arm, args):
    command = [sys.executable, "-m", "racing.train", "--branch-from", str(parent),
               "--steps", str(end_steps), "--device", args.device, "--out", str(out), "--name", arm,
               "--eval-every", str(args.eval_every), "--eval-episodes", str(args.eval_episodes),
               "--eval-seed-start", str(args.eval_seed_start)]
    for key in INHERITED:
        command += ["--" + key.replace("_", "-"), str(config[key])]
    command += ["--normalize-adv" if config["normalize_adv"] else "--no-normalize-adv"]
    command += ["--completion-mode", config.get("completion_mode", "original")]
    if arm == "kl":
        command += ["--target-kl", str(args.target_kl)]
    return command


def run(command):
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def read_csv(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def report(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dest = out / "report"
    dest.mkdir(parents=True, exist_ok=True)
    rows, initial = [], []
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for arm in ("control", "kl"):
        directory = out / arm
        if not (directory / "validation.csv").exists():
            continue
        val = read_csv(directory / "validation.csv")
        best = max(val, key=lambda r: float(r["mean_return"]))
        initial.append((arm, float(val[0]["mean_return"])))
        end = json.loads((directory / "finished.json").read_text()) if (directory / "finished.json").exists() else {}
        updates = read_csv(directory / "updates.csv") if (directory / "updates.csv").exists() else []
        for kind, row in (("best", best), ("last", val[-1])):
            item = dict(arm=arm, checkpoint=kind, validation_steps=int(row["steps"]),
                        validation_return=float(row["mean_return"]),
                        validation_coverage=float(row["mean_coverage"]),
                        validation_completion=float(row["completion_rate"]),
                        added_decisions=end.get("added_steps", ""),
                        segment_train_hours=end.get("segment_train_seconds", 0)/3600,
                        wall_hours=end.get("session_wall_seconds", 0)/3600,
                        actor_updates_mean=float(np.mean([float(r["actor_updates"]) for r in updates])) if updates else 0,
                        kl_stop_fraction=float(np.mean([float(r["kl_early_stop"]) for r in updates])) if updates else 0,
                        test_return="", test_coverage="", test_completion="")
            test = directory / f"test_{kind}" / "summary.json"
            if test.exists():
                data = json.loads(test.read_text())
                item.update(test_return=data["mean_return"], test_coverage=data["mean_coverage"], test_completion=data["completion_rate"])
            rows.append(item)
        axes[0].plot([int(r["steps"])/1000 for r in val], [float(r["mean_return"]) for r in val], "o-", label=arm)
        if updates:
            v = np.array([float(r["approx_kl"]) for r in updates])
            med = [np.median(v[max(0, i-24):i+1]) for i in range(len(v))]
            axes[1].plot([int(r["steps"])/1000 for r in updates], med, marker="." if len(updates) == 1 else None, label=arm)
    if not rows:
        plt.close(fig)
        raise ValueError("No validation results found")
    axes[0].set_ylabel("Validation mean return")
    axes[1].set_ylabel("Post-update KL: trailing median (25)")
    axes[1].set_xlabel("Cumulative decisions (thousands)")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "comparison.png", dpi=140)
    plt.close(fig)
    with (dest / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    same = len(initial) == 2 and abs(initial[0][1]-initial[1][1]) < 1e-6
    headings = "".join(f"<th>{html.escape(k)}</th>" for k in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{v:.4f}</td>" if isinstance(v, float) else f"<td>{html.escape(str(v))}</td>" for v in r.values()) + "</tr>" for r in rows)
    (dest / "index.html").write_text(f'''<!doctype html><meta charset="utf-8"><title>PPO KL 对照实验</title>
<style>body{{font:16px sans-serif;margin:30px}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ccc;padding:8px}}img{{max-width:100%}}.table{{overflow:auto}}</style>
<h1>PPO KL 对照实验</h1><p>两组起始验证回报一致：{same}。只有一个训练种子；验证集用于选择 best，不是独立测试。</p>
<p>best 可以保留分支起点，避免将退化后的模型当成最佳。last 是最终预算模型。先比较驾驶成绩，再看 KL；低 KL 本身不是成功。</p>
<img src="comparison.png"><div class="table"><table><tr>{headings}</tr>{body}</table></div>
<p>train hours 为新增训练部分，wall hours 包含本次评估。测试列空白表示尚未测试。验证步数以日志为准；未完成的运行显示当前已有验证记录。</p>''', encoding="utf-8")
    print(f"Report: {(dest / 'index.html').resolve()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["train", "report", "videos", "test"], default="train")
    p.add_argument("--arm", choices=["both", "control", "kl"], default="both")
    p.add_argument("--checkpoint", default="runs_tuning_lr1e4_1000k/ppo_s0/best.pt")
    p.add_argument("--out", default="runs_ppo_kl_ablation")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--additional-steps", type=int, default=200000)
    p.add_argument("--target-kl", type=float, default=.02)
    p.add_argument("--eval-every", type=int, default=100000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--eval-seed-start", type=int, default=1000000)
    p.add_argument("--test-episodes", type=int, default=50)
    p.add_argument("--test-seed-start", type=int, default=3000000)
    args = p.parse_args()
    out = Path(args.out)
    arms = ["control", "kl"] if args.arm == "both" else [args.arm]
    if args.stage == "report":
        report(out)
        return
    if args.stage == "train":
        parent = Path(args.checkpoint).resolve()
        ck = torch.load(parent, map_location="cpu", weights_only=False)
        if ck["config"]["algo"] != "ppo":
            p.error("Parent checkpoint must be PPO")
        if args.additional_steps <= 0 or args.additional_steps % ck["config"]["num_envs"]:
            p.error("additional-steps must be positive and divisible by num-envs")
        if not np.isfinite(args.target_kl) or args.target_kl <= 0 or min(args.eval_every, args.eval_episodes) <= 0 or args.eval_seed_start < 0:
            p.error("Invalid KL or evaluation settings")
        for arm in arms:
            if (out / arm).exists():
                p.error(f"Output already exists: {out / arm}. Choose a new --out, or run the other --arm")
        manifest = dict(parent=str(parent), parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
                        initial_steps=ck["steps"], end_steps=ck["steps"]+args.additional_steps,
                        target_kl=args.target_kl, eval_every=args.eval_every, eval_episodes=args.eval_episodes,
                        eval_seed_start=args.eval_seed_start, device=args.device)
        manifest_path = out / "experiment.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            p.error("Experiment settings differ from existing experiment.json; choose a new --out")
        save_json(manifest_path, manifest)
        for arm in arms:
            run(training_command(parent, ck["config"], manifest["end_steps"], out, arm, args))
        report(out)
    else:
        commands = []
        for arm in arms:
            for kind in (["best"] if args.stage == "videos" else ["best", "last"]):
                directory = out / arm
                dest = directory / ("validation_video_best" if args.stage == "videos" else f"test_{kind}")
                if not (directory / f"{kind}.pt").exists() or dest.exists():
                    p.error(f"Missing checkpoint or output already exists: {directory}, {dest}")
                seed = args.eval_seed_start if args.stage == "videos" else args.test_seed_start
                episodes = 5 if args.stage == "videos" else args.test_episodes
                if episodes < 1 or seed < 0:
                    p.error("Invalid evaluation episodes or seed")
                commands.append([sys.executable, "-m", "racing.evaluate", "--checkpoint", str(directory / f"{kind}.pt"),
                                 "--device", args.device, "--episodes", str(episodes), "--seed-start", str(seed),
                                 "--videos", "5", "--out", str(dest)])
        for command in commands:
            run(command)
        if args.stage == "test":
            report(out)


if __name__ == "__main__":
    main()
