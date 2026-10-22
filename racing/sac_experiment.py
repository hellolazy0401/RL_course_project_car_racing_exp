"""Sequential architecture study, validation-only selection and final delivery.

Run from a frozen source copy. Every command, configuration and failure is
recorded. Re-running skips completed stages; incomplete training is resumed
into a new segment, never overwritten. A process lock prevents duplicate jobs.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

from .sac_networks import ARCHITECTURES
from .utils import save_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def metrics(run):
    paths = [run] if (run/"validation.csv").exists() else []
    paths += sorted(run.glob("segment_*"))
    observations = {}
    best_path, best_value = None, -float("inf")
    final = None
    for segment in paths:
        if not (segment/"validation.csv").exists():
            continue
        rows = csv_rows(segment/"validation.csv")
        for row in rows:
            observations[int(row["steps"])] = row
        # best.pt includes the maximum in this segment, including initial val.
        value = max(float(row["mean_return"]) for row in rows)
        if value > best_value and (segment/"best.pt").exists():
            best_value, best_path = value, segment/"best.pt"
        if (segment/"finished.json").exists():
            final = segment
    if not final:
        raise RuntimeError(f"No completed training: {run}")
    ordered = [observations[s] for s in sorted(observations)]
    config = read(final/"config.json")
    finished = read(final/"finished.json")
    best_row = max(ordered, key=lambda r: float(r["mean_return"]))
    return dict(seed=config["seed"], run=str(run),
                final_run=str(final), best_checkpoint=str(best_path),
                best_validation=best_value, best_steps=int(best_row["steps"]),
                best_completion=float(best_row["completion_rate"]),
                final_validation=float(ordered[-1]["mean_return"]),
                final_completion=float(ordered[-1]["completion_rate"]),
                last_three_validation=sum(float(r["mean_return"]) for r in ordered[-3:])/len(ordered[-3:]),
                train_hours=finished["train_seconds"]/3600, total_steps=finished["steps"],
                **read(final/"model_info.json"))


class Study:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.out).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.source = Path(__file__).resolve().parents[1]

    def status(self, state, **data):
        save_json(self.root/"status.json", dict(status=state, updated_at=timestamp(),
                  coordinator_pid=os.getpid(), **data))

    def command(self, label, cmd):
        log_dir = self.root/"logs"
        log_dir.mkdir(exist_ok=True)
        with (self.root/"commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(time=timestamp(), label=label, cwd=str(self.source), argv=cmd))+"\n")
        with (log_dir/f"{label}.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(cmd, cwd=self.source, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, PYTHONUNBUFFERED="1"))
            self.status("running", job=label, child_pid=child.pid, log=str(log_dir/f"{label}.log"))
            code = child.wait()
        if code:
            raise RuntimeError(f"{label} exited with {code}; see {log_dir/f'{label}.log'}")

    def train(self, arch, seed, smoke=False):
        root = self.root/("smoke" if smoke else "training")
        run = root/f"{arch}_s{seed}"
        if (run/"complete.json").exists():
            return metrics(run)
        steps = 128 if smoke else self.args.steps
        cmd = [sys.executable, "-m", "racing.train", "--algo", "sac", "--sac-arch", arch,
               "--seed", str(seed), "--steps", str(steps), "--completion-mode", "lap95",
               "--num-envs", "2" if smoke else "4", "--rollout", "16" if smoke else "128",
               "--batch-size", "16" if smoke else "128", "--eval-every", "128" if smoke else "25000",
               "--eval-episodes", "1" if smoke else "10", "--device", self.args.device,
               "--sac-buffer-size", "256" if smoke else "50000",
               "--sac-learning-starts", "32" if smoke else "5000"]
        if (run/"finished.json").exists() or any(run.glob("segment_*/finished.json")):
            result = metrics(run)
        else:
            destination, name = root, run.name
            if run.exists():
                checkpoints = sorted([*run.glob("last.pt"), *run.glob("interrupted.pt"),
                                      *run.glob("segment_*/last.pt"), *run.glob("segment_*/interrupted.pt")],
                                     key=lambda path: path.stat().st_mtime)
                if not checkpoints:
                    # Keep the failed initialization and use a fresh segment.
                    destination, name = run, f"segment_{len(list(run.glob('segment_*')))+1:03d}"
                else:
                    destination, name = run, f"segment_{len(list(run.glob('segment_*')))+1:03d}"
                    cmd += ["--resume", str(checkpoints[-1])]
            cmd += ["--out", str(destination), "--name", name]
            self.command(f"{'smoke' if smoke else 'train'}_{arch}_s{seed}_{name}", cmd)
            result = metrics(run)
        if result["total_steps"] != steps:
            raise RuntimeError("Completed training budget does not match this study")
        save_json(run/"complete.json", result)
        return result

    def evaluate(self, label, checkpoint, episodes, seed_start, videos=0):
        out = self.root/"evaluations"/label
        if (out/"summary.json").exists():
            return read(out/"summary.json")
        if out.exists():
            # Preserve partial attempts rather than mixing episode CSVs.
            i = 1
            while out.with_name(out.name+f"_failed_{i}").exists():
                i += 1
            out.rename(out.with_name(out.name+f"_failed_{i}"))
        self.command(label, [sys.executable, "-m", "racing.evaluate", "--checkpoint", str(checkpoint),
                            "--out", str(out), "--device", self.args.device, "--episodes", str(episodes),
                            "--seed-start", str(seed_start), "--videos", str(videos)])
        return read(out/"summary.json")

    def run(self):
        plan = dict(architectures=list(ARCHITECTURES), screening_seed=0, confirmation_seed=1,
                    steps=self.args.steps, selection="mean best validation return over seeds 0 and 1 for baseline and screening winner",
                    validation_tracks="1000000..1000009", heldout_tracks="3000000..3000049",
                    frozen_source=str(self.source), created_at=timestamp())
        if not (self.root/"plan.json").exists():
            save_json(self.root/"plan.json", plan)
        elif read(self.root/"plan.json")["steps"] != self.args.steps:
            raise ValueError("Study budget changed; use a new study root")
        if self.args.smoke_only:
            for arch in ARCHITECTURES:
                result = self.train(arch, 0, smoke=True)
                self.evaluate("smoke_"+arch, result["best_checkpoint"], 1, 1000000, 0)
            self.status("smoke_completed")
            return
        results = []
        for arch in ARCHITECTURES:
            results.append(self.train(arch, 0))
            save_json(self.root/"screening.json", results)
        challenger = max((r for r in results if r["architecture"] != "baseline"),
                         key=lambda r: (r["best_validation"], r["last_three_validation"]))["architecture"]
        for arch in ("baseline", challenger):
            results.append(self.train(arch, 1))
            save_json(self.root/"training_results.json", results)
        averages = {arch: sum(r["best_validation"] for r in results if r["architecture"] == arch)/2
                    for arch in ("baseline", challenger)}
        winner = max(averages, key=averages.get)
        model = max((r for r in results if r["architecture"] == winner), key=lambda r: r["best_validation"])
        selection = dict(winner=winner, challenger=challenger, validation_means=averages, model=model,
                         selected_before_test=True, validation_improvement=averages[challenger]-averages["baseline"])
        save_json(self.root/"selection.json", selection)
        # Test data never changes which model is selected or which variant advances.
        for r in results:
            if r["architecture"] in ("baseline", challenger):
                test = self.evaluate(f"test_{r['architecture']}_s{r['seed']}", r["best_checkpoint"],
                                     50, 3000000, 3 if r == model else 0)
                r["test_return"] = test["mean_return"]
                r["test_completion"] = test["completion_rate"]
        save_json(self.root/"results.json", results)
        self.deliver(results, selection)
        self.status("completed", delivery=str(self.root/"delivery"), winner=winner)

    def deliver(self, results, selection):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        delivery = self.root/"delivery"
        delivery.mkdir(exist_ok=True)
        checkpoint = Path(selection["model"]["best_checkpoint"])
        shutil.copy2(checkpoint, delivery/"model.pt")
        code = delivery/"code"
        code.mkdir(exist_ok=True)
        for folder in ("racing", "tests"):
            shutil.copytree(self.source/folder, code/folder, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for filename in ("requirements.txt", "requirements-resolved-local.txt"):
            if (self.source/filename).exists():
                shutil.copy2(self.source/filename, code/filename)
        records = delivery/"records"
        records.mkdir(exist_ok=True)
        for filename in ("plan.json", "selection.json", "results.json", "commands.jsonl", "source_manifest.json", "preflight.json"):
            if (self.root/filename).exists():
                shutil.copy2(self.root/filename, records/filename)
        if (self.source/"docs/SAC网络结构实验计划.md").exists():
            shutil.copy2(self.source/"docs/SAC网络结构实验计划.md", records/"实验计划.md")
        for r in results:
            source = Path(r["run"])
            target = records/f"{r['architecture']}_s{r['seed']}"
            for folder in [source, *sorted(source.glob("segment_*"))]:
                dest = target/folder.relative_to(source)
                dest.mkdir(parents=True, exist_ok=True)
                for p in folder.iterdir():
                    if p.is_file() and p.suffix in (".csv", ".json"):
                        shutil.copy2(p, dest/p.name)
        shutil.copytree(self.root/"evaluations", records/"evaluations", dirs_exist_ok=True)
        shutil.copytree(self.root/"logs", records/"logs", dirs_exist_ok=True)
        fields = sorted(set().union(*(r.keys() for r in results)))
        with (delivery/"comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)
        fig, ax = plt.subplots(figsize=(10, 5))
        for r in results:
            vals = {}
            run = Path(r["run"])
            for segment in [run, *sorted(run.glob("segment_*"))]:
                if (segment/"validation.csv").exists():
                    for row in csv_rows(segment/"validation.csv"):
                        vals[int(row["steps"])] = float(row["mean_return"])
            ax.plot(sorted(vals), [vals[k] for k in sorted(vals)], label=f"{r['architecture']} / seed {r['seed']}")
        ax.set(xlabel="Policy decisions", ylabel="Validation mean return", title="SAC architecture study")
        ax.grid(alpha=.2); ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(delivery/"learning_curves.png", dpi=160); plt.close(fig)
        improved = selection["winner"] != "baseline"
        lines = ["# SAC 网络结构实验记录", "", "## 结论", "",
                 ("候选结构在两种子验证选模指标上优于原网络。" if improved else "本轮没有确认候选结构优于原网络，交付原网络基线中的最佳验证模型。"),
                 f"交付结构：{selection['winner']}；训练种子：{selection['model']['seed']}；保存决策数：{selection['model']['best_steps']}。",
                 "这是先筛选再复验的探索实验，两种子不足以证明普遍优势。测试集只在模型选择完成后使用；若测试未确认提升，应如实以测试结果为准。",
                 "", "## 实验协议", "", f"五结构 seed 0 各 {self.args.steps} 决策，候选与基线追加 seed 1。lap95，图像输入，4 环境，batch 128；其余 SAC 参数不变。",
                 "每 25000 决策验证 10 条固定赛道。先以 seed 0 的最佳验证均值选出一个改动，再按候选与基线两种子的最佳验证均值选结构，最后选该结构中验证最好的单模型。",
                 "GRU 使用 8 决策有限历史窗口，连续序列回放、回合边界清零、末转移损失；不是无限历史 RNN。不同结构的参数量、计算量不同，统一的是环境交互预算。",
                 "", "|结构|种子|最佳验证|末次验证|末三次验证均值|新测试均值|测试完赛率|训练小时|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in results:
            test = f"{r['test_return']:.2f}" if "test_return" in r else "未测试（筛选未入选）"
            completion = f"{r['test_completion']:.0%}" if "test_completion" in r else "—"
            lines.append(f"|{r['architecture']}|{r['seed']}|{r['best_validation']:.2f}|{r['final_validation']:.2f}|{r['last_three_validation']:.2f}|{test}|{completion}|{r['train_hours']:.2f}|")
        lines += ["", "![验证曲线](learning_curves.png)", "", "## 交付与复现", "",
                  "model.pt 是所选评估模型；完整源码位于 code/，配置、源文件哈希、每条命令、原始 CSV 和日志位于 records/。last.pt 和经验池保留在原实验目录，未塞入交付压缩包。",
                  "从 code 目录执行：", "", "```powershell",
                  "python -m racing.evaluate --checkpoint ../model.pt --episodes 50 --seed-start 3000000 --videos 3 --device cuda --out ../evaluation_recheck",
                  "```", "", "best.pt 不含优化器和经验池，不能直接续训。更改结构/预算的实验应另建目录，不与旧基线混报。"]
        (delivery/"实验记录.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        save_json(delivery/"model_manifest.json", dict(sha256=hashlib.sha256((delivery/"model.pt").read_bytes()).hexdigest(),
                  source=str(checkpoint), **selection))
        archive = self.root/"SAC_architecture_delivery.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(delivery.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(delivery))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=300000)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    study = Study(args)
    lock = study.root/"coordinator.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    if os.name == "nt":
        import ctypes
        # Scoped idle-sleep prevention, released on exit; no power-plan edits.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        study.run()
    except BaseException as exc:
        study.status("failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
