"""Reproducible completion-first Recurrent PPO study and final held-out test.

Use --phase train for candidate runs, then --phase finalize after reviewing
validation. Finalization freezes selection before it touches the test tracks.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import numpy as np
import torch
from .utils import append_csv, save_json
from .recurrent_ppo import completion_summary, selection_key, agent_from_config
from .recurrent_env import RecurrentRacing
from .evaluate import evaluate


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def frozen_source(root):
    source = Path(__file__).resolve().parents[1]
    dest = root / "source"
    if not dest.exists():
        dest.mkdir(parents=True)
        for name in ("racing", "tests", "scripts"):
            shutil.copytree(source / name, dest / name, ignore=shutil.ignore_patterns("__pycache__"))
        for name in ("README.md", "requirements.txt", "requirements-resolved-local.txt", "pytest.ini"):
            shutil.copy2(source / name, dest / name)
        if (source / "docs").exists():
            shutil.copytree(source / "docs", dest / "docs")
        save_json(root / "source_manifest.json", {
            str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dest.rglob("*") if p.is_file()})
    return dest


def run_command(root, source, name, command):
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    with (root / "commands.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(time=timestamp(), job=name, argv=command, cwd=str(source))) + "\n")
    with (logs / f"{name}.log").open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"))
        save_json(root / "status.json", dict(status="running", job=name, pid=proc.pid,
                                           updated_at=timestamp(), log=str(logs / f"{name}.log")))
        code = proc.wait()
        if code:
            save_json(root / "status.json", dict(status="failed", job=name, exit_code=code,
                                               updated_at=timestamp(), log=str(logs / f"{name}.log")))
            raise RuntimeError(f"Failed: {name}; see log")


def train_candidate(root, source, name, *, observation="state", control="residual",
                    seed=0, split=False, hidden=128, steps=196608):
    out = root / "training" / name
    if (out / "finished.json").exists() or any(out.glob("segment_*/finished.json")):
        return
    cmd = [sys.executable, "-m", "racing.train", "--algo", "recurrent-ppo",
           "--name", name, "--out", str(root / "training"), "--seed", str(seed),
           "--steps", str(steps), "--num-envs", "8", "--rollout", "256",
           "--batch-size", "256", "--rp-sequence", "32", "--ppo-epochs", "4",
           "--rp-observation", observation, "--rp-control", control,
           "--rp-hidden", str(hidden), "--rp-speed-scale", "2.2",
           "--rp-residual-scale", ".2", "--actor-lr", ".0001" if control == "residual" else ".0003",
           "--critic-lr", ".0003", "--gamma", ".995", "--reward-scale", ".01",
           "--target-kl", ".02", "--max-grad-norm", ".5", "--eval-every", "16384",
           "--eval-episodes", "30", "--completion-mode", "lap95",
           "--device", "cuda" if observation == "pixels" else "cpu"]
    if split:
        cmd.append("--rp-split-head")
    if out.exists():
        checkpoints = [p for p in [out / "last.pt", *out.glob("segment_*/last.pt")] if p.exists()]
        ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime) if checkpoints else out / "last.pt"
        if not ckpt.exists():
            raise RuntimeError(f"Incomplete run without checkpoint: {out}")
        # Segment subdirectories preserve the earlier evidence and best weights.
        segment = 1
        while (out / f"segment_{segment}").exists():
            segment += 1
        cmd[cmd.index("--name")+1] = f"{name}/segment_{segment}"
        cmd += ["--resume", str(ckpt)]
    run_command(root, source, name, cmd)


def controller_evaluate(out, seeds, speed_scale=2.2, profile="v1"):
    out.mkdir(parents=True, exist_ok=False)
    env = RecurrentRacing(completion_mode="lap95", control_mode="residual", speed_scale=speed_scale,
                         controller_profile=profile)
    rows = []
    try:
        for seed in seeds:
            env.reset(seed=int(seed))
            while True:
                _, _, term, trunc, info = env.step(np.zeros(2, np.float32))
                if term or trunc:
                    row = dict(track_seed=int(seed), **info["episode_metrics"])
                    rows.append(row)
                    append_csv(out / "episodes.csv", row)
                    break
    finally:
        env.close()
    result = dict(**completion_summary(rows), speed_scale=speed_scale, controller_profile=profile, learned=False)
    save_json(out / "summary.json", result)
    return result


def evaluate_checkpoint(path, out, seeds, videos=0):
    torch.set_num_threads(2)
    if (out / "summary.json").exists():
        cached = read(out / "summary.json")
        if (cached.get("checkpoint_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()
                or cached.get("track_seed_start") != seeds.start or cached.get("track_seed_stop") != seeds.stop):
            raise ValueError(f"Stale evaluation cache; preserve it and use a new output directory: {out}")
        return cached
    out.mkdir(parents=True, exist_ok=False)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    agent = agent_from_config(ck["config"])
    agent.load_state_dict(ck["model"])
    rows = evaluate(agent, "cpu", ck["env_config"], seeds, out / "videos", videos)
    for row in rows:
        append_csv(out / "episodes.csv", row)
    result = dict(**completion_summary(rows), checkpoint=str(path), checkpoint_steps=ck["steps"],
                  config=ck["config"], env_config=ck["env_config"],
                  checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  track_seed_start=seeds.start, track_seed_stop=seeds.stop)
    save_json(out / "summary.json", result)
    return result


def confirm_candidates(root, candidates):
    torch.set_num_threads(2)
    results = []
    for name in candidates:
        run = root / "training" / name
        if not (run / "finished.json").exists() and not any(run.glob("segment_*/finished.json")):
            raise RuntimeError(f"Candidate has not finished: {name}")
        # Choose across continuation segments using their original fixed
        # validation set, then confirm one candidate per experimental run.
        paths = [p for p in [run / "best.pt", *sorted(run.glob("segment_*/best.pt"))] if p.exists()]
        if paths:
            checkpoint = max(paths, key=lambda p: selection_key(read(p.parent / "best_metrics.json")))
            if read(checkpoint.parent / "best_metrics.json")["steps"] == 0:
                # A collapsed run may never beat initialization. Evaluate its
                # trained final model, not a misleading "trained" random policy.
                checkpoint = max([p for p in [run / "last.pt", *run.glob("segment_*/last.pt")] if p.exists()],
                                 key=lambda p: p.stat().st_mtime)
            result = evaluate_checkpoint(checkpoint, root / "confirmation" / f"{name}_0",
                                         range(1100000, 1100100))
            results.append(dict(name=name, **result))
            print(json.dumps(dict(name=name, completion_rate=result["completion_rate"],
                                  mean_finish_seconds=result["mean_finish_seconds"])), flush=True)
    return results


def finalize(root, candidates, test_episodes=100, test_workers=2):
    torch.set_num_threads(2)
    selection_file = root / "selection.json"
    if selection_file.exists():
        selection = read(selection_file)
    else:
        results = confirm_candidates(root, candidates)
        if not results:
            raise RuntimeError("No completed candidates")
        # Do not pass off an untrained controller initialization as learned PPO.
        learned = [r for r in results if r["checkpoint_steps"] > 0]
        if not learned:
            raise RuntimeError("No trained candidate improved on initialization")
        winner = max(learned, key=selection_key)
        selection = dict(selected_at=timestamp(), rule="completion_rate, then lower successful finish time, then coverage/return",
                         winner=winner, candidates=results, test_seed_start=4000000,
                         test_episodes=test_episodes, selection_uses_test=False)
        save_json(selection_file, selection)
    results, jobs = [], []
    seeds = range(selection["test_seed_start"], selection["test_seed_start"] + selection["test_episodes"])
    for result in selection["candidates"]:
        name = result["name"]
        # One selected checkpoint per named candidate for final test.
        same = [r for r in selection["candidates"] if r["name"] == name]
        chosen = max(same, key=selection_key)
        if result != chosen:
            continue
        if hashlib.sha256(Path(result["checkpoint"]).read_bytes()).hexdigest() != result["checkpoint_sha256"]:
            raise ValueError(f"Selected checkpoint changed after confirmation: {name}")
        jobs.append((name, Path(result["checkpoint"]), root / "test" / name,
                     3 if result["checkpoint"] == selection["winner"]["checkpoint"] else 0))
    save_json(root / "status.json", dict(status="testing", selected=selection["winner"]["name"],
              updated_at=timestamp(), candidates=[j[0] for j in jobs], completed=[]))
    # Separate processes keep SDL/Box2D independent. Each policy still uses
    # batch-size-one CPU inference and the same track seeds as confirmation.
    with ProcessPoolExecutor(max_workers=min(test_workers, len(jobs)), mp_context=mp.get_context("spawn")) as pool:
        pending = {pool.submit(evaluate_checkpoint, checkpoint, output, seeds, videos): name
                   for name, checkpoint, output, videos in jobs}
        for future in as_completed(pending):
            name, stats = pending[future], future.result()
            results.append(dict(name=name, **stats))
            save_json(root / "status.json", dict(status="testing", selected=selection["winner"]["name"],
                      updated_at=timestamp(), completed=[r["name"] for r in results], total=len(jobs)))
            print(json.dumps(dict(event="test", name=name, completion_rate=stats["completion_rate"],
                                  mean_finish_seconds=stats["mean_finish_seconds"])), flush=True)
    order = {name: i for i, (name, *_) in enumerate(jobs)}
    results.sort(key=lambda r: order[r["name"]])
    baseline_out = root / "test" / "controller_only"
    baseline = (read(baseline_out / "summary.json") if (baseline_out / "summary.json").exists()
                else controller_evaluate(baseline_out, seeds, speed_scale=2.2, profile="v2"))
    results.append(dict(name="controller_only", **baseline))
    if baseline["controller_profile"] != "v1":
        original_out = root / "test" / "controller_v1"
        original = (read(original_out / "summary.json") if (original_out / "summary.json").exists()
                    else controller_evaluate(original_out, seeds))
        results.append(dict(name="controller_v1", **original))
    save_json(root / "results.json", results)
    delivery = root / "delivery"
    delivery.mkdir(exist_ok=True)
    shutil.copy2(selection["winner"]["checkpoint"], delivery / "best_model.pt")
    save_json(delivery / "model_card.json", dict(
        algorithm="Recurrent PPO", source=selection["winner"]["checkpoint"],
        **{k: selection["winner"][k] for k in ("checkpoint_steps", "config", "env_config", "checkpoint_sha256")},
        observation_scope="Privileged road geometry if rp_observation=state; not comparable as a pixel-only algorithm ablation.",
        selection="Frozen using 100 confirmation validation tracks before testing on 100 fresh tracks.",
        objective="Completion rate first, successful lap time second. Same lap95 and 1000-frame cap.",
        finish_assist=selection["winner"]["env_config"].get("finish_assist", False),
        derivation=(read(Path(selection["winner"]["checkpoint"]).parent / "derivation.json")
                    if (Path(selection["winner"]["checkpoint"]).parent / "derivation.json").exists() else None),
        usage="python -m racing.evaluate --checkpoint delivery/best_model.pt --episodes 100 --seed-start 5000000 --videos 3 --out new_evaluation"))
    save_json(root / "status.json", dict(status="complete", updated_at=timestamp(),
              selected=selection["winner"]["name"], delivery=str(delivery)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--phase", choices=["train", "confirm", "finalize"], required=True)
    p.add_argument("--candidates", nargs="+", default=["state_direct_s0", "residual_s0", "residual_s1", "residual_split_s0"])
    p.add_argument("--test-episodes", type=int, default=100)
    p.add_argument("--test-workers", type=int, default=2)
    args = p.parse_args()
    if args.test_workers < 1 or args.test_episodes < 1:
        p.error("test-workers and test-episodes must be positive")
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "train":
            source = frozen_source(root)
            save_json(root / "plan.json", dict(
                created_at=timestamp(), train_tracks="0..999999 random, never validation/test",
                validation_tracks="1000000..1000029",
                confirmation_tracks="1100000..1100099", test_tracks="4000000..4000099",
                objective="completion rate, then successful lap time",
                budget_note="State direct early pilot 98304 decisions; residual variants 196608 each.",
                variants=args.candidates))
            for name in args.candidates:
                train_candidate(root, source, name,
                    observation="pixels" if name.startswith("pixels") else "state",
                    control="direct" if "direct" in name or name.startswith("pixels") else "residual",
                    seed=int(name.rsplit("_s", 1)[-1]), split="split" in name,
                    hidden=256 if "h256" in name else 128)
            save_json(root / "status.json", dict(status="training_complete", updated_at=timestamp()))
        elif args.phase == "confirm":
            confirm_candidates(root, args.candidates)
        else:
            finalize(root, args.candidates, args.test_episodes, args.test_workers)
    except BaseException as error:
        save_json(root / "status.json", dict(status="failed", updated_at=timestamp(),
                  error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
