import argparse
from pathlib import Path
import time
import imageio.v2 as imageio
import numpy as np
import torch
from .envs import Racing
from .model import Agent
from .utils import append_csv, save_json, summary


@torch.no_grad()
def evaluate(agent, device, config, seeds, video_dir=None, video_count=0, domain_randomize=False):
    env = Racing(**config, domain_randomize=domain_randomize)
    rows = []
    was_training = agent.training
    agent.eval()
    try:
        for i, seed in enumerate(seeds):
            writer = None
            obs, _ = env.reset(seed=int(seed))
            if video_dir is not None and i < video_count:
                Path(video_dir).mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(str(Path(video_dir) / f"track_{seed}.mp4"),
                    fps=50, codec="libx264", quality=7, macro_block_size=2)
            env.video_writer = writer
            try:
                while True:
                    action, _, _, _ = agent.action(torch.as_tensor(obs[None], device=device), deterministic=True)
                    obs, _, term, trunc, info = env.step(action[0].cpu().numpy())
                    if term or trunc:
                        rows.append(dict(track_seed=int(seed), **info["episode_metrics"]))
                        break
            finally:
                env.video_writer = None
                if writer is not None:
                    writer.close()
    finally:
        env.close()
        agent.train(was_training)
    return rows


def main():
    p = argparse.ArgumentParser(description="Held-out deterministic evaluation and full-frame MP4 recording")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=2_000_000)
    p.add_argument("--videos", type=int, default=3)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--domain-randomize", action="store_true")
    p.add_argument("--completion-mode", choices=["original", "lap95"], default=None,
                   help="Default: checkpoint protocol. Explicit override is a separate environment diagnostic.")
    p.add_argument("--out")
    args = p.parse_args()
    if args.episodes < 1 or args.videos < 0:
        p.error("episodes must be positive; videos must be non-negative")
    checkpoint = Path(args.checkpoint)
    # Only load checkpoints produced by this project / trusted by the user.
    ckpt = torch.load(checkpoint, map_location=args.device, weights_only=False)
    env_config = dict(ckpt["env_config"])
    checkpoint_completion_mode = env_config.get("completion_mode", "original")
    env_config["completion_mode"] = args.completion_mode or checkpoint_completion_mode
    torch.set_num_threads(2)
    agent = Agent().to(args.device)
    agent.load_state_dict(ckpt["model"])
    out = Path(args.out or checkpoint.parent / ("test_ood" if args.domain_randomize else "test"))
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rows = evaluate(agent, args.device, env_config,
        range(args.seed_start, args.seed_start + args.episodes), out / "videos", args.videos, args.domain_randomize)
    for row in rows:
        append_csv(out / "episodes.csv", row)
    result = dict(**summary(rows), algo=ckpt["config"]["algo"], seed=ckpt["config"]["seed"],
        checkpoint=str(checkpoint.resolve()), checkpoint_steps=ckpt["steps"],
        train_seconds=ckpt["train_seconds"], evaluation_seconds=time.perf_counter()-start,
        domain_randomize=args.domain_randomize, seed_start=args.seed_start,
        deterministic=True, config=ckpt["config"], env_config=env_config,
        completion_mode=env_config["completion_mode"], checkpoint_completion_mode=checkpoint_completion_mode,
        protocol_override=env_config["completion_mode"] != checkpoint_completion_mode)
    save_json(out / "summary.json", result)
    print(result)


if __name__ == "__main__":
    main()
