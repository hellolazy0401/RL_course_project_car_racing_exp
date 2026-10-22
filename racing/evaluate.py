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
def evaluate_ensemble(policies, vec, device, seeds):
    """Native deterministic evaluation using existing parallel environment workers.

    Each member sees exactly the same seeds as sequential evaluation. No policy
    sampling, completion bonus, checkpoint selection, or auto-reset occurs here.
    """
    modes = [policy.training for policy in policies]
    rows = [[] for _ in policies]
    for policy in policies:
        policy.eval()
    try:
        for seed in seeds:
            observations = vec.reset_same_seed(int(seed))
            active = list(range(len(policies)))
            while active:
                actions = [policies[i].action(torch.as_tensor(observations[i:i+1], device=device),
                           deterministic=True)[0][0].cpu().numpy() for i in active]
                results = vec.step_active(active, actions)
                remaining = []
                for i, (obs, _, terminated, truncated, info) in zip(active, results):
                    observations[i] = obs
                    if terminated or truncated:
                        rows[i].append(dict(track_seed=int(seed), **info['episode_metrics']))
                    else:
                        remaining.append(i)
                active = remaining
            print(f"GRPO-C ensemble validation track={seed} finished", flush=True)
        return rows
    finally:
        for policy, mode in zip(policies, modes):
            policy.train(mode)


@torch.no_grad()
def evaluate(agent, device, config, seeds, video_dir=None, video_count=0, domain_randomize=False):
    from .recurrent_env import make_env
    env = make_env(config, domain_randomize=domain_randomize)
    rows = []
    was_training = agent.training
    agent.eval()
    try:
        for i, seed in enumerate(seeds):
            writer = None
            obs, _ = env.reset(seed=int(seed))
            recurrent_state = agent.initial_state(1, device) if getattr(agent, "recurrent", False) else None
            recurrent_start = torch.ones(1, dtype=torch.bool, device=device)
            context_length = getattr(agent, "context_length", 1)
            history = None
            if getattr(agent, "architecture", None) == "gru":
                from .sac_networks import ObservationHistory
                history = ObservationHistory(obs[None], context_length)
            if video_dir is not None and i < video_count:
                Path(video_dir).mkdir(parents=True, exist_ok=True)
                writer = imageio.get_writer(str(Path(video_dir) / f"track_{seed}.mp4"),
                    fps=50, codec="libx264", quality=7, macro_block_size=2)
            env.video_writer = writer
            try:
                while True:
                    policy_obs = history.data if history is not None else obs[None]
                    if recurrent_state is not None:
                        action, _, _, _, _, recurrent_state = agent.step(
                            torch.as_tensor(policy_obs, device=device), recurrent_state,
                            recurrent_start, deterministic=True)
                        recurrent_start.fill_(False)
                    else:
                        action, _, _, _ = agent.action(torch.as_tensor(policy_obs, device=device), deterministic=True)
                    obs, _, term, trunc, info = env.step(action[0].cpu().numpy())
                    if history is not None:
                        history.advance(obs[None], [term or trunc])
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
    if ckpt["config"]["algo"] == "recurrent-ppo":
        from .recurrent_ppo import agent_from_config
        agent = agent_from_config(ckpt["config"]).to(args.device)
    elif ckpt["config"]["algo"] == "sac":
        from .sac import SACAgent
        agent = SACAgent(ckpt["config"].get("sac_arch", "baseline"), ckpt["config"].get("sac_context", 8)).to(args.device)
    else:
        agent = Agent(with_critic=ckpt["config"]["algo"] not in ("grpo", "grpo-c"), road_state=ckpt["env_config"].get("road_state", False), road_scale=ckpt["config"].get("grpo_c_road_scale", .25)).to(args.device)
    agent.load_state_dict(ckpt["model"])
    out = Path(args.out or checkpoint.parent / ("test_ood" if args.domain_randomize else "test"))
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rows = evaluate(agent, args.device, env_config,
        range(args.seed_start, args.seed_start + args.episodes), out / "videos", args.videos, args.domain_randomize)
    for row in rows:
        append_csv(out / "episodes.csv", row)
    from .recurrent_ppo import completion_summary
    result = dict(**completion_summary(rows), algo=ckpt["config"]["algo"], seed=ckpt["config"]["seed"],
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
