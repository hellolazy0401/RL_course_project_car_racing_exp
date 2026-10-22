"""PPO-Clip with a persistent LSTM and contiguous truncated BPTT minibatches.

Hidden states are reset per episode, not per rollout. Time-limit bootstrap uses
the terminal observation and its own recurrent state, never the auto-reset obs.
"""
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_
from .envs import VectorRacing
from .model import gae
from .recurrent_env import STATE_DIM
from .utils import append_csv, provenance, save_json, summary


class RecurrentAgent(nn.Module):
    recurrent = True

    def __init__(self, observation_mode="state", hidden_size=128, split_head=False,
                 control_mode="direct"):
        super().__init__()
        self.observation_mode, self.hidden_size = observation_mode, hidden_size
        self.split_head, self.control_mode = split_head, control_mode
        if observation_mode == "pixels":
            self.encoder = nn.Sequential(
                nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(), nn.Conv2d(64, 64, 3), nn.ReLU(), nn.Flatten(),
                nn.Linear(1024, hidden_size), nn.ReLU())
        else:
            self.encoder = nn.Sequential(nn.Linear(STATE_DIM, hidden_size), nn.Tanh(),
                                         nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        if split_head:
            self.actor = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_size, 64), nn.Tanh(), nn.Linear(64, 1))
                for _ in range(2)])
        else:
            self.actor = nn.Linear(hidden_size, 2)
        self.critic = nn.Linear(hidden_size, 1)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight, np.sqrt(2))
                nn.init.zeros_(module.bias)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            else:
                nn.init.orthogonal_(param)
        heads = [a[-1] for a in self.actor] if split_head else [self.actor]
        for head in heads:
            nn.init.orthogonal_(head.weight, .01)
        nn.init.orthogonal_(self.critic.weight, 1)
        if control_mode == "direct":
            with torch.no_grad():
                (self.actor[1][-1].bias if split_head else self.actor.bias[1:]).fill_(.5)
        self.logstd = nn.Parameter(torch.full((2,), -.7 if control_mode == "direct" else -1.5))

    def initial_state(self, batch, device=None):
        device = device or next(self.parameters()).device
        return tuple(torch.zeros(1, batch, self.hidden_size, device=device) for _ in range(2))

    def features(self, obs, state, starts):
        """Inputs [time,batch,...], initial state [1,batch,hidden]."""
        time_steps, batch = obs.shape[:2]
        x = obs.flatten(0, 1).float()
        if self.observation_mode == "pixels":
            x = x / 255.
        x = self.encoder(x).reshape(time_steps, batch, self.hidden_size)
        if not starts.any():
            return self.lstm(x, state)
        outputs = []
        for t in range(time_steps):
            mask = (~starts[t]).float().view(1, batch, 1)
            state = tuple(s * mask for s in state)
            y, state = self.lstm(x[t:t+1], state)
            outputs.append(y)
        return torch.cat(outputs), state

    def sequence(self, obs, state, starts, latent=None, deterministic=False):
        features, state = self.features(obs, state, starts)
        mean = (torch.cat([head(features) for head in self.actor], -1)
                if self.split_head else self.actor(features))
        dist = Normal(mean, self.logstd.clamp(-4, .5).exp())
        latent = (mean if deterministic else dist.sample()) if latent is None else latent
        # Stored pre-tanh actions have exact PPO likelihood ratios: the fixed
        # transform Jacobian cancels between old and new policy densities.
        return (latent.tanh(), latent, dist.log_prob(latent).sum(-1),
                dist.entropy().sum(-1), self.critic(features).squeeze(-1), state)

    def step(self, obs, state, starts, deterministic=False):
        result = self.sequence(obs[None], state, starts[None], deterministic=deterministic)
        return tuple(x[0] for x in result[:-1]) + (result[-1],)


def agent_from_config(config):
    return RecurrentAgent(config["rp_observation"], config["rp_hidden"],
                          config["rp_split_head"], config["rp_control"])


def completion_summary(rows):
    result = summary(rows)
    times = [r["raw_frames"] / 50 for r in rows if r["completed"]]
    result.update(mean_finish_seconds=float(np.mean(times)) if times else None,
                  median_finish_seconds=float(np.median(times)) if times else None,
                  # Failure penalty reports time without silently dropping failures.
                  penalized_seconds=float(np.mean([r["raw_frames"] / 50 if r["completed"] else 40 for r in rows])))
    n, p = len(rows), result["completion_rate"]
    z = 1.96
    midpoint = (p + z*z/(2*n))/(1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1+z*z/n)
    result["completion_wilson95"] = [max(0., midpoint-half), min(1., midpoint+half)]
    return result


def selection_key(metrics):
    return (metrics["completion_rate"],
            -metrics["mean_finish_seconds"] if metrics["mean_finish_seconds"] is not None else -1000.,
            metrics["mean_coverage"], metrics["mean_return"])


def sequence_indices(length, envs, sequence_length):
    if length % sequence_length:
        raise ValueError("Rollout length must be divisible by sequence length")
    return [(t, e) for e in range(envs) for t in range(0, length, sequence_length)]


def ppo_update(agent, optimizer, data, args, device):
    adv, returns = gae(data["rewards"], data["values"], data["next_values"],
                       data["terminated"], data["truncated"], args.gamma, args.gae_lambda)
    if args.normalize_adv:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    indices = sequence_indices(len(adv), adv.shape[1], args.rp_sequence)
    per_batch = max(1, args.batch_size // args.rp_sequence)
    updates, stop = 0, False
    losses, kls, clips = [], [], []
    for epoch in range(args.ppo_epochs):
        order = np.random.permutation(len(indices))
        for offset in range(0, len(indices), per_batch):
            selected = [indices[i] for i in order[offset:offset+per_batch]]
            def take(x):
                return torch.stack([x[t:t+args.rp_sequence, e] for t, e in selected], 1).to(device)
            obs = take(data["obs"])
            starts, latent = take(data["starts"]), take(data["latent"])
            state = tuple(torch.stack([data[key][t, e] for t, e in selected])[None].to(device)
                          for key in ("h", "c"))
            _, _, logs, entropy, values, _ = agent.sequence(obs, state, starts, latent)
            logratio = logs - take(data["logprob"])
            kl = (logratio.exp() - 1 - logratio).mean()
            if not torch.isfinite(kl):
                raise FloatingPointError("Nonfinite recurrent PPO KL")
            # Shared recurrent trunk: stop the joint optimizer when KL is high.
            if args.target_kl is not None and kl.item() > 1.5 * args.target_kl:
                stop = True
                break
            ratio, advantages = logratio.exp(), take(adv)
            policy_loss = -torch.minimum(ratio * advantages, ratio.clamp(1-args.clip, 1+args.clip)*advantages).mean()
            value_loss = .5 * (values - take(returns)).square().mean()
            loss = policy_loss + .5 * value_loss - args.entropy_coef * entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite recurrent PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(agent.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            losses.append((policy_loss.item(), value_loss.item(), entropy.mean().item()))
            kls.append(kl.item())
            clips.append(((ratio-1).abs() > args.clip).float().mean().item())
        if stop:
            break
    mean_losses = np.mean(losses, axis=0) if losses else [0., 0., 0.]
    return dict(policy_loss=float(mean_losses[0]), value_loss=float(mean_losses[1]),
                entropy=float(mean_losses[2]), approx_kl=float(np.mean(kls)) if kls else 0.,
                clip_fraction=float(np.mean(clips)) if clips else 0.,
                optimizer_updates=updates, kl_early_stop=int(stop))


RESTORE_KEYS = ("algo", "seed", "num_envs", "rollout", "batch_size", "actor_lr", "critic_lr",
                "gamma", "gae_lambda", "reward_scale", "normalize_adv", "ppo_epochs",
                "clip", "entropy_coef", "target_kl", "max_grad_norm", "completion_mode",
                "rp_observation", "rp_control", "rp_hidden", "rp_sequence", "rp_split_head",
                "rp_speed_scale", "rp_residual_scale", "rp_completion_bonus", "rp_offroad_penalty",
                "eval_seed_start", "eval_episodes")


def validate_restore(checkpoint, config):
    if checkpoint["config"].get("rp_finish_assist", False) != config.get("rp_finish_assist", False):
        raise ValueError("Resume config mismatch: rp_finish_assist")
    if checkpoint["config"].get("rp_controller", "v1") != config.get("rp_controller", "v1"):
        raise ValueError("Resume config mismatch: rp_controller")
    for key in RESTORE_KEYS:
        if checkpoint["config"].get(key) != config.get(key):
            raise ValueError(f"Resume config mismatch: {key}")
    if "optimizer" not in checkpoint:
        raise ValueError("Checkpoint is evaluation-only")


def validate_args(args, parser):
    for key in ("steps", "num_envs", "rollout", "batch_size", "rp_hidden", "rp_sequence",
                "eval_every", "eval_episodes", "ppo_epochs", "threads"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if args.rollout % args.rp_sequence or args.batch_size % args.rp_sequence:
        parser.error("rollout and batch-size must be multiples of rp-sequence")
    if args.steps % args.num_envs:
        parser.error("steps must be divisible by num-envs")
    if args.steps % (args.num_envs * args.rollout):
        parser.error("steps must be a multiple of num-envs * rollout (exact recurrent sequences)")
    if args.rp_control in ("residual", "adaptive") and args.rp_observation != "state":
        parser.error("residual control requires state observations")
    if args.rp_finish_assist and args.rp_observation != "state":
        parser.error("finish assistance requires state observations")
    if args.branch_from or args.grpo_c_warm_start:
        parser.error("Recurrent PPO supports --resume, not incompatible feedforward checkpoints")
    if args.seed < 0 or args.eval_seed_start < 0 or not 0 < args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        parser.error("Invalid seed/gamma/lambda")
    for key in ("actor_lr", "critic_lr", "reward_scale", "max_grad_norm", "clip",
                "rp_speed_scale", "rp_residual_scale"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    for key in ("entropy_coef", "rp_completion_bonus", "rp_offroad_penalty"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f"{key} must be finite and nonnegative")
    if args.target_kl is not None and (not math.isfinite(args.target_kl) or args.target_kl <= 0):
        parser.error("target-kl must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")


def train(args, parser):
    from .evaluate import evaluate
    validate_args(args, parser)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = vars(args).copy()
    env_config = dict(recurrent=True, observation_mode=args.rp_observation,
                      control_mode=args.rp_control, speed_scale=args.rp_speed_scale,
                      controller_profile=args.rp_controller,
                      finish_assist=args.rp_finish_assist,
                      residual_scale=args.rp_residual_scale, completion_mode=args.completion_mode)
    out = Path(args.out) / (args.name or f"recurrent-ppo_s{args.seed}")
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "config.json", config)
    save_json(out / "invocation.json", dict(argv=sys.argv, executable=sys.executable,
                                          cwd=str(Path.cwd()), output=str(out.resolve())))
    save_json(out / "provenance.json", provenance())
    agent = agent_from_config(config).to(args.device)
    critic_ids = {id(p) for p in agent.critic.parameters()}
    optimizer = torch.optim.Adam([
        {"params": [p for p in agent.parameters() if id(p) not in critic_ids], "lr": args.actor_lr},
        {"params": agent.critic.parameters(), "lr": args.critic_lr}], eps=1e-5)
    steps, frames, train_seconds, updates = 0, 0, 0., 0
    best_key = (-1., -1000., 0., -1e9)
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        validate_restore(ck, config)
        agent.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        steps, frames, train_seconds, updates = (ck[k] for k in ("steps", "frames", "train_seconds", "updates"))
        # A new segment selects and stores its own best; the experiment runner
        # compares all segments, preserving the ancestor best model.
        torch.set_rng_state(ck["torch_rng"].cpu())
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        if args.device == "cuda" and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        if args.steps <= steps:
            raise ValueError("Resume target must exceed checkpoint decisions")
    save_json(out / "model_info.json", dict(parameters=sum(p.numel() for p in agent.parameters()),
              hidden_size=args.rp_hidden, sequence_length=args.rp_sequence,
              privileged_state=args.rp_observation == "state", env_config=env_config))
    def checkpoint(name):
        payload = dict(format_version=1, model=agent.state_dict(), optimizer=optimizer.state_dict(),
                       config=config, env_config=env_config, steps=steps, frames=frames,
                       train_seconds=train_seconds, updates=updates, best_key=list(best_key),
                       torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                       python_rng=random.getstate(),
                       cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else None)
        temp = out / (name + ".tmp")
        torch.save(payload, temp)
        temp.replace(out / name)
    def validate():
        nonlocal best_key
        rows = evaluate(agent, args.device, env_config,
                        range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
        metrics = completion_summary(rows)
        flat = {k: v for k, v in metrics.items() if k != "completion_wilson95"}
        append_csv(out / "validation.csv", dict(steps=steps, frames=frames, train_seconds=train_seconds, **flat))
        for row in rows:
            append_csv(out / "validation_episodes.csv", dict(steps=steps, **row))
        key = selection_key(metrics)
        if key > best_key:
            best_key = key
            checkpoint("best.pt")
            checkpoint(f"milestone_{steps:09d}.pt")
            save_json(out / "best_metrics.json", dict(steps=steps, **metrics))
        checkpoint("last.pt")
        print(json.dumps(dict(event="validation", steps=steps, **metrics)), flush=True)
    vec = VectorRacing(args.num_envs, env_config, args.seed)
    try:
        validate()
        obs = vec.reset()
        state = agent.initial_state(args.num_envs)
        starts = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        next_eval = (steps // args.eval_every + 1) * args.eval_every
        while steps < args.steps:
            began = time.perf_counter()
            records = {k: [] for k in ("obs", "h", "c", "starts", "latent", "logprob",
                                      "values", "next_values", "rewards", "terminated", "truncated")}
            for t in range(args.rollout):
                tensor_obs = torch.as_tensor(obs, device=args.device)
                records["obs"].append(torch.as_tensor(obs).clone())
                for key, s in zip(("h", "c"), state):
                    records[key].append(s[0].detach().cpu())
                records["starts"].append(starts.cpu())
                with torch.no_grad():
                    actions, latent, logprob, _, values, new_state = agent.step(tensor_obs, state, starts)
                new_obs, reward, term, trunc, infos = vec.step(actions.cpu().numpy())
                # Bootstrap each final observation with h/c after consuming obs_t.
                bootstrap_obs = new_obs.copy()
                for i, info in enumerate(infos):
                    if term[i] or trunc[i]:
                        bootstrap_obs[i] = info["terminal_observation"]
                    if "episode_metrics" in info:
                        append_csv(out / "train_episodes.csv", dict(steps=steps+args.num_envs, **info["episode_metrics"]))
                    reward[i] = (reward[i] * args.reward_scale
                                 + args.rp_completion_bonus * int(info.get("episode_metrics", {}).get("completed", False))
                                 - args.rp_offroad_penalty * info["offroad_fraction_sum"])
                    frames += info["raw_frames"]
                with torch.no_grad():
                    next_values = agent.step(torch.as_tensor(bootstrap_obs, device=args.device), new_state,
                                             torch.zeros_like(starts))[4]
                for key, value in (("latent", latent), ("logprob", logprob), ("values", values),
                                   ("next_values", next_values), ("rewards", reward),
                                   ("terminated", term), ("truncated", trunc)):
                    records[key].append(torch.as_tensor(value).detach().cpu())
                obs, state = new_obs, tuple(s.detach() for s in new_state)
                starts = torch.as_tensor(term | trunc, device=args.device)
                steps += args.num_envs
            data = {k: torch.stack(v) for k, v in records.items()}
            metrics = ppo_update(agent, optimizer, data, args, args.device)
            updates += 1
            train_seconds += time.perf_counter() - began
            append_csv(out / "updates.csv", dict(steps=steps, frames=frames, update=updates,
                       train_seconds=train_seconds, decisions_per_second=args.rollout*args.num_envs/(time.perf_counter()-began), **metrics))
            if steps >= next_eval or steps == args.steps:
                validate()
                next_eval = (steps // args.eval_every + 1) * args.eval_every
        checkpoint("last.pt")
        save_json(out / "finished.json", dict(steps=steps, frames=frames, train_seconds=train_seconds,
                                            updates=updates, best_key=list(best_key)))
    except KeyboardInterrupt:
        checkpoint("interrupted.pt")
        raise
    finally:
        vec.close()


if __name__ == "__main__":
    from .train import main
    main()
