"""A2C-GAE, PPO-Clip and ICML-2025 VPG-repeat on one controlled protocol."""
import argparse
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from .envs import VectorRacing
from .evaluate import evaluate
from .model import Agent, gae
from .utils import append_csv, provenance, save_json, summary


def full_step(loss_fn, parameters, optimizer, n, micro, max_norm):
    """Accumulate the exact full-batch mean before ONE optimizer step."""
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for start in range(0, n, micro):
        idx = slice(start, min(start + micro, n))
        loss = loss_fn(idx) * (idx.stop - idx.start) / n
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        total += loss.detach().item()
    if max_norm > 0:
        clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    optimizer.step()
    return total


def update(agent, actor_opt, critic_opt, data, args, device):
    obs, latent, old_logprob, adv, returns = data
    n = len(obs)
    if args.normalize_adv:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    def inputs(idx):
        return torch.as_tensor(obs[idx], device=device)
    def critic_loss(idx):
        return 0.5 * (agent.value(inputs(idx)) - returns[idx]).square().mean()
    def actor_loss(idx):
        _, _, logp, entropy = agent.action(inputs(idx), latent[idx])
        return -(logp * adv[idx]).mean() - args.entropy_coef * entropy.mean()
    ap, cp = agent.actor_parameters(), list(agent.critic.parameters())
    target_kl = getattr(args, "target_kl", None)
    kl_stopped, stop_kl = False, 0.
    def batch_kl():
        with torch.no_grad():
            total = 0.
            for start in range(0, n, args.batch_size):
                idx = slice(start, min(start + args.batch_size, n))
                delta = agent.action(inputs(idx), latent[idx])[2] - old_logprob[idx]
                total += (delta.exp() - 1 - delta).sum().item()
            return total / n
    if args.algo in ("a2c", "vpg50"):
        k = 1 if args.algo == "a2c" else args.value_steps
        for _ in range(k):
            vl = full_step(critic_loss, cp, critic_opt, n, args.batch_size, args.critic_grad_norm)
        # Advantages/targets fixed from collection, as in authors' single-file implementation.
        pl = full_step(actor_loss, ap, actor_opt, n, args.batch_size, args.max_grad_norm)
        actor_updates, critic_updates = 1, k
    else:
        actor_updates = critic_updates = 0
        for _ in range(args.ppo_epochs):
            order = np.random.permutation(n)
            for start in range(0, n, args.batch_size):
                idx = order[start:start + args.batch_size]
                _, _, logp, entropy = agent.action(inputs(idx), latent[idx])
                ratio = (logp - old_logprob[idx]).exp()
                loss = -torch.minimum(ratio * adv[idx],
                    ratio.clamp(1-args.clip, 1+args.clip) * adv[idx]).mean() - args.entropy_coef * entropy.mean()
                # Stop only the independent actor; preserve critic update budget.
                # Full rollout KL is measured after each actor step. The step
                # crossing the threshold is retained: this is not a hard bound.
                if not kl_stopped:
                    actor_opt.zero_grad(set_to_none=True)
                    loss.backward()
                    clip_grad_norm_(ap, args.max_grad_norm, error_if_nonfinite=True)
                    actor_opt.step()
                    actor_updates += 1
                    if target_kl is not None:
                        measured_kl = batch_kl()
                        if not math.isfinite(measured_kl):
                            raise FloatingPointError("Non-finite policy KL")
                        if measured_kl > target_kl:
                            kl_stopped, stop_kl = True, measured_kl
                vloss = critic_loss(idx)
                if not torch.isfinite(vloss):
                    raise FloatingPointError("Non-finite critic loss")
                critic_opt.zero_grad(set_to_none=True)
                vloss.backward()
                if args.critic_grad_norm > 0:
                    clip_grad_norm_(cp, args.critic_grad_norm, error_if_nonfinite=True)
                critic_opt.step()
                pl, vl = loss.item(), vloss.item()
                critic_updates += 1
    with torch.no_grad():
        predictions, newlogs = [], []
        for start in range(0, n, args.batch_size):
            idx = slice(start, start + args.batch_size)
            predictions.append(agent.value(inputs(idx)))
            newlogs.append(agent.action(inputs(idx), latent[idx])[2])
        pred, logs = torch.cat(predictions), torch.cat(newlogs)
        logratio = logs - old_logprob
        explained = 1 - (returns-pred).var(unbiased=False) / (returns.var(unbiased=False) + 1e-8)
    return dict(policy_loss=pl, value_loss=vl, value_mse=float((returns-pred).square().mean()),
        explained_variance=float(explained), approx_kl=float((logratio.exp()-1-logratio).mean()),
        clip_fraction=float(((logratio.exp()-1).abs()>args.clip).float().mean()),
        actor_updates=actor_updates, critic_updates=critic_updates,
        kl_early_stop=int(kl_stopped), stop_kl=stop_kl)


def validate_restore(ck, config, branch=False):
    if ck.get("env_config", {}).get("completion_mode", "original") != config.get("completion_mode", "original"):
        raise ValueError("Restore config mismatch: completion_mode; start a new run for a changed environment")
    keys = ("algo", "seed", "num_envs", "rollout", "actor_lr", "critic_lr", "gamma", "gae_lambda",
            "reward_scale", "normalize_adv", "value_steps", "ppo_epochs", "clip", "entropy_coef",
            "batch_size", "max_grad_norm", "critic_grad_norm")
    for key in keys:
        if ck["config"][key] != config[key]:
            raise ValueError(f"Restore config mismatch: {key}")
    if not branch and ck["config"].get("target_kl") != config.get("target_kl"):
        raise ValueError("Resume config mismatch: target_kl; use --branch-from for a KL ablation")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["a2c", "ppo", "vpg50", "grpo"], required=True)
    p.add_argument("--grpo-group-size", type=int, default=4)
    p.add_argument("--grpo-groups-per-update", type=int, default=2)
    p.add_argument("--grpo-epochs", type=int, default=4)
    p.add_argument("--grpo-beta", type=float, default=0., help="GRPO frozen initial-reference KL coefficient; 0 disables reference KL")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000, help="Aggregate policy decisions, NOT raw frames")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--gae-lambda", type=float, default=.95)
    p.add_argument("--reward-scale", type=float, default=.1)
    p.add_argument("--entropy-coef", type=float, default=0.)
    p.add_argument("--normalize-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--value-steps", type=int, default=50)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--target-kl", type=float, default=None, help="PPO actor-only early stopping; omitted disables it")
    p.add_argument("--completion-mode", choices=["original", "lap95"], default="original")
    p.add_argument("--max-grad-norm", type=float, default=1.)
    p.add_argument("--critic-grad-norm", type=float, default=0., help="0 disables critic clipping")
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-seed-start", type=int, default=1_000_000)
    p.add_argument("--out", default="runs")
    p.add_argument("--name")
    restore = p.add_mutually_exclusive_group()
    restore.add_argument("--resume", help="Restore weights/optimizers/counters; environment starts new episodes")
    restore.add_argument("--branch-from", help="Restore full state for a KL-only ablation; evaluate/save starting model")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.algo == "grpo":
        from .grpo import train
        return train(args, p)
    for key in ("steps", "num_envs", "rollout", "batch_size", "threads", "value_steps", "ppo_epochs", "eval_every", "eval_episodes"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if args.seed < 0 or not (0 < args.gamma <= 1 and 0 <= args.gae_lambda <= 1):
        p.error("Invalid seed/gamma/GAE lambda")
    if args.eval_seed_start < 0:
        p.error("eval-seed-start must be nonnegative")
    if args.target_kl is not None and (not math.isfinite(args.target_kl) or args.target_kl <= 0 or args.algo != "ppo"):
        p.error("target-kl must be finite, positive, and used with PPO")
    if args.branch_from and args.algo != "ppo":
        p.error("branch-from currently supports PPO KL ablations only")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA unavailable. Install matching PyTorch or explicitly use --device cpu")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = vars(args).copy()
    restore_path = args.resume or args.branch_from
    ck = torch.load(restore_path, map_location=args.device, weights_only=False) if restore_path else None
    if ck is not None:
        validate_restore(ck, config, branch=bool(args.branch_from))
        if args.steps <= ck["steps"]:
            p.error("--steps must exceed restored steps")
    env_config = dict(repeat=4, stack=4, size=64, completion_mode=args.completion_mode)
    run = Path(args.out) / (args.name or f"{args.algo}_s{args.seed}")
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "config.json", config)
    save_json(run / "provenance.json", provenance())
    writer = SummaryWriter(str(run / "tensorboard"))
    agent = Agent().to(args.device)
    ao = torch.optim.Adam(agent.actor_parameters(), lr=args.actor_lr, eps=1e-5)
    co = torch.optim.Adam(agent.critic.parameters(), lr=args.critic_lr, eps=1e-5)
    steps, frames, train_seconds, eval_seconds, best = 0, 0, 0., 0., -float("inf")
    if ck is not None:
        agent.load_state_dict(ck["model"])
        ao.load_state_dict(ck["actor_optimizer"])
        co.load_state_dict(ck["critic_optimizer"])
        steps, frames, train_seconds = ck["steps"], ck["frames"], ck["train_seconds"]
        # Do not reuse parent best score: this output directory needs its own best.pt.
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if args.device == "cuda" and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        if args.steps <= steps:
            raise ValueError("--steps must exceed restored steps")
    def checkpoint(name):
        state = dict(model=agent.state_dict(), actor_optimizer=ao.state_dict(), critic_optimizer=co.state_dict(),
            steps=steps, frames=frames, train_seconds=train_seconds, config=config, env_config=env_config,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else None)
        tmp = run / f"{name}.tmp"
        torch.save(state, tmp)
        tmp.replace(run / f"{name}.pt")
    initial_steps, initial_train_seconds = steps, train_seconds
    vec = VectorRacing(args.num_envs, env_config, args.seed + (steps if restore_path else 0))
    wall_start = time.perf_counter()
    next_eval = (steps // args.eval_every + 1) * args.eval_every
    try:
        obs = vec.reset()
        # Baseline evaluation at step zero (also makes learning-curve AUC meaningful).
        if steps == 0 or args.branch_from:
            et = time.perf_counter()
            rows = evaluate(agent, args.device, env_config, range(args.eval_seed_start, args.eval_seed_start+args.eval_episodes))
            sm = summary(rows)
            eval_seconds += time.perf_counter()-et
            append_csv(run / "validation.csv", dict(steps=steps, frames=frames, train_seconds=train_seconds, **sm))
            for row in rows:
                append_csv(run / "validation_episodes.csv", dict(steps=steps, **row))
            if args.branch_from:
                best = sm["mean_return"]
                checkpoint("best")
                checkpoint("last")
                save_json(run / "branch.json", dict(parent=str(Path(args.branch_from).resolve()),
                    initial_steps=steps, target_kl=args.target_kl, initial_validation=sm,
                    restored_optimizer=True, environment_reset=True))
            print(f"initial validation return={sm['mean_return']:.1f}", flush=True)
        while steps < args.steps:
            tick = time.perf_counter()
            length = min(args.rollout, math.ceil((args.steps-steps)/args.num_envs))
            ob = np.empty((length, args.num_envs, 4, 64, 64), dtype=np.uint8)
            lat, logs, rewards, vals, nextvals, terms, truncs = [], [], [], [], [], [], []
            for t in range(length):
                ob[t] = obs
                with torch.no_grad():
                    x = torch.as_tensor(obs, device=args.device)
                    action, latent, lp, _ = agent.action(x)
                    value = agent.value(x)
                newobs, reward, term, trunc, infos = vec.step(action.cpu().numpy())
                finalobs = np.stack([i.get("terminal_observation", o) for i, o in zip(infos, newobs)])
                with torch.no_grad():
                    nv = agent.value(torch.as_tensor(finalobs, device=args.device))
                steps += args.num_envs
                frames += sum(i["raw_frames"] for i in infos)
                for i in infos:
                    if "episode_metrics" in i:
                        append_csv(run / "train_episodes.csv", dict(steps=steps, **i["episode_metrics"]))
                lat.append(latent); logs.append(lp); vals.append(value); nextvals.append(nv)
                rewards.append(reward * args.reward_scale); terms.append(term); truncs.append(trunc)
                obs = newobs
            def tensor(a, dtype=None):
                return torch.as_tensor(np.array(a), device=args.device, dtype=dtype)
            adv, returns = gae(tensor(rewards, torch.float32), torch.stack(vals), torch.stack(nextvals),
                tensor(terms, torch.bool), tensor(truncs, torch.bool), args.gamma, args.gae_lambda)
            metrics = update(agent, ao, co, (ob.reshape(-1,4,64,64), torch.stack(lat).reshape(-1,3),
                torch.stack(logs).flatten(), adv.flatten(), returns.flatten()), args, args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter()-tick
            train_seconds += elapsed
            metrics.update(steps=steps, frames=frames, train_seconds=train_seconds,
                decision_sps=length*args.num_envs/elapsed,
                peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=="cuda" else 0)
            append_csv(run / "updates.csv", metrics)
            for key, val in metrics.items():
                writer.add_scalar("train/"+key, val, steps)
            print(f"{args.algo} seed={args.seed} decisions={steps}/{args.steps} "
                  f"sps={metrics['decision_sps']:.1f} value_mse={metrics['value_mse']:.4f} "
                  f"kl={metrics['approx_kl']:.4f} actor_updates={metrics['actor_updates']} "
                  f"kl_stop={metrics['kl_early_stop']}", flush=True)
            if steps >= next_eval or steps >= args.steps:
                et = time.perf_counter()
                rows = evaluate(agent, args.device, env_config, range(args.eval_seed_start,args.eval_seed_start+args.eval_episodes))
                sm = summary(rows)
                eval_seconds += time.perf_counter()-et
                append_csv(run / "validation.csv", dict(steps=steps, frames=frames, train_seconds=train_seconds, **sm))
                for row in rows:
                    append_csv(run / "validation_episodes.csv", dict(steps=steps, **row))
                for key, val in sm.items():
                    writer.add_scalar("validation/"+key, val, steps)
                if sm["mean_return"] > best:
                    best = sm["mean_return"]
                    checkpoint("best")
                checkpoint("last")
                next_eval = (steps // args.eval_every+1)*args.eval_every
                print(f"validation return={sm['mean_return']:.1f} completion={sm['completion_rate']:.0%}", flush=True)
        checkpoint("last")
        save_json(run / "finished.json", dict(steps=steps, frames=frames, train_seconds=train_seconds,
            initial_steps=initial_steps, added_steps=steps-initial_steps,
            segment_train_seconds=train_seconds-initial_train_seconds,
            evaluation_seconds=eval_seconds, session_wall_seconds=time.perf_counter()-wall_start,
            best_validation_return=best, status="completed_budget_not_proof_of_convergence"))
    except KeyboardInterrupt:
        checkpoint("interrupted")
        print("Saved interrupted.pt. Resume starts fresh episodes; see README.", flush=True)
        raise
    finally:
        vec.close()
        writer.close()


if __name__ == "__main__":
    main()
