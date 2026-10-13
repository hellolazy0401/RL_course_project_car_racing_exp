"""Outcome-supervised GRPO adaptation: same-track full episodes, no critic.

Each trajectory has equal weight, independent of its length. No GAE or reward
shaping. Optional KL is exact Normal KL(current || frozen initial reference).
"""
from copy import deepcopy
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.distributions import kl_divergence
from torch.utils.tensorboard import SummaryWriter
from .envs import VectorRacing
from .evaluate import evaluate
from .model import Agent
from .utils import append_csv, save_json, provenance, summary


def group_advantages(scores):
    values = np.asarray(scores, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("GRPO needs at least two finite episode returns per group")
    return ((values-values.mean())/(values.std(ddof=0)+1e-8)).astype(np.float32)


def trajectory_weights(lengths):
    return np.concatenate([np.full(n, 1/(len(lengths)*n), np.float32) for n in lengths])


@torch.no_grad()
def collect_group(agent, vec, seed, size, device):
    obs = vec.reset_same_seed(seed)
    trajectories = [dict(obs=[], latent=[], logprob=[], score=0., metrics=None) for _ in range(size)]
    active = list(range(size))
    frames = 0
    while active:
        actions, latent, logs, _ = agent.action(torch.as_tensor(obs[active], device=device))
        results = vec.step_active(active, actions.cpu().numpy())
        remaining = []
        for j, (i, result) in enumerate(zip(active, results)):
            next_obs, reward, term, trunc, info = result
            trajectory = trajectories[i]
            trajectory["obs"].append(obs[i].copy())
            trajectory["latent"].append(latent[j].cpu().numpy())
            trajectory["logprob"].append(float(logs[j]))
            trajectory["score"] += float(reward)
            frames += info["raw_frames"]
            obs[i] = next_obs
            if term or trunc:
                trajectory["metrics"] = info["episode_metrics"]
            else:
                remaining.append(i)
        active = remaining
    advantages = group_advantages([t["score"] for t in trajectories])
    for trajectory, advantage in zip(trajectories, advantages):
        trajectory["advantage"] = float(advantage)
    return trajectories, frames


def update_grpo(agent, reference, optimizer, trajectories, args):
    obs = np.concatenate([np.asarray(t["obs"]) for t in trajectories])
    lengths = [len(t["obs"]) for t in trajectories]
    device = args.device
    latent = torch.as_tensor(np.concatenate([t["latent"] for t in trajectories]), device=device)
    old = torch.as_tensor(np.concatenate([t["logprob"] for t in trajectories]), dtype=torch.float32, device=device)
    adv = torch.as_tensor(np.concatenate([np.full(n,t["advantage"],np.float32) for n,t in zip(lengths,trajectories)]), device=device)
    weights = torch.as_tensor(trajectory_weights(lengths), device=device)
    updates, loss_value = 0, 0.
    # No global re-normalization: retain the within-track group-relative signal.
    if torch.any(adv != 0) or args.grpo_beta > 0 or args.entropy_coef != 0:
        for _ in range(args.grpo_epochs):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            # Microbatches accumulate ONE full objective gradient per epoch.
            for start in range(0, len(obs), args.batch_size):
                idx = slice(start,start+args.batch_size)
                x = torch.as_tensor(obs[idx],device=device)
                _,_,logp,entropy = agent.action(x,latent[idx])
                ratio = (logp-old[idx]).exp()
                objective = torch.minimum(ratio*adv[idx], ratio.clamp(1-args.clip,1+args.clip)*adv[idx])
                penalty = 0.
                if reference is not None:
                    with torch.no_grad():
                        ref_dist = reference.distribution(x)
                    penalty = kl_divergence(agent.distribution(x),ref_dist).sum(-1)
                loss = (weights[idx]*(-objective+args.grpo_beta*penalty-args.entropy_coef*entropy)).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite GRPO objective")
                loss.backward()
                loss_value += loss.item()
            torch.nn.utils.clip_grad_norm_(agent.actor_parameters(),args.max_grad_norm,error_if_nonfinite=True)
            optimizer.step()
            updates += 1
    approx, clipped, ref_kl = 0., 0., 0.
    with torch.no_grad():
        for start in range(0,len(obs),args.batch_size):
            idx=slice(start,start+args.batch_size);x=torch.as_tensor(obs[idx],device=device)
            delta=agent.action(x,latent[idx])[2]-old[idx]
            approx += float((weights[idx]*(delta.exp()-1-delta)).sum())
            clipped += float((weights[idx]*((delta.exp()-1).abs()>args.clip)).sum())
            if reference is not None:
                ref_kl += float((weights[idx]*kl_divergence(agent.distribution(x),reference.distribution(x)).sum(-1)).sum())
    return dict(policy_loss=loss_value,approx_kl=approx,clip_fraction=clipped,reference_kl=ref_kl,
                actor_updates=updates,critic_updates=0,zero_advantage_fraction=float((adv==0).float().mean()))


def train(args, parser):
    if args.branch_from or args.target_kl is not None:
        parser.error("GRPO does not use the PPO branch/target-kl experiment; --grpo-beta is a reference penalty")
    if args.grpo_group_size < 2 or args.num_envs != args.grpo_group_size:
        parser.error("GRPO requires --num-envs == --grpo-group-size >= 2")
    for key in ("steps","batch_size","threads","grpo_groups_per_update","grpo_epochs","eval_every","eval_episodes"):
        if getattr(args,key) <= 0:
            parser.error(f"{key} must be positive")
    if args.seed < 0 or args.eval_seed_start < 0 or not math.isfinite(args.grpo_beta) or args.grpo_beta < 0:
        parser.error("Invalid seed or grpo-beta")
    if not (math.isfinite(args.actor_lr) and args.actor_lr > 0 and 0 < args.clip < 1 and args.max_grad_norm > 0):
        parser.error("Invalid actor learning rate, clip or gradient limit")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    config=vars(args).copy()
    config["grpo_variant"]="same_track_outcome_equal_episode_v1"
    env_config=dict(repeat=4,stack=4,size=64,completion_mode=args.completion_mode)
    agent=Agent(with_critic=False).to(args.device)
    reference=deepcopy(agent).eval().requires_grad_(False) if args.grpo_beta > 0 else None
    optimizer=torch.optim.Adam(agent.actor_parameters(),lr=args.actor_lr,eps=1e-5)
    steps=frames=groups=0;train_seconds=0.
    if args.resume:
        ck=torch.load(args.resume,map_location=args.device,weights_only=False)
        keys=("algo","seed","num_envs","batch_size","actor_lr","clip","entropy_coef","max_grad_norm",
              "grpo_group_size","grpo_groups_per_update","grpo_epochs","grpo_beta","grpo_variant")
        for key in keys:
            if ck["config"].get(key)!=config[key]:
                raise ValueError(f"GRPO resume config mismatch: {key}")
        if ck["env_config"] != env_config or args.steps <= ck["steps"]:
            raise ValueError("GRPO resume requires same environment and a larger cumulative budget")
        agent.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["actor_optimizer"])
        if reference is not None:
            reference.load_state_dict(ck["reference_model"])
        steps,frames,groups,train_seconds=ck["steps"],ck["frames"],ck["groups"],ck["train_seconds"]
        random.setstate(ck["python_rng"]);np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if args.device=="cuda" and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in ck["cuda_rng"]])
    run=Path(args.out)/(args.name or f"grpo_s{args.seed}")
    run.mkdir(parents=True,exist_ok=False)
    save_json(run/"config.json",config);save_json(run/"provenance.json",provenance())
    writer=SummaryWriter(str(run/"tensorboard"))
    initial_steps,initial_time=steps,train_seconds
    best=-float("inf");eval_seconds=0.;wall=time.perf_counter()
    def checkpoint(name):
        state=dict(model=agent.state_dict(),reference_model=reference.state_dict() if reference is not None else None,
                   actor_optimizer=optimizer.state_dict(),config=config,env_config=env_config,
                   steps=steps,frames=frames,groups=groups,train_seconds=train_seconds,
                   python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if args.device=="cuda" else None)
        tmp=run/f"{name}.tmp";torch.save(state,tmp);tmp.replace(run/f"{name}.pt")
    def validate():
        nonlocal best,eval_seconds
        tick=time.perf_counter()
        rows=evaluate(agent,args.device,env_config,range(args.eval_seed_start,args.eval_seed_start+args.eval_episodes))
        sm=summary(rows);eval_seconds+=time.perf_counter()-tick
        append_csv(run/"validation.csv",dict(steps=steps,frames=frames,train_seconds=train_seconds,**sm))
        for row in rows:
            append_csv(run/"validation_episodes.csv",dict(steps=steps,**row))
        for key,value in sm.items():writer.add_scalar("validation/"+key,value,steps)
        if sm["mean_return"] > best:best=sm["mean_return"];checkpoint("best")
        checkpoint("last")
        print(f"GRPO validation steps={steps} return={sm['mean_return']:.2f}",flush=True)
    vec=VectorRacing(args.num_envs,env_config,args.seed)
    next_eval=(steps//args.eval_every+1)*args.eval_every
    try:
        validate()
        while steps < args.steps:
            tick=time.perf_counter();trajectories=[];before=steps;group_stds=[]
            for _ in range(args.grpo_groups_per_update):
                seed=int(np.random.randint(0,1000000))
                batch,used_frames=collect_group(agent,vec,seed,args.grpo_group_size,args.device)
                count=sum(len(t["obs"]) for t in batch)
                steps+=count;frames+=used_frames;groups+=1
                scores=[t["score"] for t in batch];group_stds.append(float(np.std(scores)))
                append_csv(run/"groups.csv",dict(group=groups,track_seed=seed,steps=steps,
                           mean_return=float(np.mean(scores)),std_return=group_stds[-1],decisions=count))
                for i,t in enumerate(batch):
                    append_csv(run/"train_episodes.csv",dict(steps=steps,group=groups,member=i,track_seed=seed,**t["metrics"]))
                trajectories.extend(batch)
                if steps>=args.steps:break  # complete groups; never label partial episodes as outcomes
            metrics=update_grpo(agent,reference,optimizer,trajectories,args)
            if args.device=="cuda":torch.cuda.synchronize()
            elapsed=time.perf_counter()-tick;train_seconds+=elapsed
            metrics.update(steps=steps,frames=frames,train_seconds=train_seconds,decision_sps=(steps-before)/elapsed,
                           group_reward_std=float(np.mean(group_stds)),
                           peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=="cuda" else 0)
            append_csv(run/"updates.csv",metrics)
            for key,value in metrics.items():writer.add_scalar("train/"+key,value,steps)
            print(f"GRPO decisions={steps}/{args.steps} sps={metrics['decision_sps']:.1f} KL={metrics['approx_kl']:.4f} group_std={metrics['group_reward_std']:.2f}",flush=True)
            if steps>=next_eval or steps>=args.steps:
                validate();next_eval=(steps//args.eval_every+1)*args.eval_every
        checkpoint("last")
        save_json(run/"finished.json",dict(steps=steps,requested_steps=args.steps,budget_overshoot=steps-args.steps,
                   frames=frames,train_seconds=train_seconds,initial_steps=initial_steps,added_steps=steps-initial_steps,
                   segment_train_seconds=train_seconds-initial_time,evaluation_seconds=eval_seconds,
                   session_wall_seconds=time.perf_counter()-wall,best_validation_return=best,
                   status="completed_full_group_budget_not_proof_of_convergence"))
    except KeyboardInterrupt:
        checkpoint("interrupted")
        raise
    finally:
        vec.close();writer.close()
