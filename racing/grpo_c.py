"""CarRacing adaptation of arXiv:2507.19555 (continuous GRPO).

One independently trained Gaussian policy per vector environment. Every policy
drives one complete episode on the same track before trajectory/state clustering
and a joint clipped update. The paper is a framework, not a published CarRacing
implementation; the pixel representation and differentiable diversity proxy are
specified here and logged as this project's variant.
"""
from copy import deepcopy
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from .envs import VectorRacing
from .evaluate import evaluate
from .model import Agent
from .utils import append_csv, provenance, save_json, summary


def kmeans_trajectories(features, clusters, iterations=12):
    """Deterministic k-means; fewer occupied clusters are allowed for ties."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not 1 <= clusters <= len(x):
        raise ValueError("Invalid trajectory features or cluster count")
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    centers = [x[0].copy()]
    for _ in range(1, clusters):
        distance = np.min(np.stack([((x-c)**2).sum(1) for c in centers]), axis=0)
        centers.append(x[int(np.argmax(distance))].copy())
    centers = np.stack(centers)
    for _ in range(iterations):
        labels = ((x[:, None] - centers[None])**2).sum(-1).argmin(1)
        revised = np.stack([x[labels == j].mean(0) if np.any(labels == j) else centers[j]
                            for j in range(clusters)])
        if np.allclose(revised, centers):
            break
        centers = revised
    return ((x[:, None] - centers[None])**2).sum(-1).argmin(1)


def dbscan_states(features, eps, min_samples):
    """Exact small-batch DBSCAN; -1 denotes noise, including borderless points."""
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or eps <= 0 or min_samples < 1:
        raise ValueError("Invalid DBSCAN settings")
    n = len(x)
    labels = np.full(n, -2, dtype=np.int32)  # unseen
    if n == 0:
        return labels
    # Row-wise distance calculation keeps peak memory small for long episodes.
    neighbors = [np.flatnonzero(((x-x[i])**2).sum(axis=1) <= eps*eps) for i in range(n)]
    cluster = 0
    for start in range(n):
        if labels[start] != -2:
            continue
        if len(neighbors[start]) < min_samples:
            labels[start] = -1
            continue
        labels[start] = cluster
        queue = list(map(int, neighbors[start]))
        queued = np.zeros(n, dtype=bool)
        queued[neighbors[start]] = True
        head = 0
        while head < len(queue):
            point = queue[head]
            head += 1
            if labels[point] == -1:
                labels[point] = cluster
            if labels[point] != -2:
                continue
            labels[point] = cluster
            if len(neighbors[point]) >= min_samples:
                additions = neighbors[point][~queued[neighbors[point]]]
                queued[additions] = True
                queue.extend(map(int, additions))
        cluster += 1
    return labels


def discounted_returns(rewards, gamma):
    output = np.empty(len(rewards), dtype=np.float32)
    running = 0.
    for i in range(len(rewards)-1, -1, -1):
        running = float(rewards[i]) + gamma*running
        output[i] = running
    return output


def visual_state_features(observations):
    # Shared ego-centred visual state proxy: mean of four grayscale frames,
    # area-resized to 8x8. No learned encoder or critic is introduced.
    return np.stack([cv2.resize(obs.astype(np.float32).mean(0), (8, 8),
                                interpolation=cv2.INTER_AREA).reshape(-1)/255.
                     for obs in observations])


def state_aware_advantages(trajectories, policy_clusters, gamma, eps, min_samples, base_clip):
    """Within trajectory-policy cluster: state RTG minus DBSCAN neighbourhood mean."""
    n = len(trajectories)
    raw = [None] * n
    noise_count = 0
    state_cluster_count = 0
    for group in np.unique(policy_clusters):
        members = np.flatnonzero(policy_clusters == group)
        lengths = [len(trajectories[i]["obs"]) for i in members]
        rtg = np.concatenate([discounted_returns(trajectories[i]["rewards"], gamma) for i in members])
        observations = np.concatenate([np.asarray(trajectories[i]["obs"]) for i in members])
        labels = dbscan_states(visual_state_features(observations), eps, min_samples)
        noise_count += int(np.sum(labels == -1))
        state_cluster_count += len(set(labels) - {-1})
        baseline = np.empty(len(rtg), dtype=np.float32)
        for label in np.unique(labels):
            index = labels == label
            # DBSCAN noise has no neighbourhood: use this policy group's mean.
            baseline[index] = (rtg.mean() if label == -1 else rtg[index].mean())
        offset = 0
        for member, length in zip(members, lengths):
            raw[member] = rtg[offset:offset+length] - baseline[offset:offset+length]
            offset += length
    global_std = float(np.std(np.concatenate(raw)))
    group_clips = {}
    for group in np.unique(policy_clusters):
        members = np.flatnonzero(policy_clusters == group)
        values = np.concatenate([raw[i] for i in members])
        mean, sigma = float(values.mean()), float(values.std())
        group_clips[int(group)] = base_clip * max(1., sigma/(global_std+1e-8))
        for member in members:
            trajectories[member]["advantage"] = (raw[member]-mean)/(sigma+1e-8)
            trajectories[member]["clip"] = group_clips[int(group)]
    return dict(state_clusters=state_cluster_count, state_noise_fraction=noise_count/sum(map(len, raw)),
                clip_mean=float(np.mean(list(group_clips.values()))),
                advantage_std=global_std)


@torch.no_grad()
def collect_ensemble(policies, vec, seed, device):
    observations = vec.reset_same_seed(seed)
    trajectories = [dict(obs=[], latent=[], logprob=[], rewards=[], score=0., metrics=None)
                    for _ in policies]
    active = list(range(len(policies)))
    frames = 0
    while active:
        actions = []
        latents = []
        logprobs = []
        for i in active:
            action, latent, logp, _ = policies[i].action(torch.as_tensor(observations[i:i+1], device=device))
            actions.append(action[0].cpu().numpy())
            latents.append(latent[0].cpu().numpy())
            logprobs.append(float(logp[0]))
        results = vec.step_active(active, actions)
        remaining = []
        for j, (i, result) in enumerate(zip(active, results)):
            next_obs, reward, terminated, truncated, info = result
            trajectory = trajectories[i]
            trajectory["obs"].append(observations[i].copy())
            trajectory["latent"].append(latents[j])
            trajectory["logprob"].append(logprobs[j])
            trajectory["rewards"].append(float(reward))
            trajectory["score"] += float(reward)
            frames += info["raw_frames"]
            observations[i] = next_obs
            if terminated or truncated:
                trajectory["metrics"] = info["episode_metrics"]
            else:
                remaining.append(i)
        active = remaining
    return trajectories, frames


@torch.no_grad()
def trajectory_features(policies, reference, trajectories, device):
    features = []
    for policy, trajectory in zip(policies, trajectories):
        sampled = np.asarray(trajectory["obs"])[::max(1, len(trajectory["obs"])//16)][:16]
        x = torch.as_tensor(sampled, device=device)
        entropy = float(policy.distribution(x).entropy().sum(-1).mean())
        action_variance = float((policy.distribution(x).stddev**2).sum(-1).mean())
        kl = float(kl_divergence(policy.distribution(x), reference.distribution(x)).sum(-1).mean())
        features.append([trajectory["score"]/len(trajectory["obs"]), entropy, action_variance, kl])
    return np.asarray(features, dtype=np.float32)


def update_grpo_c(policies, reference, optimizers, trajectories, clusters, args):
    device = args.device
    count = len(policies)
    total_loss = 0.
    for _ in range(args.grpo_c_epochs):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.
        for policy, trajectory in zip(policies, trajectories):
            observations = np.asarray(trajectory["obs"])
            old = torch.as_tensor(trajectory["logprob"], device=device, dtype=torch.float32)
            latent = torch.as_tensor(np.asarray(trajectory["latent"]), device=device)
            advantage = torch.as_tensor(trajectory["advantage"], device=device)
            length = len(observations)
            clip = trajectory["clip"]
            for start in range(0, length, args.batch_size):
                end = min(start+args.batch_size, length)
                x = torch.as_tensor(observations[start:end], device=device)
                _, _, logprob, entropy = policy.action(x, latent[start:end])
                ratio = (logprob-old[start:end]).exp()
                surrogate = torch.minimum(ratio*advantage[start:end],
                    ratio.clamp(1-clip, 1+clip)*advantage[start:end])
                loss = (-surrogate - args.entropy_coef*entropy).sum()/(count*length)
                if args.grpo_c_beta:
                    with torch.no_grad():
                        ref_dist = reference.distribution(x)
                    loss = loss + args.grpo_c_beta*kl_divergence(
                        policy.distribution(x), ref_dist).sum(-1).sum()/(count*length)
                if args.grpo_c_smooth and start < length-1:
                    # Include the next observation across microbatch boundaries.
                    a = policy.actor(x.float()/255.)
                    next_x = torch.as_tensor(observations[start+1:min(end+1,length)], device=device)
                    b = policy.actor(next_x.float()/255.)
                    loss = loss + args.grpo_c_smooth*torch.linalg.vector_norm(a[:len(b)]-b,dim=-1).sum()/(
                        count*max(length-1,1))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite GRPO-C loss")
                loss.backward()
                epoch_loss += float(loss.detach())
        if args.grpo_c_diversity and len(np.unique(clusters)) >= 2:
            means = []
            for group in np.unique(clusters):
                members = np.flatnonzero(clusters == group)
                outputs = [policies[i].actor(torch.as_tensor(
                    np.asarray(trajectories[i]["obs"][0:1]),device=device).float()/255.)[0]
                           for i in members]
                means.append(torch.stack(outputs).mean(0))
            penalties = [F.relu(F.cosine_similarity(means[i],means[j],dim=0,eps=1e-6)
                                  - args.grpo_c_diversity_threshold)
                         for i in range(len(means)) for j in range(i+1,len(means))]
            diversity = args.grpo_c_diversity*torch.stack(penalties).mean()
            diversity.backward()
            epoch_loss += float(diversity.detach())
        for policy, optimizer in zip(policies, optimizers):
            torch.nn.utils.clip_grad_norm_(policy.actor_parameters(),args.max_grad_norm,error_if_nonfinite=True)
            optimizer.step()
        total_loss = epoch_loss
    approx = clipped = reference_kl = 0.
    with torch.no_grad():
        for policy, trajectory in zip(policies, trajectories):
            observations = np.asarray(trajectory["obs"])
            old = torch.as_tensor(trajectory["logprob"], device=device)
            latent = torch.as_tensor(np.asarray(trajectory["latent"]), device=device)
            length = len(observations)
            for start in range(0,length,args.batch_size):
                end=min(start+args.batch_size,length)
                x=torch.as_tensor(observations[start:end],device=device)
                delta=policy.action(x,latent[start:end])[2]-old[start:end]
                weight=1/(count*length)
                approx+=float((delta.exp()-1-delta).sum())*weight
                clipped+=float(((delta.exp()-1).abs()>trajectory["clip"]).sum())*weight
                reference_kl+=float(kl_divergence(policy.distribution(x),reference.distribution(x)).sum())*weight
    return dict(policy_loss=total_loss,approx_kl=approx,clip_fraction=clipped,
                reference_kl=reference_kl,actor_updates=args.grpo_c_epochs*count,critic_updates=0,
                occupied_policy_clusters=len(np.unique(clusters)))


@torch.no_grad()
def update_reference(reference, policies, scores, tau):
    # Paper's top-policy reference mixture, instantiated as parameter averaging.
    top = np.argsort(scores)[-max(1,len(policies)//2):]
    for name, parameter in reference.named_parameters():
        average = torch.stack([dict(policies[i].named_parameters())[name].detach() for i in top]).mean(0)
        parameter.lerp_(average, tau)


def train(args, parser):
    if args.branch_from or args.target_kl is not None:
        parser.error("GRPO-C does not use PPO branch-from or target-kl")
    if args.num_envs < 2 or not 1 <= args.grpo_c_clusters <= args.num_envs:
        parser.error("GRPO-C requires num-envs >= 2 and 1 <= clusters <= num-envs")
    for key in ("steps","batch_size","threads","grpo_c_epochs","eval_every","eval_episodes",
                "grpo_c_state_min_samples"):
        if getattr(args,key) <= 0:
            parser.error(f"{key} must be positive")
    nonnegative=("grpo_c_beta","grpo_c_smooth","grpo_c_diversity")
    if any(not math.isfinite(getattr(args,k)) or getattr(args,k)<0 for k in nonnegative):
        parser.error("GRPO-C regularization weights must be finite and nonnegative")
    if not (0 < args.grpo_c_state_eps and math.isfinite(args.grpo_c_state_eps)
            and 0 <= args.grpo_c_diversity_threshold < 1
            and 0 < args.grpo_c_reference_tau <= 1
            and args.seed >= 0 and args.eval_seed_start >= 0
            and 0 < args.gamma <= 1 and math.isfinite(args.actor_lr) and args.actor_lr > 0
            and 0 < args.clip < 1 and args.max_grad_norm > 0):
        parser.error("Invalid GRPO-C clustering, reference, seed or optimizer settings")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    config=vars(args).copy()
    config["grpo_c_variant"]="pixel8_dbscan_multipolicy_v1"
    env_config=dict(repeat=4,stack=4,size=64,completion_mode=args.completion_mode)
    policies=[Agent(with_critic=False).to(args.device) for _ in range(args.num_envs)]
    reference=deepcopy(policies[0]).eval().requires_grad_(False)
    optimizers=[torch.optim.Adam(policy.actor_parameters(),lr=args.actor_lr,eps=1e-5)
                for policy in policies]
    steps=frames=iterations=0;train_seconds=0.
    if args.resume:
        ck=torch.load(args.resume,map_location=args.device,weights_only=False)
        keys=("algo","seed","num_envs","batch_size","actor_lr","clip","entropy_coef","gamma",
              "max_grad_norm","grpo_c_clusters","grpo_c_state_eps","grpo_c_state_min_samples",
              "grpo_c_epochs","grpo_c_beta","grpo_c_smooth","grpo_c_diversity",
              "grpo_c_diversity_threshold","grpo_c_reference_tau","grpo_c_variant")
        for key in keys:
            if ck["config"].get(key)!=config[key]:
                raise ValueError(f"GRPO-C resume config mismatch: {key}")
        if ck["env_config"]!=env_config or args.steps<=ck["steps"]:
            raise ValueError("GRPO-C resume requires same environment and larger cumulative budget")
        for policy,state in zip(policies,ck["ensemble"]):policy.load_state_dict(state)
        for optimizer,state in zip(optimizers,ck["optimizers"]):optimizer.load_state_dict(state)
        reference.load_state_dict(ck["reference_model"])
        steps,frames,iterations,train_seconds=(ck[k] for k in ("steps","frames","iterations","train_seconds"))
        random.setstate(ck["python_rng"]);np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if args.device=="cuda" and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
    run=Path(args.out)/(args.name or f"grpo-c_s{args.seed}")
    run.mkdir(parents=True,exist_ok=False)
    save_json(run/"config.json",config);save_json(run/"provenance.json",provenance())
    writer=SummaryWriter(str(run/"tensorboard"))
    initial_steps,initial_time=steps,train_seconds
    best=-float("inf");eval_seconds=0.;wall=time.perf_counter();selected=0

    def checkpoint(name):
        state=dict(model=policies[selected].state_dict(),ensemble=[p.state_dict() for p in policies],
                   optimizers=[o.state_dict() for o in optimizers],reference_model=reference.state_dict(),
                   selected_member=selected,config=config,env_config=env_config,steps=steps,
                   frames=frames,iterations=iterations,train_seconds=train_seconds,
                   python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if args.device=="cuda" else None)
        tmp=run/f"{name}.tmp";torch.save(state,tmp);tmp.replace(run/f"{name}.pt")

    def validate():
        nonlocal best,eval_seconds,selected
        tick=time.perf_counter();results=[]
        member_episodes=[]
        for member, policy in enumerate(policies):
            episodes=evaluate(policy,args.device,env_config,
                              range(args.eval_seed_start,args.eval_seed_start+args.eval_episodes))
            member_episodes.append(episodes)
            sm=summary(episodes);results.append(sm)
            append_csv(run/"member_validation.csv",dict(steps=steps,member=member,**sm))
            for episode in episodes:
                append_csv(run/"member_validation_episodes.csv",dict(steps=steps,member=member,**episode))
        eval_seconds+=time.perf_counter()-tick
        selected=int(np.argmax([r["mean_return"] for r in results]))
        chosen=results[selected]
        append_csv(run/"validation.csv",dict(steps=steps,frames=frames,train_seconds=train_seconds,
                    selected_member=selected,**chosen))
        for episode in member_episodes[selected]:
            append_csv(run/"validation_episodes.csv",dict(steps=steps,**episode))
        for key,value in chosen.items():writer.add_scalar("validation/"+key,value,steps)
        if chosen["mean_return"]>best:
            best=chosen["mean_return"];checkpoint("best")
        checkpoint("last")
        print(f"GRPO-C validation steps={steps} member={selected} return={chosen['mean_return']:.2f}",flush=True)

    vec=VectorRacing(args.num_envs,env_config,args.seed)
    next_eval=(steps//args.eval_every+1)*args.eval_every
    try:
        validate()
        while steps<args.steps:
            tick=time.perf_counter();seed=int(np.random.randint(0,1_000_000))
            trajectories,used_frames=collect_ensemble(policies,vec,seed,args.device)
            before=steps;steps+=sum(len(t["obs"]) for t in trajectories)
            frames+=used_frames;iterations+=1
            features=trajectory_features(policies,reference,trajectories,args.device)
            clusters=kmeans_trajectories(features,args.grpo_c_clusters)
            state_stats=state_aware_advantages(trajectories,clusters,args.gamma,
                args.grpo_c_state_eps,args.grpo_c_state_min_samples,args.clip)
            metrics=update_grpo_c(policies,reference,optimizers,trajectories,clusters,args)
            scores=[t["score"] for t in trajectories]
            update_reference(reference,policies,scores,args.grpo_c_reference_tau)
            for member,trajectory in enumerate(trajectories):
                append_csv(run/"train_episodes.csv",dict(steps=steps,iteration=iterations,
                           member=member,track_seed=seed,policy_cluster=int(clusters[member]),
                           **trajectory["metrics"]))
            if args.device=="cuda":torch.cuda.synchronize()
            elapsed=time.perf_counter()-tick;train_seconds+=elapsed
            metrics.update(**state_stats,steps=steps,frames=frames,train_seconds=train_seconds,
                           decision_sps=(steps-before)/elapsed,group_reward_std=float(np.std(scores)),
                           peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=="cuda" else 0)
            append_csv(run/"updates.csv",metrics)
            for key,value in metrics.items():writer.add_scalar("train/"+key,value,steps)
            print(f"GRPO-C decisions={steps}/{args.steps} sps={metrics['decision_sps']:.1f} "
                  f"clusters={metrics['occupied_policy_clusters']} KL={metrics['approx_kl']:.4f}",flush=True)
            if steps>=next_eval or steps>=args.steps:
                validate();next_eval=(steps//args.eval_every+1)*args.eval_every
        checkpoint("last")
        save_json(run/"finished.json",dict(steps=steps,requested_steps=args.steps,
                  budget_overshoot=steps-args.steps,frames=frames,train_seconds=train_seconds,
                  initial_steps=initial_steps,added_steps=steps-initial_steps,
                  segment_train_seconds=train_seconds-initial_time,evaluation_seconds=eval_seconds,
                  session_wall_seconds=time.perf_counter()-wall,best_validation_return=best,
                  selected_member=selected,status="completed_full_ensemble_budget_not_proof_of_convergence"))
    except KeyboardInterrupt:
        checkpoint("interrupted")
        raise
    finally:
        vec.close();writer.close()
