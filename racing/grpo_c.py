"""CarRacing adaptation of arXiv:2507.19555 (continuous GRPO).

One independently trained Gaussian policy per vector environment. Every policy
drives one complete episode on the same track before trajectory/state clustering
and a joint clipped update. The paper is a framework, not a published CarRacing
implementation; the pixel representation and differentiable diversity proxy are
specified here and logged as this project's variant.
"""
import math
import hashlib
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .envs import VectorRacing
from .evaluate import evaluate, evaluate_ensemble
from .model import Agent
from .grpo_c_reference import ReferenceMixture
from .utils import append_csv, provenance, save_json, summary


def kmeans_trajectories(features, clusters, iterations=12, standardize=True):
    """Deterministic k-means; fewer occupied clusters are allowed for ties."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not 1 <= clusters <= len(x):
        raise ValueError("Invalid trajectory features or cluster count")
    if standardize:
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


def policy_groups(features, requested):
    """Merge singleton groups: one trajectory cannot form a relative comparison."""
    labels = kmeans_trajectories(features, requested)
    x = (features-features.mean(0))/(features.std(0)+1e-8)
    merges = 0
    while len(np.unique(labels)) > 1:
        groups, counts = np.unique(labels, return_counts=True)
        small = groups[counts < 2]
        if not len(small):
            break
        source = small[0]
        others = groups[groups != source]
        center = x[labels == source].mean(0)
        target = min(others, key=lambda g: float(np.square(center-x[labels == g].mean(0)).sum()))
        labels[labels == source] = target
        merges += 1
    return np.unique(labels, return_inverse=True)[1], merges


@torch.no_grad()
def initialize_policy(policy, mode):
    if mode == "drive":
        # Keep all three continuous action dimensions and the original bounds.
        # Low initial brake is an explicit prior, NOT reward shaping or a rule
        # overriding the network. Validation at step zero measures its benefit.
        policy.actor[-1].bias.copy_(torch.tensor([0., 0., math.atanh(2*.02-1)],
                                                 device=policy.logstd.device))
    elif mode != "standard":
        raise ValueError(f"Unknown GRPO-C initialization: {mode}")


def warm_start_policies(policies, path, env_config, allow_completion_transfer=False, allow_observation_transfer=False):
    """Transfer the PPO actor only. No critic, optimizer or PPO counters survive."""
    path = Path(path).resolve()
    ck = torch.load(path, map_location=next(policies[0].parameters()).device, weights_only=False)
    if ck["config"]["algo"] != "ppo":
        raise ValueError("GRPO-C warm start requires a PPO checkpoint")
    source_env = dict(ck["env_config"])
    target_env = dict(env_config)
    source_env.setdefault("completion_mode", "original")
    target_env.setdefault("completion_mode", "original")
    transfer = source_env["completion_mode"] != target_env["completion_mode"]
    if source_env.get("road_state", False):
        raise ValueError("Warm-start environment mismatch: source must be pixel PPO")
    observation_transfer = not source_env.get("road_state", False) and target_env.get("road_state", False)
    if observation_transfer and not allow_observation_transfer:
        raise ValueError("Observation transfer must be explicit")
    source_physics = {k:v for k,v in source_env.items() if k not in ("completion_mode", "road_state")}
    target_physics = {k:v for k,v in target_env.items() if k not in ("completion_mode", "road_state")}
    if (source_physics != target_physics or (transfer and not allow_completion_transfer)
            or source_env["completion_mode"] not in ("original", "lap95")):
        raise ValueError("Warm-start environment mismatch: observations/physics must match; completion transfer must be explicit")
    actor = {k: v for k, v in ck["model"].items() if k == "logstd" or k.startswith("actor.")}
    for policy in policies:
        state = dict(actor)
        if observation_transfer:
            if policy.road_head is None:
                raise ValueError("Target policy requires road state head")
            for key, value in policy.state_dict().items():
                if key.startswith("road_head."):
                    state[key] = torch.zeros_like(value)
        policy.load_state_dict(state, strict=True)
    return dict(method="ppo_pretraining_then_grpo_c", checkpoint=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_algo="ppo", source_steps=ck["steps"], source_frames=ck.get("frames"),
                source_train_seconds=ck.get("train_seconds"), source_env_config=source_env,
                target_env_config=target_env, completion_protocol_transfer=transfer, observation_protocol_transfer=bool(observation_transfer))


def validation_rank(result, criterion):
    if criterion == "completion":
        return (result["completion_rate"], result["mean_coverage"], result["mean_return"])
    if criterion == "return":
        return (result["mean_return"],)
    raise ValueError(f"Unknown validation criterion: {criterion}")


def configure_trainable(policy, mode):
    if mode == "full":
        policy.requires_grad_(True)
    elif mode == "road-head":
        if policy.road_head is None:
            raise ValueError("road-head requires road-state observations")
        policy.requires_grad_(False)
        policy.road_head.requires_grad_(True)
    elif mode == "head":
        policy.requires_grad_(False)
        policy.actor[-1].requires_grad_(True)
    elif mode == "gas-bias":
        policy.requires_grad_(False)
        policy.actor[-1].bias.requires_grad_(True)
        policy.actor[-1].bias.register_hook(lambda grad: grad * grad.new_tensor([0., 1., 0.]))
    else:
        raise ValueError(f"Unknown trainable mode: {mode}")


def completion_reward(reward, terminated, metrics, bonus):
    # Coverage alone and timeouts never earn this bonus. Environment metrics
    # remain native so training and validation CSVs are still interpretable.
    return float(reward) + (bonus if terminated and metrics and metrics["completed"] else 0.)


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
    if np.asarray(observations).shape[1] == 5:
        return (np.asarray(observations)[:, 4].reshape(len(observations), -1)[:, :16].astype(np.float32)-128)/127
    # Retain current view AND motion. Averaging all frames erased motion in v1.
    current = np.stack([cv2.resize(obs[-1].astype(np.float32), (8, 8),
                        interpolation=cv2.INTER_AREA).reshape(-1)/255. for obs in observations])
    oldest = np.stack([cv2.resize(obs[0].astype(np.float32), (8, 8),
                       interpolation=cv2.INTER_AREA).reshape(-1)/255. for obs in observations])
    return np.concatenate([current, current-oldest], axis=1)


def cluster_states(features, eps, min_samples, fallback_clusters):
    # RMS-scaled visual distances; floor prevents static pixels amplifying noise.
    x = (features-features.mean(0))/np.maximum(features.std(0), .05)
    x = x/np.sqrt(x.shape[1])
    labels = dbscan_states(x, eps, min_samples)
    _, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest = float(counts.max()/len(labels)) if len(counts) else 0.
    noise = float(np.mean(labels == -1))
    fallback = fallback_clusters > 0 and (largest > .9 or noise > .5)
    if fallback:
        # Bounded Voronoi groups break DBSCAN's density chains. Identical views
        # may still occupy one cluster; never manufacture distinctions by index.
        labels = kmeans_trajectories(x, min(fallback_clusters, len(x)), standardize=False)
    return labels, dict(state_dbscan_largest_fraction=largest,
                        state_dbscan_noise_fraction=noise, state_fallback=int(fallback))


def state_aware_advantages(trajectories, policy_clusters, gamma, eps, min_samples, base_clip,
                           fallback_clusters=16):
    """Global state baseline, then policy-group normalization (paper Algorithm 1).

    Baselines exclude the scored trajectory to avoid subtracting its own reward
    signal. Missing peers fall back to other trajectories at a similar horizon.
    """
    n = len(trajectories)
    lengths = [len(t["obs"]) for t in trajectories]
    rtg = np.concatenate([discounted_returns(t["rewards"], gamma) for t in trajectories])
    owners = np.repeat(np.arange(n), lengths)
    times = np.concatenate([np.arange(length) for length in lengths])
    observations = np.concatenate([np.asarray(t["obs"]) for t in trajectories])
    labels, stats = cluster_states(visual_state_features(observations), eps, min_samples, fallback_clusters)
    baseline = np.zeros(len(rtg), dtype=np.float32)
    matched = 0
    for member in range(n):
        own = owners == member
        for label in np.unique(labels[own]):
            for time_bin in np.unique(times[own & (labels == label)]//16):
                target = own & (labels == label) & (times//16 == time_bin)
                other = (~own) & (times//16 == time_bin)
                peers = other & (labels == label)
                if label >= 0 and peers.any():
                    baseline[target] = rtg[peers].mean()
                    matched += int(target.sum())
                else:
                    baseline[target] = rtg[other].mean() if other.any() else 0.
    raw = np.split(rtg-baseline, np.cumsum(lengths)[:-1])
    global_std = float(np.std(np.concatenate(raw)))
    group_clips = {}
    for group in np.unique(policy_clusters):
        members = np.flatnonzero(policy_clusters == group)
        values = np.concatenate([raw[i] for i in members])
        mean, sigma = float(values.mean()), float(values.std())
        group_clips[int(group)] = base_clip * max(1., sigma/(global_std+1e-8))
        for member in members:
            trajectories[member]["raw_advantage"] = raw[member]
            trajectories[member]["advantage"] = (raw[member]-mean)/(sigma+1e-8)
            trajectories[member]["clip"] = group_clips[int(group)]
    return dict(**stats, state_clusters=len(set(labels)-{-1}),
                state_noise_fraction=float(np.mean(labels == -1)), state_peer_fraction=matched/len(rtg),
                clip_mean=float(np.mean(list(group_clips.values()))),
                advantage_std=global_std)


@torch.no_grad()
def collect_ensemble(policies, vec, seed, device, completion_bonus=0., offroad_penalty=0.):
    observations = vec.reset_same_seed(seed)
    trajectories = [dict(obs=[], latent=[], logprob=[], rewards=[], actions=[], score=0., metrics=None)
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
            reward = completion_reward(reward, terminated, info.get("episode_metrics"), completion_bonus) - offroad_penalty*info.get("offroad_fraction_sum", 0.)
            trajectory = trajectories[i]
            trajectory["obs"].append(observations[i].copy())
            trajectory["latent"].append(latents[j])
            trajectory["logprob"].append(logprobs[j])
            trajectory["rewards"].append(float(reward))
            trajectory["actions"].append(actions[j])
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
        kl = float(reference.kl(policy.distribution(x), x).mean())
        features.append([trajectory["score"]/len(trajectory["obs"]), entropy, action_variance, kl])
    return np.asarray(features, dtype=np.float32)


def merge_episode_batches(batches):
    """Concatenate scored episodes only after computing per-episode returns.

    Preserve per-sample clipping and mask reset boundaries for smoothness.
    A policy stays frozen throughout collection of all batches.
    """
    merged = []
    for episodes in zip(*batches):
        item = {key: np.concatenate([np.asarray(t[key]) for t in episodes])
                for key in ("obs", "latent", "logprob", "advantage")}
        item["clip"] = np.concatenate([np.full(len(t["obs"]), t["clip"], np.float32)
                                       for t in episodes])
        item["smooth_mask"] = np.concatenate([
            np.r_[np.ones(len(t["obs"])-1, dtype=bool), False] for t in episodes])
        merged.append(item)
    return merged


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
            clips = torch.as_tensor(trajectory["clip"], device=device)
            smooth_mask = torch.as_tensor(trajectory.get("smooth_mask", np.r_[
                np.ones(length-1, dtype=bool), False]), device=device)
            for start in range(0, length, args.batch_size):
                end = min(start+args.batch_size, length)
                x = torch.as_tensor(observations[start:end], device=device)
                _, _, logprob, entropy = policy.action(x, latent[start:end])
                ratio = (logprob-old[start:end]).exp()
                clip = clips if clips.ndim == 0 else clips[start:end]
                surrogate = torch.minimum(ratio*advantage[start:end],
                    ratio.clamp(1-clip, 1+clip)*advantage[start:end])
                loss = (-surrogate - args.entropy_coef*entropy).sum()/(count*length)
                if args.grpo_c_beta:
                    loss = loss + args.grpo_c_beta*reference.kl(
                        policy.distribution(x), x).sum()/(count*length)
                if args.grpo_c_smooth and start < length-1:
                    # Include the next observation across microbatch boundaries.
                    a = policy.distribution(x).loc
                    next_x = torch.as_tensor(observations[start+1:min(end+1,length)], device=device)
                    b = policy.distribution(next_x).loc
                    distances = torch.linalg.vector_norm(a[:len(b)]-b,dim=-1)
                    loss = loss + args.grpo_c_smooth*(distances*smooth_mask[start:start+len(b)]).sum()/(
                        count*smooth_mask.sum().clamp_min(1))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite GRPO-C loss")
                loss.backward()
                epoch_loss += float(loss.detach())
        if args.grpo_c_diversity and len(np.unique(clusters)) >= 2:
            means = []
            for group in np.unique(clusters):
                members = np.flatnonzero(clusters == group)
                outputs = [policies[i].distribution(torch.as_tensor(
                    np.asarray(trajectories[i]["obs"][0:1]),device=device)).loc[0]
                           for i in members]
                means.append(torch.stack(outputs).mean(0))
            # Smooth norm floor avoids a 1/||mean|| gradient singularity near
            # the original near-zero actor initialization.
            normalized = [mean/torch.sqrt(mean.square().sum()+1.) for mean in means]
            penalties = [F.relu(torch.dot(normalized[i],normalized[j])
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
                clips=torch.as_tensor(trajectory["clip"],device=device)
                clip=clips if clips.ndim==0 else clips[start:end]
                clipped+=float(((delta.exp()-1).abs()>clip).sum())*weight
                reference_kl+=float(reference.kl(policy.distribution(x),x).sum())*weight
    return dict(policy_loss=total_loss,approx_kl=approx,clip_fraction=clipped,
                reference_kl=reference_kl,actor_updates=args.grpo_c_epochs*count,critic_updates=0,
                occupied_policy_clusters=len(np.unique(clusters)))


@torch.no_grad()
def update_reference(reference, policies, scores, tau):
    reference.update(policies, scores, tau)


def train(args, parser):
    if args.grpo_c_trainable != "full" and not (args.grpo_c_warm_start or args.resume):
        parser.error("Restricted fine-tuning requires a pretrained warm start or matching resume")
    if args.grpo_c_trainable == "road-head" and not args.grpo_c_road_state:
        parser.error("road-head requires --grpo-c-road-state")
    if not math.isfinite(args.grpo_c_road_scale) or args.grpo_c_road_scale <= 0:
        parser.error("road-scale must be finite and positive")
    if args.grpo_c_offroad_penalty and not args.grpo_c_road_state:
        parser.error("offroad penalty belongs to the explicit road-state protocol")
    if args.branch_from or args.target_kl is not None:
        parser.error("GRPO-C does not use PPO branch-from or target-kl")
    if args.grpo_c_state_fallback_clusters < 0:
        parser.error("state-fallback-clusters must be nonnegative")
    if args.num_envs < 2 or not 1 <= args.grpo_c_clusters <= args.num_envs:
        parser.error("GRPO-C requires num-envs >= 2 and 1 <= clusters <= num-envs")
    for key in ("steps","batch_size","threads","grpo_c_epochs","eval_every","eval_episodes",
                "grpo_c_state_min_samples", "grpo_c_episodes_per_update"):
        if getattr(args,key) <= 0:
            parser.error(f"{key} must be positive")
    nonnegative=("grpo_c_beta","grpo_c_smooth","grpo_c_diversity","grpo_c_completion_bonus","grpo_c_offroad_penalty")
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
    config["grpo_c_variant"]="pixel_road_state_global_loo_mixture_v3" if args.grpo_c_road_state else "pixel_motion_global_loo_mixture_v2"
    env_config=dict(repeat=4,stack=4,size=64,completion_mode=args.completion_mode)
    if args.grpo_c_road_state:
        env_config["road_state"] = True
    policies=[Agent(with_critic=False, road_state=args.grpo_c_road_state, road_scale=args.grpo_c_road_scale).to(args.device) for _ in range(args.num_envs)]
    lineage = None
    if args.grpo_c_warm_start:
        lineage = warm_start_policies(policies, args.grpo_c_warm_start, env_config,
                                      args.grpo_c_allow_completion_transfer, args.grpo_c_road_state)
    else:
        for policy in policies:
            initialize_policy(policy, args.grpo_c_init)
    for policy in policies:
        configure_trainable(policy, args.grpo_c_trainable)
    reference=ReferenceMixture(policies).to(args.device)
    optimizers=[torch.optim.Adam(policy.actor_parameters(),lr=args.actor_lr,eps=1e-5)
                for policy in policies]
    steps=frames=iterations=0;train_seconds=0.
    if args.resume:
        ck=torch.load(args.resume,map_location=args.device,weights_only=False)
        if ck.get("inference_only"):
            raise ValueError("Inference-only checkpoint: resume its source training checkpoint instead")
        if ck["config"].get("grpo_c_variant") != config["grpo_c_variant"]:
            raise ValueError("GRPO-C v2 changes the estimator/reference: start a new run. Old checkpoints remain evaluable.")
        keys=("algo","seed","num_envs","batch_size","actor_lr","clip","entropy_coef","gamma",
              "max_grad_norm","grpo_c_clusters","grpo_c_state_eps","grpo_c_state_min_samples",
              "grpo_c_epochs","grpo_c_beta","grpo_c_smooth","grpo_c_diversity",
              "grpo_c_diversity_threshold","grpo_c_reference_tau","grpo_c_variant",
              "grpo_c_init","grpo_c_state_fallback_clusters")
        for key in keys:
            if ck["config"].get(key)!=config[key]:
                raise ValueError(f"GRPO-C resume config mismatch: {key}")
        if ck["config"].get("grpo_c_episodes_per_update", 1) != args.grpo_c_episodes_per_update:
            raise ValueError("GRPO-C resume config mismatch: grpo_c_episodes_per_update; start a new run")
        for key, default in (("grpo_c_selection", "return"), ("grpo_c_completion_bonus", 0.),
                             ("grpo_c_trainable", "full"), ("grpo_c_road_state", False),
                             ("grpo_c_road_scale", .25), ("grpo_c_offroad_penalty", 0.)):
            if ck["config"].get(key, default) != config[key]:
                raise ValueError(f"GRPO-C resume config mismatch: {key}")
        lineage = ck.get("warm_start")
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
    if lineage:
        save_json(run/"warm_start.json",lineage)
    writer=SummaryWriter(str(run/"tensorboard"))
    initial_steps,initial_time=steps,train_seconds
    best=-float("inf");eval_seconds=0.;wall=time.perf_counter();selected=0
    best_rank = None
    best_summary = None

    def checkpoint(name):
        state=dict(model=policies[selected].state_dict(),ensemble=[p.state_dict() for p in policies],
                   optimizers=[o.state_dict() for o in optimizers],reference_model=reference.state_dict(),
                   selected_member=selected,config=config,env_config=env_config,steps=steps,
                   frames=frames,iterations=iterations,train_seconds=train_seconds,
                   warm_start=lineage,
                   python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if args.device=="cuda" else None)
        tmp=run/f"{name}.tmp";torch.save(state,tmp);tmp.replace(run/f"{name}.pt")

    def validate():
        nonlocal best,eval_seconds,selected,best_rank,best_summary
        tick=time.perf_counter();results=[]
        member_episodes=[]
        parallel_rows = (evaluate_ensemble(policies, vec, args.device,
                         range(args.eval_seed_start,args.eval_seed_start+args.eval_episodes))
                         if args.grpo_c_parallel_eval and not (steps == 0 and lineage) else None)
        for member, policy in enumerate(policies):
            print(f"GRPO-C evaluating steps={steps} member={member} episodes={args.eval_episodes}",flush=True)
            if parallel_rows is not None:
                episodes=parallel_rows[member]
            elif steps == 0 and lineage and member > 0:
                # Warm-start actors are exact copies; deterministic evaluation
                # on identical tracks can safely reuse member 0 at step zero.
                episodes=[dict(row) for row in member_episodes[0]]
            else:
                episodes=evaluate(policy,args.device,env_config,
                                  range(args.eval_seed_start,args.eval_seed_start+args.eval_episodes))
            member_episodes.append(episodes)
            sm=summary(episodes);results.append(sm)
            print(f"  return={sm['mean_return']:.2f} coverage={sm['mean_coverage']:.1%} completion={sm['completion_rate']:.1%}",flush=True)
            append_csv(run/"member_validation.csv",dict(steps=steps,member=member,**sm))
            for episode in episodes:
                append_csv(run/"member_validation_episodes.csv",dict(steps=steps,member=member,**episode))
        eval_seconds+=time.perf_counter()-tick
        selected=max(range(len(results)),key=lambda i: validation_rank(results[i],args.grpo_c_selection))
        chosen=results[selected]
        append_csv(run/"validation.csv",dict(steps=steps,frames=frames,train_seconds=train_seconds,
                    selected_member=selected,**chosen))
        for episode in member_episodes[selected]:
            append_csv(run/"validation_episodes.csv",dict(steps=steps,**episode))
        for key,value in chosen.items():writer.add_scalar("validation/"+key,value,steps)
        rank=validation_rank(chosen,args.grpo_c_selection)
        if best_rank is None or rank>best_rank:
            best_rank=rank;best_summary=dict(chosen)
            best=chosen["mean_return"];checkpoint("best")
        checkpoint("last")
        print(f"GRPO-C validation steps={steps} member={selected} return={chosen['mean_return']:.2f} "
              f"coverage={chosen['mean_coverage']:.1%}",flush=True)

    vec=VectorRacing(args.num_envs,env_config,args.seed)
    next_eval=(steps//args.eval_every+1)*args.eval_every
    try:
        validate()
        if steps == 0:
            checkpoint("initial")
        while steps<args.steps:
            tick=time.perf_counter();before=steps
            batches=[];seeds=[];feature_batches=[]
            for _ in range(args.grpo_c_episodes_per_update):
                seed=int(np.random.randint(0,1_000_000))
                batch,used_frames=collect_ensemble(policies,vec,seed,args.device,args.grpo_c_completion_bonus,args.grpo_c_offroad_penalty)
                batches.append(batch);seeds.append(seed)
                steps+=sum(len(t["obs"]) for t in batch);frames+=used_frames
                feature_batches.append(trajectory_features(policies,reference,batch,args.device))
            iterations+=1
            features=np.mean(feature_batches,axis=0)
            clusters,merged=policy_groups(features,args.grpo_c_clusters)
            # Compare peers on the same track; never cluster unrelated tracks
            # together or propagate reward-to-go across episode resets.
            stats=[state_aware_advantages(batch,clusters,args.gamma,
                args.grpo_c_state_eps,args.grpo_c_state_min_samples,args.clip,
                args.grpo_c_state_fallback_clusters) for batch in batches]
            state_stats={key:float(np.mean([s[key] for s in stats])) for key in stats[0]}
            training_batch=batches[0] if len(batches)==1 else merge_episode_batches(batches)
            metrics=update_grpo_c(policies,reference,optimizers,training_batch,clusters,args)
            scores=np.mean([[t["score"] for t in batch] for batch in batches],axis=0)
            update_reference(reference,policies,scores,args.grpo_c_reference_tau)
            trajectories=[t for batch in batches for t in batch]
            for index,trajectory in enumerate(trajectories):
                member=index%len(policies);seed=seeds[index//len(policies)]
                actions=np.asarray(trajectory["actions"])
                append_csv(run/"train_episodes.csv",dict(steps=steps,iteration=iterations,
                           member=member,track_seed=seed,policy_cluster=int(clusters[member]),
                           training_objective_return=float(trajectory["score"]),
                           mean_gas=float(actions[:,1].mean()),mean_brake=float(actions[:,2].mean()),
                           gas_brake_overlap=float(((actions[:,1]>.2)&(actions[:,2]>.2)).mean()),
                           **trajectory["metrics"]))
            if args.device=="cuda":torch.cuda.synchronize()
            elapsed=time.perf_counter()-tick;train_seconds+=elapsed
            metrics.update(**state_stats,steps=steps,frames=frames,train_seconds=train_seconds,
                           episodes_per_policy=len(batches),sampled_episodes=len(trajectories),
                           merged_singleton_groups=merged,
                           train_mean_return=float(np.mean([t['metrics']['return_raw'] for t in trajectories])),
                           train_mean_objective_return=float(np.mean(scores)),
                           train_completion_rate=float(np.mean([t['metrics']['completed'] for t in trajectories])),
                           train_mean_coverage=float(np.mean([t['metrics']['coverage'] for t in trajectories])),
                           mean_gas=float(np.concatenate([t['actions'] for t in trajectories])[:,1].mean()),
                           mean_brake=float(np.concatenate([t['actions'] for t in trajectories])[:,2].mean()),
                           decision_sps=(steps-before)/elapsed,group_reward_std=float(np.std(scores)),
                           peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if args.device=="cuda" else 0)
            append_csv(run/"updates.csv",metrics)
            for key,value in metrics.items():writer.add_scalar("train/"+key,value,steps)
            print(f"GRPO-C decisions={steps}/{args.steps} sps={metrics['decision_sps']:.1f} "
                  f"groups={metrics['occupied_policy_clusters']} states={metrics['state_clusters']} "
                  f"return={metrics['train_mean_return']:.1f} coverage={metrics['train_mean_coverage']:.1%} "
                  f"brake={metrics['mean_brake']:.3f} KL={metrics['approx_kl']:.4f}",flush=True)
            if steps>=next_eval or steps>=args.steps:
                validate();next_eval=(steps//args.eval_every+1)*args.eval_every
        checkpoint("last")
        save_json(run/"finished.json",dict(steps=steps,requested_steps=args.steps,
                  budget_overshoot=steps-args.steps,frames=frames,train_seconds=train_seconds,
                  initial_steps=initial_steps,added_steps=steps-initial_steps,
                  segment_train_seconds=train_seconds-initial_time,evaluation_seconds=eval_seconds,
                  session_wall_seconds=time.perf_counter()-wall,best_validation_return=best,
                  selection_criterion=args.grpo_c_selection,best_validation=best_summary,
                  warm_start=lineage,
                  selected_member=selected,status="completed_full_ensemble_budget_not_proof_of_convergence"))
    except KeyboardInterrupt:
        checkpoint("interrupted")
        raise
    finally:
        vec.close();writer.close()
