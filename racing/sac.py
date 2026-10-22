"""Pixel SAC-v2: twin Q, reparameterized policy, automatic temperature."""
import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from .envs import VectorRacing
from .sac_networks import actor_net, q_encoder, ObservationHistory
from .utils import append_csv, provenance, save_json, summary


class SACActor(nn.Module):
    def __init__(self, architecture="baseline"):
        super().__init__()
        self.net = actor_net(architecture)

    def forward(self, obs, deterministic=False):
        mean, logstd = self.net(obs.float() / 255.).chunk(2, dim=-1)
        dist = Normal(mean, logstd.clamp(-20, 2).exp())
        z = mean if deterministic else dist.rsample()
        u = z.tanh()
        # Entropy is defined in normalized [-1, 1]^3 coordinates. Q sees
        # physical actions; the constant affine Jacobian is deliberately
        # excluded together with the corresponding target-entropy shift.
        logp = (dist.log_prob(z) - 2 * (math.log(2) - z - F.softplus(-2*z))).sum(-1)
        action = torch.stack((u[:, 0], (u[:, 1]+1)/2, (u[:, 2]+1)/2), -1)
        return action, z, logp, -logp


class QNetwork(nn.Module):
    def __init__(self, architecture="baseline"):
        super().__init__()
        self.encoder = q_encoder(architecture)
        layers = [nn.Linear(259, 256)]
        if architecture == "layernorm":
            layers.append(nn.LayerNorm(256))
        self.head = nn.Sequential(*layers, nn.ReLU(), nn.Linear(256, 1))

    def forward(self, obs, action):
        features = self.encoder(obs.float() / 255.)
        return self.head(torch.cat((features, action), -1)).squeeze(-1)


class SACAgent(nn.Module):
    def __init__(self, architecture="baseline", context_length=8):
        super().__init__()
        self.architecture = architecture
        self.context_length = context_length if architecture == "gru" else 1
        self.actor = SACActor(architecture)
        self.q1, self.q2 = QNetwork(architecture), QNetwork(architecture)

    def action(self, obs, deterministic=False):
        return self.actor(obs, deterministic)


def terminal_observations(next_obs, infos):
    """Store the final state, never the auto-reset state, at either boundary."""
    return np.stack([info.get("terminal_observation", obs) for obs, info in zip(next_obs, infos)])


class ReplayBuffer:
    def __init__(self, capacity, shape=(4, 64, 64)):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.obs = np.empty((capacity, *shape), np.uint8)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty((capacity, 3), np.float32)
        self.rewards = np.empty(capacity, np.float32)
        self.terminated = np.empty(capacity, np.bool_)

    def add(self, obs, actions, rewards, next_obs, terminated, boundaries=None):
        for o, a, r, no, t in zip(obs, actions, rewards, next_obs, terminated):
            i = self.pos
            self.obs[i], self.actions[i], self.rewards[i] = o, a, r
            self.next_obs[i], self.terminated[i] = no, t
            self.pos = (i + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        indices = np.random.randint(self.size, size=batch_size)
        return tuple(torch.as_tensor(getattr(self, k)[indices], device=device) for k in
                     ("obs", "actions", "rewards", "next_obs", "terminated"))

    def state_dict(self):
        return dict(size=self.size, pos=self.pos, capacity=self.capacity,
                    **{k: getattr(self, k)[:self.size] for k in
                       ("obs", "actions", "rewards", "next_obs", "terminated")})

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity mismatch")
        self.size, self.pos = state["size"], state["pos"]
        for k in ("obs", "actions", "rewards", "next_obs", "terminated"):
            getattr(self, k)[:self.size] = state[k]


class SequenceReplayBuffer(ReplayBuffer):
    """Links refer to monotonically increasing IDs, so overwritten slots
    cannot create false histories. Each environment has a separate chain.
    The minibatch has one supervised final transition per sampled sequence.
    """
    def __init__(self, capacity, num_envs, length, shape=(4, 64, 64)):
        super().__init__(capacity, shape)
        self.length, self.count = length, 0
        self.ids = np.full(capacity, -1, dtype=np.int64)
        self.prev = np.full(capacity, -1, dtype=np.int64)
        self.last_ids = np.full(num_envs, -1, dtype=np.int64)

    def start_episodes(self):
        self.last_ids.fill(-1)

    def add(self, obs, actions, rewards, next_obs, terminated, boundaries=None):
        boundaries = terminated if boundaries is None else boundaries
        for env in range(len(obs)):
            i = self.pos
            self.ids[i], self.prev[i] = self.count, self.last_ids[env]
            self.last_ids[env] = -1 if boundaries[env] else self.count
            self.count += 1
            super().add(obs[env:env+1], actions[env:env+1], rewards[env:env+1],
                        next_obs[env:env+1], terminated[env:env+1])

    def sequences(self, indices):
        obs = np.zeros((len(indices), self.length, *self.obs.shape[1:]), np.uint8)
        current = self.ids[indices].copy()
        for t in reversed(range(self.length)):
            slots = current % self.capacity
            valid = (current >= 0) & (self.ids[slots] == current)
            obs[valid, t] = self.obs[slots[valid]]
            current = np.where(valid, self.prev[slots], -1)
        next_obs = np.concatenate((obs[:, 1:], self.next_obs[indices, None]), axis=1)
        return obs, next_obs

    def sample(self, batch_size, device):
        indices = np.random.randint(self.size, size=batch_size)
        obs, next_obs = self.sequences(indices)
        return tuple(torch.as_tensor(x, device=device) for x in
                     (obs, self.actions[indices], self.rewards[indices], next_obs, self.terminated[indices]))

    def state_dict(self):
        state = super().state_dict()
        state.update(length=self.length, count=self.count, ids=self.ids, prev=self.prev, last_ids=self.last_ids)
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if state["length"] != self.length or len(state["last_ids"]) != len(self.last_ids):
            raise ValueError("Recurrent replay shape mismatch")
        self.count = state["count"]
        for key in ("ids", "prev", "last_ids"):
            getattr(self, key)[:] = state[key]


def bellman_target(rewards, terminated, next_q, next_logp, alpha, gamma):
    # Time limits bootstrap, true terminal states do not.
    return rewards + gamma * (~terminated).float() * (next_q - alpha * next_logp)


class SACLearner:
    def __init__(self, args):
        self.args = args
        self.agent = SACAgent(args.sac_arch, args.sac_context).to(args.device)
        self.targets = nn.ModuleList([copy.deepcopy(self.agent.q1), copy.deepcopy(self.agent.q2)])
        self.targets.requires_grad_(False)
        self.qparams = list(self.agent.q1.parameters()) + list(self.agent.q2.parameters())
        self.actor_opt = torch.optim.Adam(self.agent.actor.parameters(), lr=args.actor_lr)
        self.q_opt = torch.optim.Adam(self.qparams, lr=args.critic_lr)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(args.sac_alpha), device=args.device))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.sac_alpha_lr)
        self.updates = 0

    @staticmethod
    def step(loss, optimizer):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite SAC loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def update(self, batch):
        obs, actions, rewards, next_obs, terminated = batch
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, _, next_logp, _ = self.agent.action(next_obs)
            next_q = torch.minimum(*(q(next_obs, next_action) for q in self.targets))
            target = bellman_target(rewards, terminated, next_q, next_logp, alpha, self.args.gamma)
        q1, q2 = self.agent.q1(obs, actions), self.agent.q2(obs, actions)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.step(q_loss, self.q_opt)
        self.q_opt.zero_grad(set_to_none=True)
        for p in self.qparams:
            p.requires_grad_(False)
        try:
            action, _, logp, _ = self.agent.action(obs)
            q = torch.minimum(self.agent.q1(obs, action), self.agent.q2(obs, action))
            policy_loss = (alpha * logp - q).mean()
            self.step(policy_loss, self.actor_opt)
        finally:
            for p in self.qparams:
                p.requires_grad_(True)
        alpha_loss = -(self.log_alpha * (logp.detach() + self.args.sac_target_entropy)).mean()
        if self.args.sac_auto_alpha:
            self.step(alpha_loss, self.alpha_opt)
        with torch.no_grad():
            for source, target_net in zip((self.agent.q1, self.agent.q2), self.targets):
                for src, dst in zip(source.parameters(), target_net.parameters()):
                    dst.lerp_(src, self.args.sac_tau)
        self.updates += 1
        return dict(policy_loss=policy_loss.item(), q_loss=q_loss.item(),
                    q1_mean=q1.detach().mean().item(), q2_mean=q2.detach().mean().item(),
                    target_q_mean=target.mean().item(), alpha=self.log_alpha.exp().item(),
                    alpha_loss=alpha_loss.item(), entropy=-logp.detach().mean().item())


def validate_args(args, parser):
    for key in ("steps", "num_envs", "rollout", "batch_size", "threads", "eval_every", "eval_episodes",
                "sac_buffer_size", "sac_gradient_steps", "sac_context"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if args.steps % args.num_envs:
        parser.error("SAC steps must be divisible by num-envs for an exact decision budget")
    if args.sac_buffer_size < args.batch_size or args.sac_learning_starts < 0:
        parser.error("SAC buffer must hold a batch; learning-starts must be nonnegative")
    for key in ("actor_lr", "critic_lr", "reward_scale", "sac_alpha", "sac_alpha_lr"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"{key} must be finite and positive")
    if not (0 < args.gamma <= 1 and 0 < args.sac_tau <= 1) or not math.isfinite(args.sac_target_entropy):
        parser.error("Invalid SAC gamma/tau/target entropy")
    if args.seed < 0 or args.eval_seed_start < 0:
        parser.error("Seeds must be nonnegative")
    if args.branch_from or args.target_kl is not None:
        parser.error("SAC does not use PPO branch-from/target-kl")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu")


def validate_restore(ck, config):
    if "replay" not in ck:
        raise ValueError("SAC resume needs last.pt or interrupted.pt including replay; best.pt is evaluation-only")
    keys = ("algo", "seed", "num_envs", "batch_size", "actor_lr", "critic_lr", "gamma", "reward_scale",
            "completion_mode", "sac_buffer_size", "sac_learning_starts", "sac_gradient_steps",
            "sac_tau", "sac_alpha", "sac_auto_alpha", "sac_alpha_lr", "sac_target_entropy")
    for key in keys:
        if ck["config"].get(key) != config[key]:
            raise ValueError(f"SAC resume config mismatch: {key}")
    for key, default in (("sac_arch", "baseline"), ("sac_context", 8)):
        if ck["config"].get(key, default) != config.get(key, default):
            raise ValueError(f"SAC resume config mismatch: {key}")


def train(args, parser):
    from .evaluate import evaluate
    validate_args(args, parser)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = vars(args).copy()
    env_config = dict(repeat=4, stack=4, size=64, completion_mode=args.completion_mode)
    # Load on CPU: replay must not consume GPU memory on restore.
    ck = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if ck is not None:
        validate_restore(ck, config)
        if args.steps <= ck["steps"]:
            parser.error("--steps must exceed restored steps")
    run = Path(args.out) / (args.name or f"sac_s{args.seed}")
    run.mkdir(parents=True, exist_ok=False)
    save_json(run / "config.json", config)
    save_json(run / "provenance.json", provenance())
    learner = SACLearner(args)
    recurrent = args.sac_arch == "gru"
    replay = (SequenceReplayBuffer(args.sac_buffer_size, args.num_envs, args.sac_context)
              if recurrent else ReplayBuffer(args.sac_buffer_size))
    save_json(run / "model_info.json", dict(architecture=args.sac_arch,
              parameters=sum(p.numel() for p in learner.agent.parameters()),
              context_length=learner.agent.context_length))
    steps, frames, train_seconds, eval_seconds = 0, 0, 0., 0.
    best = -float("inf")
    if ck is not None:
        learner.agent.load_state_dict(ck["model"])
        learner.targets.load_state_dict(ck["targets"])
        learner.actor_opt.load_state_dict(ck["actor_optimizer"])
        learner.q_opt.load_state_dict(ck["critic_optimizer"])
        learner.log_alpha.data.copy_(ck["log_alpha"].to(args.device))
        learner.alpha_opt.load_state_dict(ck["alpha_optimizer"])
        learner.updates = ck["updates"]
        replay.load_state_dict(ck["replay"])
        steps, frames, train_seconds = ck["steps"], ck["frames"], ck["train_seconds"]
        random.setstate(ck["python_rng"])
        np.random.set_state(ck["numpy_rng"])
        torch.set_rng_state(ck["torch_rng"])
        if args.device == "cuda" and ck["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        del ck
    if recurrent:
        replay.start_episodes()  # A resumed Box2D process starts fresh episodes.
    writer = SummaryWriter(str(run / "tensorboard"))

    def checkpoint(name, full=True):
        state = dict(model=learner.agent.state_dict(), config=config, env_config=env_config,
                     steps=steps, frames=frames, train_seconds=train_seconds, updates=learner.updates)
        if full:
            state.update(targets=learner.targets.state_dict(), replay=replay.state_dict(),
                         actor_optimizer=learner.actor_opt.state_dict(), critic_optimizer=learner.q_opt.state_dict(),
                         log_alpha=learner.log_alpha.detach(), alpha_optimizer=learner.alpha_opt.state_dict(),
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if args.device == "cuda" else None)
        tmp = run / f"{name}.tmp"
        torch.save(state, tmp)
        tmp.replace(run / f"{name}.pt")

    def validation():
        nonlocal best, eval_seconds
        tick = time.perf_counter()
        rows = evaluate(learner.agent, args.device, env_config,
                        range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
        sm = summary(rows)
        eval_seconds += time.perf_counter() - tick
        append_csv(run / "validation.csv", dict(steps=steps, frames=frames, train_seconds=train_seconds, **sm))
        for row in rows:
            append_csv(run / "validation_episodes.csv", dict(steps=steps, **row))
        for key, value in sm.items():
            writer.add_scalar("validation/"+key, value, steps)
        if sm["mean_return"] > best:
            best = sm["mean_return"]
            checkpoint("best", full=False)
        print(f"SAC validation steps={steps} return={sm['mean_return']:.1f}", flush=True)

    initial_steps, initial_seconds = steps, train_seconds
    vec = VectorRacing(args.num_envs, env_config, args.seed + steps)
    wall_start = time.perf_counter()
    next_eval = (steps // args.eval_every + 1) * args.eval_every
    try:
        obs = vec.reset()
        history = ObservationHistory(obs, args.sac_context) if recurrent else None
        validation()
        while steps < args.steps:
            tick = time.perf_counter()
            block_steps, block_updates, sums = 0, 0, {}
            for _ in range(min(args.rollout, (args.steps-steps)//args.num_envs)):
                if steps < args.sac_learning_starts:
                    actions = np.random.uniform([-1, 0, 0], [1, 1, 1], (args.num_envs, 3)).astype(np.float32)
                else:
                    with torch.no_grad():
                        policy_obs = history.data if recurrent else obs
                        actions = learner.agent.action(torch.as_tensor(policy_obs, device=args.device))[0].cpu().numpy()
                next_obs, rewards, term, trunc, infos = vec.step(actions)
                replay.add(obs, actions, rewards * args.reward_scale, terminal_observations(next_obs, infos), term, term | trunc)
                if recurrent:
                    history.advance(next_obs, term | trunc)
                obs = next_obs
                steps += args.num_envs
                block_steps += args.num_envs
                frames += sum(info["raw_frames"] for info in infos)
                for info in infos:
                    if "episode_metrics" in info:
                        append_csv(run / "train_episodes.csv", dict(steps=steps, **info["episode_metrics"]))
                if steps >= args.sac_learning_starts and replay.size >= args.batch_size:
                    for _ in range(args.sac_gradient_steps):
                        metrics = learner.update(replay.sample(args.batch_size, args.device))
                        for key, value in metrics.items():
                            sums[key] = sums.get(key, 0.) + value
                        block_updates += 1
                if steps >= next_eval:
                    break
            if args.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            train_seconds += elapsed
            keys = ("policy_loss", "q_loss", "q1_mean", "q2_mean", "target_q_mean", "alpha", "alpha_loss", "entropy")
            metrics = {key: sums[key]/block_updates if block_updates else float("nan") for key in keys}
            metrics.update(steps=steps, frames=frames, train_seconds=train_seconds, decision_sps=block_steps/elapsed,
                           replay_size=replay.size, actor_updates=block_updates, critic_updates=block_updates,
                           total_updates=learner.updates,
                           peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20 if args.device == "cuda" else 0)
            append_csv(run / "updates.csv", metrics)
            for key, value in metrics.items():
                writer.add_scalar("train/"+key, value, steps)
            print(f"sac seed={args.seed} decisions={steps}/{args.steps} updates={learner.updates} "
                  f"sps={metrics['decision_sps']:.1f}", flush=True)
            if steps >= next_eval or steps >= args.steps:
                validation()
                checkpoint("last")
                next_eval = (steps // args.eval_every + 1) * args.eval_every
        save_json(run / "finished.json", dict(steps=steps, frames=frames, train_seconds=train_seconds,
                  initial_steps=initial_steps, added_steps=steps-initial_steps,
                  segment_train_seconds=train_seconds-initial_seconds, evaluation_seconds=eval_seconds,
                  session_wall_seconds=time.perf_counter()-wall_start, best_validation_return=best,
                  total_updates=learner.updates, status="completed_budget_not_proof_of_convergence"))
    except KeyboardInterrupt:
        checkpoint("interrupted")
        raise
    finally:
        vec.close()
        writer.close()
