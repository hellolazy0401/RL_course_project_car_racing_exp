"""Identical pixel/action/reward protocol for every algorithm."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
from collections import deque
import multiprocessing as mp
import traceback
import cv2
import gymnasium as gym
import numpy as np


class LapFinishGate:
    """Re-entry to tile 0, forward motion and >95% coverage; all tiles also finish."""
    def __init__(self, on_start):
        self.was_on_start = bool(on_start)
        self.left_start = False

    def update(self, on_start, forward, visited, total):
        entered = on_start and not self.was_on_start
        if not on_start:
            self.left_start = True
        finish = visited == total or (self.left_start and entered and forward and visited / total > .95)
        self.was_on_start = bool(on_start)
        return bool(finish)


class Racing(gym.Wrapper):
    def __init__(self, repeat=4, stack=4, size=64, domain_randomize=False, completion_mode="original"):
        if completion_mode not in ("original", "lap95"):
            raise ValueError("Unknown completion_mode")
        super().__init__(gym.make("CarRacing-v3", continuous=True,
            render_mode="rgb_array", domain_randomize=domain_randomize,
            lap_complete_percent=0.95, max_episode_steps=1000))
        self.repeat, self.stack, self.size = repeat, stack, size
        self.frames = deque(maxlen=stack)
        self.observation_space = gym.spaces.Box(0, 255, (stack, size, size), np.uint8)
        self.video_writer = None
        self.completion_mode = completion_mode

    def on_start(self):
        return any(tile.idx == 0 for wheel in self.unwrapped.car.wheels for tile in wheel.tiles)

    def image(self, obs):
        return cv2.resize(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY),
                          (self.size, self.size), interpolation=cv2.INTER_AREA)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.finish_gate = LapFinishGate(self.on_start())
        track = self.unwrapped.track
        self.start_tangent = np.array(track[1][2:4]) - np.array(track[-1][2:4])
        self.frames.clear()
        self.frames.extend([self.image(obs)] * self.stack)
        self.raw_return, self.raw_steps, self.decisions = 0., 0, 0
        return np.stack(self.frames), info

    def step(self, action):
        reward, used = 0., 0
        for _ in range(self.repeat):
            obs, r, terminated, truncated, info = self.env.step(action)
            if self.completion_mode == "lap95":
                base = self.unwrapped
                forward = np.dot(np.array(base.car.hull.linearVelocity), self.start_tangent) > 0
                finished = self.finish_gate.update(self.on_start(), forward, base.tile_visited_count, len(base.track))
                # Keep native failure termination (e.g. outside playfield).
                failed = terminated and not info.get("lap_finished", False)
                if not failed:
                    terminated = finished
                    info["lap_finished"] = finished
            reward += r
            used += 1
            if self.video_writer is not None:
                self.video_writer.append_data(self.env.render())
            if terminated or truncated:
                break
        self.frames.append(self.image(obs))
        self.raw_return += reward
        self.raw_steps += used
        self.decisions += 1
        info["raw_frames"] = used
        if terminated or truncated:
            base = self.unwrapped
            coverage = float(base.tile_visited_count / len(base.track))
            # Completion is distinct from crossing 95% before returning to start.
            complete = bool(terminated and info.get("lap_finished", False))
            info["episode_metrics"] = dict(return_raw=float(self.raw_return),
                raw_frames=self.raw_steps, decisions=self.decisions,
                coverage=coverage, completed=complete, terminated=bool(terminated),
                truncated=bool(truncated), score_ge_900=bool(self.raw_return >= 900))
        return np.stack(self.frames), reward, terminated, truncated, info


def _worker(conn, config, seed):
    env = None
    try:
        cv2.setNumThreads(1)
        env = Racing(**config)
        rng = np.random.default_rng(seed)
        while True:
            cmd, data = conn.recv()
            if cmd == "close":
                break
            if cmd == "reset":
                result = env.reset(seed=int(rng.integers(0, 1_000_000)))[0]
            elif cmd == "step":
                obs, reward, term, trunc, info = env.step(data)
                if term or trunc:
                    info["terminal_observation"] = obs
                    obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
                result = (obs, reward, term, trunc, info)
            else:
                raise ValueError(cmd)
            conn.send((True, result))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        conn.send((False, traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        conn.close()


class VectorRacing:
    """Small spawn-safe process vector. Auto-reset preserves terminal observations."""
    def __init__(self, n, config, seed):
        ctx = mp.get_context("spawn")
        self.pipes, self.processes = [], []
        for i in range(n):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker, args=(child, config, seed * 1000 + i), daemon=True)
            proc.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(proc)

    def receive(self):
        results = []
        for pipe in self.pipes:
            if not pipe.poll(180):
                raise TimeoutError("Environment worker did not reply within 180 s")
            ok, result = pipe.recv()
            if not ok:
                raise RuntimeError(result)
            results.append(result)
        return results

    def reset(self):
        for pipe in self.pipes:
            pipe.send(("reset", None))
        return np.stack(self.receive())

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(("step", action))
        obs, rew, term, trunc, infos = zip(*self.receive())
        return np.stack(obs), np.array(rew), np.array(term), np.array(trunc), infos

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        for pipe in self.pipes:
            pipe.close()
