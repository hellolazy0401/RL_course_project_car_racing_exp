"""Explicit state/control variants for Recurrent PPO; unchanged Box2D physics.

State variants use privileged track geometry. They are engineering extensions,
not pixel-only algorithm comparisons. The lap95 gate and 1000-frame cap remain.
"""
import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_racing import TRACK_WIDTH
from .envs import Racing


OFFSETS = (1, 3, 6, 10, 16, 24)
STATE_DIM = 34


def geometry(base):
    centers = np.asarray([p[2:4] for p in base.track])
    car = base.car
    position = np.asarray(car.hull.position)
    angle = car.hull.angle
    right = np.array([np.cos(angle), np.sin(angle)])
    forward = np.array([-np.sin(angle), np.cos(angle)])
    velocity = np.asarray(car.hull.linearVelocity)
    index = int(np.square(centers - position).sum(1).argmin())
    tangent = centers[(index + 1) % len(centers)] - centers[(index - 1) % len(centers)]
    tangent /= max(np.linalg.norm(tangent), 1e-8)
    return centers, position, right, forward, velocity, index, tangent


def controller(base, speed_scale=1.0, profile="v1"):
    """Preview steering and curvature-limited speed, with exclusive gas/brake."""
    centers, pos, right, forward, velocity, index, tangent = geometry(base)
    speed = float(velocity @ forward)
    # Velocity-dependent preview expressed in physical world units.
    if profile not in ("v1", "v2"):
        raise ValueError("Unknown controller profile")
    gain, yaw_gain, preview, cap = ((1.4, .12, .32, 30) if profile == "v1" else (.7, .08, .4, 45))
    lookahead = np.clip(5.0 + preview * max(speed, 0), 6, cap)
    distances = np.linalg.norm(centers - pos, axis=1)
    target = centers[(index + 1) % len(centers)]
    for offset in range(1, 25):
        candidate = (index + offset) % len(centers)
        target = centers[candidate]
        if distances[candidate] >= lookahead:
            break
    delta = target - pos
    # Positive action steers right (Gym internally negates the wheel angle).
    heading_error = np.arctan2(delta @ right, delta @ forward)
    steer = np.clip(gain * heading_error + yaw_gain * base.car.hull.angularVelocity, -1, 1)
    # Preview changes of road direction rather than the car's current yaw.
    curvature = 0.0
    for k in (0, 3, 6, 10):
        a, b, c = (centers[(index + k + j) % len(centers)] for j in (0, 3, 6))
        ab, bc = b - a, c - b
        turn = abs(np.arctan2(ab[0] * bc[1] - ab[1] * bc[0], ab @ bc))
        curvature = max(curvature, turn / max(np.linalg.norm(ab), 1e-6))
    target_speed = np.clip(np.sqrt(16.0 / max(curvature, .001)), 24, 78) * speed_scale
    if abs(heading_error) > .65:
        target_speed = min(target_speed, 28)
    error = target_speed - speed
    accel = np.clip(error * .10, -.8, 1)
    return np.array([steer, accel], dtype=np.float32), target_speed


def controls(command):
    steering, longitudinal = np.clip(command, -1, 1)
    return np.array([steering, max(longitudinal, 0), max(-longitudinal, 0)], np.float32)


class RecurrentRacing(Racing):
    def __init__(self, observation_mode="state", control_mode="direct",
                 speed_scale=1.0, residual_scale=.2, controller_profile="v1",
                 finish_assist=False, **kwargs):
        if observation_mode not in ("pixels", "state"):
            raise ValueError("observation_mode must be pixels or state")
        if control_mode not in ("direct", "residual", "adaptive"):
            raise ValueError("control_mode must be direct, residual or adaptive")
        if control_mode in ("residual", "adaptive") and observation_mode != "state":
            raise ValueError("Residual control requires explicitly privileged state observations")
        self.observation_mode, self.control_mode = observation_mode, control_mode
        self.speed_scale, self.residual_scale = speed_scale, residual_scale
        if controller_profile not in ("v1", "v2"):
            raise ValueError("Unknown controller profile")
        self.controller_profile = controller_profile
        if finish_assist and observation_mode != "state":
            raise ValueError("Finish assistance requires explicitly privileged state")
        self.finish_assist = finish_assist
        self.assist_active = False
        self.previous_command = np.zeros(2, np.float32)
        super().__init__(**kwargs)
        self.action_space = gym.spaces.Box(-1, 1, (2,), np.float32)
        self._original_render = self.unwrapped._render
        self._render_phase, self._resetting = 0, True
        self._render_cache = np.zeros((96, 96, 3), np.uint8)
        if observation_mode == "state":
            self.observation_space = gym.spaces.Box(-5, 5, (STATE_DIM,), np.float32)
            # Skip only unused observation rendering. RGB video rendering still
            # calls the original renderer; physics and reward are untouched.
            original_render = self.unwrapped._render
            blank = np.zeros((96, 96, 3), np.uint8)
            self.unwrapped._render = lambda mode: blank if mode == "state_pixels" else original_render(mode)
        else:
            self.unwrapped._render = self._render_policy_pixels

    def _render_policy_pixels(self, mode):
        # Action repeat consumes only its last observation. Avoid drawing the
        # unused intermediate policy frames; full-frame RGB videos are separate.
        if mode != "state_pixels" or self._resetting:
            return self._original_render(mode)
        self._render_phase += 1
        if self._render_phase % self.repeat == 0:
            self._render_cache = self._original_render(mode)
        return self._render_cache

    def reset(self, **kwargs):
        self.previous_command = np.zeros(2, np.float32)
        self.assist_active = False
        self._resetting = True
        try:
            return super().reset(**kwargs)
        finally:
            self._resetting = False
            self._render_phase = 0

    def observation(self):
        if self.observation_mode == "pixels":
            return super().observation()
        base = self.unwrapped
        centers, pos, right, forward, velocity, index, tangent = geometry(base)
        normal = np.array([tangent[1], -tangent[0]])
        prior, target_speed = controller(base, self.speed_scale, self.controller_profile)
        values = [
            velocity @ forward / 80, velocity @ right / 30,
            base.car.hull.angularVelocity / 4, (pos - centers[index]) @ normal / TRACK_WIDTH,
            tangent @ right, tangent @ forward,
        ]
        for offset in OFFSETS:
            delta = centers[(index + offset) % len(centers)] - pos
            values.extend([delta @ right / 100, delta @ forward / 100])
        values.extend([
            base.tile_visited_count / len(centers), self.raw_steps / 1000,
            base.car.wheels[0].joint.angle / .5,
            sum(bool(w.tiles) for w in base.car.wheels) / 4,
            *self.previous_command, *prior, target_speed / 100,
            *[w.omega / 100 for w in base.car.wheels],
            float(base.car.hull.linearVelocity.length) / 100,
            float(index) / len(centers), len(centers) / 400,
        ])
        return np.clip(np.asarray(values, np.float32), -5, 5)

    def step(self, action):
        if np.shape(action) != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected two finite action values")
        command = np.clip(action, -1, 1)
        if self.control_mode == "adaptive":
            # Learn a local speed target as well as steering. A throttle-only
            # residual has little authority when the base controller saturates.
            prior, _ = controller(self.unwrapped, self.speed_scale * (1 + .25 * command[1]),
                                  self.controller_profile)
            command = np.array([np.clip(prior[0] + self.residual_scale * command[0], -1, 1),
                                prior[1]], np.float32)
        elif self.control_mode == "residual":
            prior, _ = controller(self.unwrapped, self.speed_scale, self.controller_profile)
            command = np.clip(prior + self.residual_scale * command, -1, 1)
        if self.finish_assist:
            base = self.unwrapped
            centers, pos, _, _, _, index, _ = geometry(base)
            near_finish = ((index >= len(centers)-24 or index < 3)
                           and np.linalg.norm(pos-centers[0]) < 80)
            if base.tile_visited_count / len(centers) > .90 and near_finish:
                self.assist_active = True
            if self.assist_active:
                # Physical driving only: never modify contacts, tile counts,
                # rewards, termination flags or the original lap finish gate.
                command, _ = controller(base, speed_scale=1.2, profile="v1")
        self.previous_command = command.astype(np.float32)
        obs, reward, term, trunc, info = super().step(controls(command))
        if self.observation_mode == "pixels" and (term or trunc) and self._render_phase % self.repeat:
            # A terminal before the final repeat still needs its actual final
            # observation for GAE bootstrap, never a cached previous frame.
            final_pixels = self._original_render("state_pixels")
            self.unwrapped.state = final_pixels
            self.frames[-1] = self.image(final_pixels)
            obs = self.observation()
        info["applied_command"] = self.previous_command.copy()
        return obs, reward, term, trunc, info


def make_env(config, **extra):
    config = dict(config)
    recurrent = config.pop("recurrent", False)
    return (RecurrentRacing if recurrent else Racing)(**config, **extra)
