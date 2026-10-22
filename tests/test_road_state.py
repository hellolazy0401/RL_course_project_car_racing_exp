import numpy as np
import pytest
import torch
from types import SimpleNamespace

from racing.envs import Racing
from racing.model import Agent
from racing.grpo_c import configure_trainable, warm_start_policies, visual_state_features
from racing.grpo_c import collect_ensemble
from racing.road_state import road_features, TRACK_WIDTH


def test_geometry_uses_vehicle_frame_and_signed_road_offset():
    centers = np.column_stack([np.zeros(30), np.arange(30)])
    wheels = [SimpleNamespace(tiles={1}, joint=SimpleNamespace(angle=.1)) for _ in range(4)]
    hull = SimpleNamespace(position=(1., 5.), linearVelocity=(3., 12.), angle=0., angularVelocity=.4)
    base = SimpleNamespace(car=SimpleNamespace(hull=hull, wheels=wheels), tile_visited_count=15)
    values = road_features(base, centers, 200)
    np.testing.assert_allclose(values[:6], [.2, .1, .1, 1/TRACK_WIDTH, 0., 1.], atol=1e-6)
    np.testing.assert_allclose(values[6:12], [-1/20, 3/20, -1/40, 8/40, -1/70, 16/70], atol=1e-6)
    np.testing.assert_allclose(values[12:], [.5, .2, .2, 1.], atol=1e-6)


def test_training_penalty_does_not_replace_native_metrics():
    class Policy:
        def action(self, obs):
            return torch.zeros((1, 3)), torch.zeros((1, 3)), torch.zeros(1), None

    class OneStep:
        def reset_same_seed(self, seed):
            return np.zeros((2, 5, 64, 64), dtype=np.uint8)

        def step_active(self, active, actions):
            return [(np.zeros((5, 64, 64), np.uint8), 10., i == 0, i == 1,
                     dict(raw_frames=4, offroad_fraction_sum=2.,
                          episode_metrics=dict(completed=i == 0, return_raw=10.)))
                    for i in active]

    trajectories, frames = collect_ensemble([Policy(), Policy()], OneStep(), 1, 'cpu', 200., .1)
    assert frames == 8
    assert trajectories[0]['score'] == pytest.approx(209.8)
    assert trajectories[1]['score'] == pytest.approx(9.8)
    assert all(t['metrics']['return_raw'] == 10. for t in trajectories)


def test_state_observations_preserve_pixels_rewards_and_endings():
    plain = Racing(completion_mode='lap95')
    extra = Racing(completion_mode='lap95', road_state=True)
    try:
        a, _ = plain.reset(seed=917)
        b, _ = extra.reset(seed=917)
        assert b.shape == (5, 64, 64)
        np.testing.assert_array_equal(a, b[:4])
        for _ in range(250):
            a, ar, at, ax, ai = plain.step(np.array([.1, .3, 0]))
            b, br, bt, bx, bi = extra.step(np.array([.1, .3, 0]))
            np.testing.assert_array_equal(a, b[:4])
            assert (ar, at, ax) == (br, bt, bx)
            features = visual_state_features(b[None])
            assert features.shape == (1, 16)
            assert np.isfinite(features).all() and np.abs(features).max() <= 1
            assert 0 <= bi['offroad_fraction_sum'] <= bi['raw_frames']
            if at or ax:
                assert ai['episode_metrics'] == bi['episode_metrics']
                break
        else:
            pytest.fail('Episode did not respect 1000-frame limit')
    finally:
        plain.close()
        extra.close()


def test_warm_transfer_zero_residual_and_frozen_visual_policy(tmp_path):
    torch.set_num_threads(2)
    source = Agent()
    target = Agent(with_critic=False, road_state=True)
    env = dict(repeat=4, stack=4, size=64, completion_mode='lap95')
    path = tmp_path/'source.pt'
    torch.save(dict(config={'algo': 'ppo'}, env_config=env, model=source.state_dict(), steps=1), path)
    with pytest.raises(ValueError, match='explicit'):
        warm_start_policies([target], path, dict(env, road_state=True))
    lineage = warm_start_policies([target], path, dict(env, road_state=True), allow_observation_transfer=True)
    assert lineage['observation_protocol_transfer']
    with pytest.raises(ValueError, match='environment mismatch'):
        warm_start_policies([target], path, dict(env, road_state=True, repeat=2), allow_observation_transfer=True)
    obs = torch.randint(1, 256, (3, 5, 64, 64), dtype=torch.uint8)
    torch.testing.assert_close(source.action(obs[:, :4], deterministic=True)[0],
                               target.action(obs, deterministic=True)[0], rtol=0, atol=0)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    configure_trainable(target, 'road-head')
    optimizer = torch.optim.Adam(target.actor_parameters(), lr=.01)
    target.distribution(obs).loc.sum().backward()
    optimizer.step()
    for key, value in target.state_dict().items():
        if not key.startswith('road_head.'):
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert not torch.equal(target.road_head.weight, before['road_head.weight'])
    altered = obs.clone()
    altered[:, 4] = 128
    assert not torch.equal(target.distribution(obs).loc, target.distribution(altered).loc)
