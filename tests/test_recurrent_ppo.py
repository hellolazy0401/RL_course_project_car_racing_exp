import numpy as np
import pytest
import torch
from racing.recurrent_ppo import (RecurrentAgent, completion_summary, selection_key,
                                  sequence_indices, ppo_update, validate_restore)
from racing.recurrent_env import RecurrentRacing, controls, STATE_DIM
from racing.envs import Racing
from racing.train import parser


def test_lstm_sequence_matches_streaming_and_resets():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    agent = RecurrentAgent(hidden_size=16)
    obs = torch.randn(8, 2, STATE_DIM)
    starts = torch.zeros(8, 2, dtype=torch.bool)
    starts[0] = True
    starts[4, 0] = True
    state = agent.initial_state(2)
    seq = agent.sequence(obs, state, starts, deterministic=True)
    outputs = []
    for t in range(8):
        result = agent.step(obs[t], state, starts[t], deterministic=True)
        state = result[-1]
        outputs.append(result[0])
    torch.testing.assert_close(torch.stack(outputs), seq[0])
    solo = agent.sequence(obs[4:, :1], agent.initial_state(1), starts[4:, :1], deterministic=True)
    torch.testing.assert_close(solo[0], seq[0][4:, :1])
    # Changing pre-reset observations cannot affect post-reset outputs.
    obs[:4, 0] += 50
    changed = agent.sequence(obs, agent.initial_state(2), starts, deterministic=True)[0]
    torch.testing.assert_close(changed[4:, 0], seq[0][4:, 0])


def test_bptt_has_temporal_gradient_and_no_cross_episode_gradient():
    agent = RecurrentAgent(hidden_size=16)
    obs = torch.randn(6, 1, STATE_DIM, requires_grad=True)
    starts = torch.zeros(6, 1, dtype=torch.bool)
    starts[3] = True
    result = agent.sequence(obs, agent.initial_state(1), starts, deterministic=True)
    result[0][-1].sum().backward()
    assert obs.grad[:3].abs().sum() == 0
    assert obs.grad[3:5].abs().sum() > 0


def test_state_observations_and_actions_preserve_physics_and_finish():
    state = RecurrentRacing(completion_mode="lap95")
    original = Racing(completion_mode="lap95")
    try:
        obs, _ = state.reset(seed=89)
        original.reset(seed=89)
        assert obs.shape == (STATE_DIM,) and state.observation_space.contains(obs)
        for command in ([.1, .5], [.2, -.2], [-.1, .8]) * 5:
            a = np.array(command, np.float32)
            obs, r1, t1, u1, i1 = state.step(a)
            _, r2, t2, u2, i2 = original.step(controls(a))
            assert state.observation_space.contains(obs)
            assert (r1, t1, u1) == (r2, t2, u2)
            np.testing.assert_array_equal(state.unwrapped.car.hull.position, original.unwrapped.car.hull.position)
            applied = controls(a)
            assert applied[1] * applied[2] == 0
    finally:
        state.close()
        original.close()


def test_completion_ranking_prioritizes_finish_over_speed_and_return():
    def row(done, frames, score):
        return dict(completed=done, raw_frames=frames, return_raw=score, coverage=1 if done else .9)
    reliable = completion_summary([row(True, 900, 800), row(True, 950, 800)])
    fast = completion_summary([row(True, 500, 950), row(False, 1000, 890)])
    assert selection_key(reliable) > selection_key(fast)
    faster = completion_summary([row(True, 800, 750), row(True, 850, 750)])
    assert selection_key(faster) > selection_key(reliable)
    assert fast["mean_finish_seconds"] == 10
    assert fast["penalized_seconds"] == 25


def test_recurrent_ppo_updates_and_resume_contract():
    args = parser().parse_args(["--algo", "recurrent-ppo", "--device", "cpu",
                               "--rp-sequence", "4", "--batch-size", "8", "--ppo-epochs", "2"])
    agent = RecurrentAgent(hidden_size=16, split_head=True)
    obs = torch.randn(8, 2, STATE_DIM)
    starts = torch.zeros(8, 2, dtype=torch.bool)
    starts[0] = True
    h, c, latent, logs, values = [], [], [], [], []
    state = agent.initial_state(2)
    with torch.no_grad():
        for t in range(8):
            h.append(state[0][0]); c.append(state[1][0])
            _, z, lp, _, v, state = agent.step(obs[t], state, starts[t])
            latent.append(z); logs.append(lp); values.append(v)
    data = dict(obs=obs, starts=starts, h=torch.stack(h), c=torch.stack(c),
                latent=torch.stack(latent), logprob=torch.stack(logs), values=torch.stack(values),
                next_values=torch.zeros(8, 2), rewards=torch.randn(8, 2),
                terminated=torch.zeros(8, 2, dtype=torch.bool), truncated=torch.zeros(8, 2, dtype=torch.bool))
    old = {k: v.clone() for k, v in agent.state_dict().items()}
    result = ppo_update(agent, torch.optim.Adam(agent.parameters(), lr=3e-4), data, args, "cpu")
    assert result["optimizer_updates"] == 4
    for prefix in ("lstm", "actor", "critic"):
        assert any(not torch.equal(v, old[k]) for k, v in agent.state_dict().items() if k.startswith(prefix))
    assert np.isfinite(list(result.values())).all()
    assert len(sequence_indices(8, 2, 4)) == 4
    with pytest.raises(ValueError):
        sequence_indices(7, 2, 4)
    config = vars(args)
    validate_restore({"config": config, "optimizer": {}}, config)
    with pytest.raises(ValueError, match="rp_control"):
        validate_restore({"config": config, "optimizer": {}}, dict(config, rp_control="residual"))


def test_pixels_encoder():
    agent = RecurrentAgent("pixels", 16)
    result = agent.step(torch.zeros(2, 4, 64, 64, dtype=torch.uint8),
                        agent.initial_state(2), torch.ones(2, dtype=torch.bool))
    assert result[0].shape == (2, 2) and result[4].shape == (2,)


def test_controller_v2_matches_screened_parameters():
    from racing.recurrent_env import controller
    from scripts.tune_recurrent_controller import candidate_controller
    env = RecurrentRacing(completion_mode="lap95", control_mode="residual",
                          speed_scale=2.2, controller_profile="v2")
    try:
        env.reset(seed=1000011)
        for _ in range(40):
            actual = controller(env.unwrapped, 2.2, "v2")
            expected = candidate_controller(env.unwrapped, 2.2, steering_gain=.7, yaw_gain=.08, preview=.4)
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1] == expected[1]
            env.step(np.zeros(2, np.float32))
    finally:
        env.close()


def test_evaluation_resets_recurrent_memory_per_track():
    from racing.evaluate import evaluate
    torch.set_num_threads(2)
    agent = RecurrentAgent(hidden_size=16, control_mode="residual")
    config = dict(recurrent=True, observation_mode="state", control_mode="residual",
                  completion_mode="lap95", speed_scale=2.2)
    repeated = evaluate(agent, "cpu", config, [99, 100, 99])
    assert repeated[0] == repeated[2]
    assert agent.training


def test_suite_recurrent_uses_valid_sequences(tmp_path, monkeypatch):
    import sys
    from racing import suite
    calls = []
    monkeypatch.setattr(sys, "argv", ["suite", "--algos", "recurrent-ppo", "--preset", "smoke",
                                     "--device", "cpu", "--out", str(tmp_path)])
    monkeypatch.setattr(suite.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    suite.main()
    cmd = calls[0]
    assert cmd[cmd.index("--rp-sequence")+1] == "16"
    assert int(cmd[cmd.index("--steps")+1]) % (2*16) == 0


def test_adaptive_zero_action_matches_controller_and_speed_action_has_authority():
    a = RecurrentRacing(control_mode="adaptive", controller_profile="v2", speed_scale=2.2)
    b = RecurrentRacing(control_mode="residual", controller_profile="v2", speed_scale=2.2)
    try:
        a.reset(seed=23); b.reset(seed=23)
        for _ in range(50):
            xa, ra, ta, ua, _ = a.step(np.zeros(2, np.float32))
            xb, rb, tb, ub, _ = b.step(np.zeros(2, np.float32))
            np.testing.assert_array_equal(xa, xb)
            assert (ra, ta, ua) == (rb, tb, ub)
        for _ in range(10):
            a.step(np.array([0, -1], np.float32))
            b.step(np.zeros(2, np.float32))
        assert a.unwrapped.car.hull.linearVelocity.length < b.unwrapped.car.hull.linearVelocity.length
    finally:
        a.close(); b.close()


def test_pixel_render_skip_is_bit_identical_including_partial_terminal_repeat():
    fast = RecurrentRacing(observation_mode="pixels", repeat=7, completion_mode="lap95")
    normal = Racing(repeat=7, completion_mode="lap95")
    try:
        a, _ = fast.reset(seed=43)
        b, _ = normal.reset(seed=43)
        np.testing.assert_array_equal(a, b)
        for step in range(150):
            command = np.array([.05, .3] if step < 20 else [0., -.2], np.float32)
            a, ra, ta, ua, ia = fast.step(command)
            b, rb, tb, ub, ib = normal.step(controls(command))
            np.testing.assert_array_equal(a, b)
            assert (ra, ta, ua) == (rb, tb, ub)
            if ta or ua:
                assert ia["raw_frames"] == 6  # 1000 % 7
                assert ia["episode_metrics"] == ib["episode_metrics"]
                break
        else:
            pytest.fail("Expected timeout")
    finally:
        fast.close(); normal.close()


def test_finish_assist_preserves_normal_driving_and_cannot_fabricate_finish():
    a = RecurrentRacing(finish_assist=True, completion_mode="lap95")
    b = RecurrentRacing(completion_mode="lap95")
    try:
        a.reset(seed=10); b.reset(seed=10)
        command = np.array([.02, .3], np.float32)
        for _ in range(10):
            xa, ra, ta, ua, _ = a.step(command)
            xb, rb, tb, ub, _ = b.step(command)
            np.testing.assert_array_equal(xa, xb)
            assert (ra, ta, ua) == (rb, tb, ub)
            assert not a.assist_active
        # Sitting at the start with high coverage must still not finish without
        # the original gate's departure/reentry or genuinely all visited tiles.
        a.reset(seed=10)
        base = a.unwrapped
        base.tile_visited_count = len(base.track)-3
        before = base.tile_visited_count
        _, _, term, trunc, info = a.step(np.zeros(2, np.float32))
        assert a.assist_active
        assert not term and not trunc and "episode_metrics" not in info
        assert base.tile_visited_count == before
    finally:
        a.close(); b.close()
