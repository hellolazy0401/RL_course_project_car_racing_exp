import numpy as np
import pytest
from racing.envs import Racing, LapFinishGate
from racing.train import parser, validate_restore


def test_start_departure_forward_reentry_and_threshold():
    gate = LapFinishGate(True)
    assert not gate.update(True, True, 96, 100)  # cannot finish merely sitting at start
    assert not gate.update(False, True, 96, 100)
    assert not gate.update(True, False, 96, 100)  # reverse entry
    assert not gate.update(False, True, 95, 100)
    assert not gate.update(True, True, 95, 100)  # strict >95%, like native condition
    assert not gate.update(False, True, 96, 100)
    assert gate.update(True, True, 96, 100)  # visited start must still be eligible


def test_all_tiles_rule_preserved():
    assert LapFinishGate(True).update(False, False, 100, 100)


@pytest.mark.parametrize("mode,finish", [("original", False), ("lap95", True)])
def test_new_finish_is_real_termination_inside_action_repeat(mode, finish):
    env = Racing(completion_mode=mode)
    try:
        env.reset(seed=100)
        base = env.unwrapped
        base.tile_visited_count = len(base.track)-1
        env.finish_gate = LapFinishGate(False)
        env.finish_gate.left_start = True
        env.on_start = lambda: True
        base.car.hull.linearVelocity = tuple(env.start_tangent)
        env.env.step = lambda a: (np.zeros((96,96,3), np.uint8), -.1, False, False, {})
        _, reward, term, trunc, info = env.step(np.zeros(3))
        assert term == finish and not trunc
        assert info["raw_frames"] == (1 if finish else 4)
        assert reward == pytest.approx(-.1 if finish else -.4)
        if finish:
            assert info["episode_metrics"]["completed"]
    finally:
        env.close()


def test_native_failure_is_not_relabelled_as_completion():
    env = Racing(completion_mode="lap95")
    try:
        env.reset(seed=100)
        env.unwrapped.tile_visited_count = len(env.unwrapped.track)
        env.env.step = lambda a: (np.zeros((96,96,3), np.uint8), -100., True, False, {"lap_finished":False})
        _,r,term,_,info = env.step(np.zeros(3))
        assert term and r == -100 and not info["episode_metrics"]["completed"]
    finally:
        env.close()


def test_modes_have_identical_physics_and_reward_before_finish():
    a, b = Racing(), Racing(completion_mode="lap95")
    try:
        x,_=a.reset(seed=100);y,_=b.reset(seed=100)
        np.testing.assert_array_equal(x,y)
        for _ in range(15):
            x,rx,tx,ux,_=a.step(np.array([.1,.4,0],np.float32))
            y,ry,ty,uy,_=b.step(np.array([.1,.4,0],np.float32))
            np.testing.assert_array_equal(x,y)
            assert (rx,tx,ux)==(ry,ty,uy)
    finally:
        a.close();b.close()


def test_restore_rejects_environment_change():
    cfg=vars(parser().parse_args(["--algo","ppo","--completion-mode","lap95"]))
    with pytest.raises(ValueError,match="completion_mode"):
        validate_restore({"config":cfg,"env_config":{}},cfg,branch=True)
