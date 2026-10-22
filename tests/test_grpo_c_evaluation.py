import numpy as np
import torch

from racing.evaluate import evaluate_ensemble


class FixedPolicy(torch.nn.Module):
    def action(self, obs, deterministic=False):
        assert deterministic
        return torch.zeros((len(obs), 3)), None, None, None


class StaggeredEpisodes:
    def reset_same_seed(self, seed):
        self.seed = seed
        self.count = [0, 0]
        return np.zeros((2, 1, 2, 2), np.uint8)

    def step_active(self, active, actions):
        assert len(active) == len(actions)
        result = []
        for i in active:
            assert self.count[i] < i + 1  # no stepping an already finished member
            self.count[i] += 1
            done = self.count[i] == i + 1
            metrics = dict(return_raw=float(self.seed+i), coverage=.5+i*.5,
                           completed=i==1, raw_frames=self.count[i])
            result.append((np.zeros((1,2,2),np.uint8), 0., done and i==1,
                           done and i==0, {'episode_metrics':metrics} if done else {}))
        return result


def test_parallel_evaluation_keeps_seeds_members_endings_and_training_modes():
    policies = [FixedPolicy(), FixedPolicy().eval()]
    rng = torch.get_rng_state().clone()
    rows = evaluate_ensemble(policies, StaggeredEpisodes(), 'cpu', [10, 11])
    assert [[r['track_seed'] for r in member] for member in rows] == [[10,11],[10,11]]
    assert [r['return_raw'] for r in rows[0]] == [10.,11.]
    assert [r['return_raw'] for r in rows[1]] == [11.,12.]
    assert all(not r['completed'] for r in rows[0])
    assert all(r['completed'] for r in rows[1])
    assert policies[0].training and not policies[1].training
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
