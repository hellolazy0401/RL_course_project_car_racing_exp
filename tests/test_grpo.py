from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from racing.grpo import group_advantages, trajectory_weights, update_grpo, collect_group
from racing.model import Agent
from racing.envs import VectorRacing


def test_group_relative_scores_and_constant_groups():
    np.testing.assert_allclose(group_advantages([1,2,3]),[-1.2247449,0,1.2247449],rtol=1e-6)
    np.testing.assert_array_equal(group_advantages([7,7,7]),np.zeros(3))
    with pytest.raises(ValueError):group_advantages([1])


def test_equal_trajectory_weight_not_equal_step_weight():
    w=trajectory_weights([2,4])
    assert w[:2].sum()==pytest.approx(.5)
    assert w[2:].sum()==pytest.approx(.5)


def test_no_critic_and_microbatch_equivalent_update():
    torch.set_num_threads(2);torch.manual_seed(9)
    agent=Agent(with_critic=False)
    assert agent.critic is None
    assert not any(k.startswith('critic') for k in agent.state_dict())
    obs=np.random.default_rng(1).integers(0,256,(6,4,64,64),dtype=np.uint8)
    with torch.no_grad():_,z,lp,_=agent.action(torch.as_tensor(obs))
    trajectories=[dict(obs=obs[:2],latent=z[:2].numpy(),logprob=lp[:2].numpy(),advantage=-1.),
                  dict(obs=obs[2:],latent=z[2:].numpy(),logprob=lp[2:].numpy(),advantage=1.)]
    agents=[]
    for micro in [2,6]:
        a=deepcopy(agent);ref=deepcopy(agent).requires_grad_(False)
        args=SimpleNamespace(device='cpu',batch_size=micro,grpo_beta=.01,grpo_epochs=2,clip=.2,entropy_coef=0.,max_grad_norm=1.)
        metrics=update_grpo(a,ref,torch.optim.SGD(a.parameters(),lr=.001),trajectories,args)
        assert metrics['critic_updates']==0 and metrics['actor_updates']==2
        assert all(np.isfinite(v) for v in metrics.values())
        assert all(p.grad is None for p in ref.parameters())
        agents.append(a)
    for a,b in zip(agents[0].parameters(),agents[1].parameters()):
        torch.testing.assert_close(a,b,atol=2e-7,rtol=1e-5)


def test_group_collection_does_not_step_finished_members():
    class AgentStub:
        def action(self,x):
            n=len(x);return torch.zeros(n,3),torch.zeros(n,3),torch.zeros(n),torch.zeros(n)
    class VecStub:
        def reset_same_seed(self,seed):
            assert seed==321
            self.calls=[];self.t=0
            return np.zeros((2,4,64,64),np.uint8)
        def step_active(self,indices,actions):
            self.calls.append(indices.copy());self.t+=1
            return [(np.zeros((4,64,64),np.uint8),float(i+1),i==0 or self.t==2,False,
                     dict(raw_frames=4,episode_metrics={'completed':True})) for i in indices]
    vec=VecStub();trajectories,frames=collect_group(AgentStub(),vec,321,2,'cpu')
    assert vec.calls==[[0,1],[1]]
    assert [len(t['obs']) for t in trajectories]==[1,2]
    assert [t['score'] for t in trajectories]==[1,4]
    assert frames==12
    np.testing.assert_allclose([t['advantage'] for t in trajectories],[-1,1])


def test_real_group_reset_same_seed():
    env=VectorRacing(2,dict(repeat=1000,stack=4,size=64,completion_mode='lap95'),0)
    try:
        obs=env.reset_same_seed(4321)
        np.testing.assert_array_equal(obs[0],obs[1])
        rows=env.step_active([0,1],np.zeros((2,3),np.float32))
        assert all(r[3] for r in rows)
        assert all('terminal_observation' not in r[4] for r in rows)
    finally:env.close()
