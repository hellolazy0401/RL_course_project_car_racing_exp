from types import SimpleNamespace
import numpy as np
import pytest
import torch
from racing.model import Agent, gae
from racing.train import full_step, update
from racing.envs import Racing, VectorRacing


def test_time_limit_bootstrap_and_no_reset_leakage():
    # Two transitions, first time-limited, second belongs to a reset episode.
    rew=torch.tensor([[1.], [100.]])
    val=torch.tensor([[2.], [3.]])
    nv=torch.tensor([[10.], [20.]])
    term=torch.tensor([[False], [True]])
    trunc=torch.tensor([[True], [False]])
    adv,ret=gae(rew,val,nv,term,trunc,.9,.95)
    torch.testing.assert_close(adv,torch.tensor([[8.], [97.]]))
    torch.testing.assert_close(ret,torch.tensor([[10.], [100.]]))


def test_trace_crosses_only_nonterminal_transitions():
    rew=torch.tensor([[1.], [2.]])
    z=torch.zeros_like(rew); no=torch.zeros_like(rew,dtype=torch.bool)
    adv,_=gae(rew,z,z,no,no,.9,1.)
    torch.testing.assert_close(adv,torch.tensor([[2.8],[2.]]))


def test_action_bounds_and_stored_logprob():
    torch.set_num_threads(2)
    agent=Agent()
    obs=torch.zeros((8,4,64,64),dtype=torch.uint8)
    action,z,lp,_=agent.action(obs)
    assert (action[:,0].abs()<=1).all()
    assert ((action[:,1:]>=0)&(action[:,1:]<=1)).all()
    torch.testing.assert_close(lp,agent.action(obs,z)[2])


def test_microbatch_equals_full_batch_update():
    torch.manual_seed(0)
    a=torch.nn.Linear(3,1); b=torch.nn.Linear(3,1);b.load_state_dict(a.state_dict())
    x=torch.randn(7,3);y=torch.randn(7,1)
    oa=torch.optim.SGD(a.parameters(),lr=.01); ob=torch.optim.SGD(b.parameters(),lr=.01)
    full_step(lambda i:(a(x[i])-y[i]).square().mean(),list(a.parameters()),oa,7,3,0)
    full_step(lambda i:(b(x[i])-y[i]).square().mean(),list(b.parameters()),ob,7,7,0)
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa,pb)


@pytest.mark.parametrize("algo,k,expected",[("a2c",50,1),("vpg50",3,3),("ppo",50,4)])
def test_optimizer_counts_and_finite_updates(algo,k,expected):
    torch.set_num_threads(2)
    torch.manual_seed(1)
    agent=Agent();obs=np.zeros((4,4,64,64),np.uint8)
    with torch.no_grad():
        _,z,lp,_=agent.action(torch.as_tensor(obs))
    ao=torch.optim.Adam(agent.actor_parameters(),lr=3e-4)
    co=torch.optim.Adam(agent.critic.parameters(),lr=3e-4)
    args=SimpleNamespace(algo=algo,value_steps=k,normalize_adv=True,batch_size=2,
        critic_grad_norm=0.,max_grad_norm=1.,entropy_coef=0.,ppo_epochs=2,clip=.2)
    m=update(agent,ao,co,(obs,z,lp,torch.arange(4.).float(),torch.ones(4)),args,"cpu")
    assert m["critic_updates"]==expected
    assert m["actor_updates"]==(4 if algo=="ppo" else 1)
    assert all(np.isfinite(v) for v in m.values())


def test_critic_update_cannot_change_actor():
    torch.set_num_threads(2)
    agent=Agent();x=torch.zeros((2,4,64,64),dtype=torch.uint8)
    before=[p.detach().clone() for p in agent.actor_parameters()]
    opt=torch.optim.Adam(agent.critic.parameters())
    full_step(lambda i:(agent.value(x[i])-1).square().mean(),list(agent.critic.parameters()),opt,2,1,0)
    for p,q in zip(agent.actor_parameters(),before):
        torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_real_environment_repeat_and_seed():
    env=Racing()
    try:
        a,_=env.reset(seed=100)
        b,_=env.reset(seed=100)
        np.testing.assert_array_equal(a,b)
        assert a.shape==(4,64,64) and a.dtype==np.uint8
        total=0.
        for _ in range(250):
            obs,r,term,trunc,info=env.step(np.zeros(3,np.float32))
            total+=r
            if term or trunc:
                break
        assert trunc and not term
        assert info["episode_metrics"]["raw_frames"]==1000
        assert info["episode_metrics"]["return_raw"]==pytest.approx(total)
        assert not info["episode_metrics"]["completed"]
    finally:
        env.close()


def test_vector_autoreset_preserves_terminal_observation():
    env=VectorRacing(1,dict(repeat=1000,stack=4,size=64),0)
    try:
        env.reset()
        obs,rew,term,trunc,info=env.step(np.zeros((1,3),np.float32))
        assert trunc[0]
        assert info[0]["terminal_observation"].shape==(4,64,64)
        assert obs.shape==(1,4,64,64)
        assert info[0]["raw_frames"]==1000
    finally:
        env.close()
