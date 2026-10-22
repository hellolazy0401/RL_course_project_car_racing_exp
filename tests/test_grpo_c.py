from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from racing.grpo_c import (cluster_states, dbscan_states, discounted_returns,
                           initialize_policy, kmeans_trajectories, merge_episode_batches, policy_groups,
                           state_aware_advantages, update_grpo_c, update_reference)
from racing.grpo_c_reference import ReferenceMixture
from racing.grpo_c import warm_start_policies, validation_rank, completion_reward, configure_trainable
from racing.model import Agent


def test_ppo_warm_start_preserves_actions_and_drops_critic(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(23)
    source = Agent()
    config = dict(repeat=4, stack=4, size=64, completion_mode='lap95')
    path = tmp_path/'ppo.pt'
    torch.save(dict(model=source.state_dict(),config={'algo':'ppo'},env_config=config,
                    steps=800256,frames=3200505,train_seconds=120.),path)
    policies = [Agent(with_critic=False),Agent(with_critic=False)]
    lineage = warm_start_policies(policies,path,config)
    obs = torch.randint(0,256,(3,4,64,64),dtype=torch.uint8)
    with torch.no_grad():
        for policy in policies:
            assert policy.critic is None
            torch.testing.assert_close(policy.action(obs,deterministic=True)[0],
                                       source.action(obs,deterministic=True)[0],rtol=0,atol=0)
            torch.testing.assert_close(policy.logstd,source.logstd,rtol=0,atol=0)
    assert lineage['source_steps']==800256 and len(lineage['sha256'])==64
    with pytest.raises(ValueError,match='environment mismatch'):
        warm_start_policies(policies,path,dict(config,completion_mode='original'))
    transferred = warm_start_policies(policies,path,dict(config,completion_mode='original'),True)
    assert transferred['completion_protocol_transfer']
    assert transferred['source_env_config']['completion_mode']=='lap95'
    assert transferred['target_env_config']['completion_mode']=='original'
    with pytest.raises(ValueError,match='environment mismatch'):
        warm_start_policies(policies,path,dict(config,size=32),True)


def test_completion_selection_prefers_finishes_then_coverage():
    high_score=dict(completion_rate=.1,mean_coverage=.99,mean_return=900.)
    more_finishes=dict(completion_rate=.2,mean_coverage=.8,mean_return=700.)
    more_coverage=dict(completion_rate=.2,mean_coverage=.9,mean_return=650.)
    assert validation_rank(high_score,'return') > validation_rank(more_finishes,'return')
    assert validation_rank(more_coverage,'completion') > validation_rank(more_finishes,'completion') > validation_rank(high_score,'completion')


def test_completion_bonus_never_rewards_timeout_or_coverage_alone():
    assert completion_reward(-.4,False,None,200.) == -.4
    assert completion_reward(-.4,False,{'completed':False,'coverage':.99},200.) == -.4
    assert completion_reward(-100.,True,{'completed':False},200.) == -100.
    assert completion_reward(-.1,True,{'completed':True},200.) == 199.9
    assert completion_reward(-.1,True,{'completed':True},0.) == -.1


def test_head_finetuning_changes_actions_but_preserves_features_and_noise():
    torch.set_num_threads(2)
    policy = Agent(with_critic=False)
    configure_trainable(policy, 'head')
    before = deepcopy(policy.state_dict())
    obs = torch.randint(0,256,(2,4,64,64),dtype=torch.uint8)
    optimizer = torch.optim.Adam(policy.actor_parameters(),lr=1e-4)
    loss = policy.distribution(obs).loc.sum()
    loss.backward()
    optimizer.step()
    last_prefix = f'actor.{len(policy.actor)-1}.'
    assert any(not torch.equal(value,before[key]) for key,value in policy.state_dict().items()
               if key.startswith(last_prefix))
    for key,value in policy.state_dict().items():
        if not key.startswith(last_prefix):
            torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    assert not policy.logstd.requires_grad
    assert all(not p.requires_grad for p in policy.actor[:-1].parameters())


def test_gas_bias_training_preserves_every_other_parameter_and_action_axis():
    torch.set_num_threads(2)
    policy = Agent(with_critic=False)
    configure_trainable(policy, 'gas-bias')
    reference = ReferenceMixture([policy])
    before = deepcopy(policy.state_dict())
    obs = torch.randint(0,256,(2,4,64,64),dtype=torch.uint8)
    with torch.no_grad():
        original = policy.action(obs,deterministic=True)[0].clone()
    optimizer = torch.optim.Adam(policy.actor_parameters(),lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        policy.distribution(obs).loc.sum().backward()
        optimizer.step()
    bias_key = f'actor.{len(policy.actor)-1}.bias'
    for key,value in policy.state_dict().items():
        if key == bias_key:
            assert not torch.equal(value[1],before[key][1])
            torch.testing.assert_close(value[[0,2]],before[key][[0,2]],rtol=0,atol=0)
        else:
            torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    with torch.no_grad():
        updated = policy.action(obs,deterministic=True)[0]
    torch.testing.assert_close(updated[:,[0,2]],original[:,[0,2]],rtol=0,atol=0)
    assert bool((updated[:,1] != original[:,1]).all())
    assert all(p.grad is None for p in reference.parameters())


def test_trajectory_and_state_clusters_are_deterministic():
    features = np.array([[0, 0], [.1, 0], [4, 4], [4.1, 4]], np.float32)
    first = kmeans_trajectories(features, 2)
    second = kmeans_trajectories(features, 2)
    np.testing.assert_array_equal(first, second)
    assert first[0] == first[1] and first[2] == first[3] and first[0] != first[2]
    labels = dbscan_states(np.array([[0], [.05], [2], [2.05], [8]], np.float32), .1, 2)
    assert labels[0] == labels[1] >= 0
    assert labels[2] == labels[3] >= 0 and labels[0] != labels[2]
    assert labels[4] == -1


def test_state_aware_advantage_excludes_own_trajectory_from_baseline():
    assert discounted_returns([1, 2, 3], .5).tolist() == [2.75, 3.5, 3.0]
    zero = np.zeros((4, 64, 64), np.uint8)
    trajectories = [dict(obs=[zero]*3, rewards=[1, 1, 1]),
                    dict(obs=[zero]*3, rewards=[0, 0, 0])]
    info = state_aware_advantages(trajectories, np.array([0, 0]), 1., .5, 2, .2)
    first, second = [t['advantage'] for t in trajectories]
    assert first.mean() > 0 and second.mean() < 0
    # The successful episode is compared with the other episode's zero RTG.
    # It must not subtract half of its own reward signal as the v1 baseline did.
    np.testing.assert_allclose(trajectories[0]['raw_advantage'], [3., 2., 1.])
    np.testing.assert_allclose(trajectories[1]['raw_advantage'], [-2., -2., -2.])
    np.testing.assert_allclose(np.mean(np.concatenate([first, second])), 0, atol=1e-6)
    assert info['state_clusters'] == 1 and info['state_noise_fraction'] == 0
    assert trajectories[0]['clip'] == trajectories[1]['clip'] == .2


def test_raw_state_advantages_do_not_depend_on_policy_group_partition():
    dark = np.zeros((4, 64, 64), np.uint8)
    bright = np.full((4, 64, 64), 255, np.uint8)
    source = [dict(obs=[dark, dark, bright, bright], rewards=rewards)
              for rewards in ([1, 1, 2, 0], [0, 2, 0, 1], [3, 0, 0, 0], [0, 0, 1, 0])]
    together, separated = deepcopy(source), deepcopy(source)
    state_aware_advantages(together, np.array([0, 0, 0, 0]), .99, .65, 2, .2)
    state_aware_advantages(separated, np.array([0, 0, 1, 1]), .99, .65, 2, .2)
    for first, second in zip(together, separated):
        np.testing.assert_allclose(first['raw_advantage'], second['raw_advantage'])
    # A change to policy groups can change normalization, but never state peers.
    assert any(not np.allclose(first['advantage'], second['advantage'])
               for first, second in zip(together, separated))


def test_policy_groups_merge_singletons_without_destroying_valid_groups():
    features = np.array([[0.], [.1], [.2], [20.]], np.float32)
    labels, merges = policy_groups(features, 2)
    assert merges == 1
    assert len(np.unique(labels)) == 1
    paired = np.array([[0.], [.1], [10.], [10.1]], np.float32)
    labels, merges = policy_groups(paired, 2)
    assert merges == 0
    assert np.unique(labels, return_counts=True)[1].tolist() == [2, 2]


def test_state_fallback_breaks_density_chains_without_splitting_identical_views():
    # Nearby states connect the whole range through a DBSCAN density chain.
    features = np.repeat(np.linspace(0, 1, 64, dtype=np.float32)[:, None], 2, axis=1)
    labels, info = cluster_states(features, eps=.65, min_samples=4, fallback_clusters=4)
    assert info['state_dbscan_largest_fraction'] == 1.
    assert info['state_fallback'] == 1
    assert len(np.unique(labels)) == 4
    for label in np.unique(labels):
        assert np.ptp(features[labels == label, 0]) < .5
    identical, _ = cluster_states(np.zeros((32, 2), np.float32), .65, 4, 4)
    assert len(np.unique(identical)) == 1


def test_drive_initialization_lowers_brake_and_preserves_continuous_actions():
    torch.set_num_threads(2)
    policy = Agent(with_critic=False)
    obs = torch.zeros((1, 4, 64, 64), dtype=torch.uint8)
    with torch.no_grad():
        standard = policy.action(obs, deterministic=True)[0]
        standard_scale = policy.distribution(obs).scale.clone()
    initialize_policy(policy, 'drive')
    with torch.no_grad():
        driving = policy.action(obs, deterministic=True)[0]
        distribution = policy.distribution(obs)
        latent = torch.tensor([[-2., -2., -2.], [0., 0., 0.], [2., 2., 2.]])
        actions = policy.action(obs.expand(3, -1, -1, -1), latent=latent)[0]
    assert driving.shape == (1, 3)
    assert float(driving[0, 2]) < .03 < float(standard[0, 2])
    torch.testing.assert_close(driving[:, :2], standard[:, :2])
    torch.testing.assert_close(distribution.scale, standard_scale)
    assert bool((distribution.scale > 0).all())
    assert bool(((actions[:, 0] > -1) & (actions[:, 0] < 1)).all())
    assert bool(((actions[:, 1:] > 0) & (actions[:, 1:] < 1)).all())
    assert bool((actions[1:] > actions[:-1]).all())


def test_joint_policy_update_and_reference_top_member():
    torch.set_num_threads(2)
    torch.manual_seed(10)
    policies = [Agent(with_critic=False), Agent(with_critic=False)]
    reference = ReferenceMixture(policies)
    obs = np.zeros((2, 4, 64, 64), np.uint8)
    trajectories = []
    for policy, advantage in zip(policies, [1., -1.]):
        with torch.no_grad():
            _, latent, logprob, _ = policy.action(torch.as_tensor(obs))
        trajectories.append(dict(obs=list(obs), latent=list(latent.numpy()),
                                 logprob=list(logprob.numpy()), advantage=np.full(2, advantage), clip=.2))
    optimizers = [torch.optim.SGD(policy.actor_parameters(), lr=.001) for policy in policies]
    args = SimpleNamespace(device='cpu', grpo_c_epochs=1, batch_size=1,
                           grpo_c_beta=.01, grpo_c_smooth=.01, grpo_c_diversity=.01,
                           grpo_c_diversity_threshold=.2, max_grad_norm=1., entropy_coef=0.)
    before = [[p.detach().clone() for p in policy.actor_parameters()] for policy in policies]
    metrics = update_grpo_c(policies, reference, optimizers, trajectories, np.array([0, 1]), args)
    assert metrics['actor_updates'] == 2 and metrics['critic_updates'] == 0
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(any(not torch.equal(a, b) for a, b in zip(previous, policy.actor_parameters()))
               for previous, policy in zip(before, policies))
    assert all(p.grad is None for p in reference.parameters())
    update_reference(reference, policies, [0., 1.], 1.)
    torch.testing.assert_close(reference.weights, torch.tensor([0., 1.]))
    for component, policy in zip(reference.components, policies):
        for name, parameter in component.named_parameters():
            torch.testing.assert_close(parameter, dict(policy.named_parameters())[name])


def test_multiepisode_update_masks_resets_and_is_microbatch_invariant():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    policy = Agent(with_critic=False)
    batches = []
    # Constant views inside each episode, different across resets: any smooth
    # gradient would be an erroneous penalty across unrelated episodes.
    for length, pixel, clip in [(2, 0, .1), (3, 255, .3)]:
        obs = np.full((length, 4, 64, 64), pixel, np.uint8)
        with torch.no_grad():
            _, latent, logprob, _ = policy.action(torch.as_tensor(obs))
        batches.append([dict(obs=obs, latent=latent.numpy(), logprob=logprob.numpy(),
                             advantage=np.zeros(length, np.float32), clip=clip)])
    merged = merge_episode_batches(batches)
    np.testing.assert_array_equal(merged[0]['smooth_mask'], [True, False, True, True, False])
    np.testing.assert_allclose(merged[0]['clip'], [.1, .1, .3, .3, .3])
    args = SimpleNamespace(device='cpu', grpo_c_epochs=1, batch_size=2,
                           grpo_c_beta=0., grpo_c_smooth=1., grpo_c_diversity=0.,
                           max_grad_norm=100., entropy_coef=0.)
    reference = ReferenceMixture([policy])
    before = deepcopy(policy.state_dict())
    optimizer = torch.optim.SGD(policy.actor_parameters(), lr=.01)
    update_grpo_c([policy], reference, [optimizer], merged, np.array([0]), args)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    # Nonzero policy loss must agree across computational chunk sizes.
    merged[0]['advantage'] = np.array([1., -1., .2, -.4, .7], np.float32)
    other = deepcopy(policy)
    update_grpo_c([policy], reference, [optimizer], merged, np.array([0]), args)
    args.batch_size = 5
    update_grpo_c([other], reference,
                  [torch.optim.SGD(other.actor_parameters(), lr=.01)], merged, np.array([0]), args)
    for a, b in zip(policy.parameters(), other.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
