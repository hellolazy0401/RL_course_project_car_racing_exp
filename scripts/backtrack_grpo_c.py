"""Validation-only backtracking of aligned, frozen-feature GRPO-C head updates."""
import argparse
import hashlib
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from racing.envs import VectorRacing
from racing.evaluate import evaluate_ensemble
from racing.grpo_c import validation_rank
from racing.model import Agent
from racing.utils import append_csv, save_json, summary


def interpolate_aligned_heads(before, after, alpha):
    if not 0 <= alpha <= 1 or before.keys() != after.keys():
        raise ValueError('Invalid interpolation coefficient or mismatched policy keys')
    last = max(int(k.split('.')[1]) for k in before if k.startswith('actor.'))
    allowed = {f'actor.{last}.weight', f'actor.{last}.bias'}
    output = {}
    for key in before:
        if before[key].shape != after[key].shape:
            raise ValueError(f'Policy shape mismatch: {key}')
        if key not in allowed and not torch.equal(before[key], after[key]):
            raise ValueError(f'Feature/noise alignment mismatch: {key}')
        output[key] = (before[key].clone() if alpha == 0 or key not in allowed else
                       after[key].clone() if alpha == 1 else before[key].lerp(after[key], alpha))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--alphas', nargs='+', type=float, default=[0., .1, .25, .5, 1.])
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=1_000_000)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.episodes < 1 or args.seed_start < 0 or 0. not in args.alphas or 1. not in args.alphas:
        p.error('Positive episodes, nonnegative seed, and both alpha endpoints are required')
    args.alphas = sorted(set(args.alphas))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    baseline = torch.load(args.baseline,map_location=args.device,weights_only=False)
    candidate = torch.load(args.candidate,map_location=args.device,weights_only=False)
    if baseline['env_config'] != candidate['env_config'] or candidate['config']['algo'] != 'grpo-c':
        p.error('Require a GRPO-C candidate and matching environment protocols')
    if baseline["env_config"].get("road_state", False):
        p.error("This backtracking tool supports pixel actor heads only")
    policies = []
    for alpha in args.alphas:
        policy = Agent(with_critic=False).to(args.device)
        policy.load_state_dict(interpolate_aligned_heads(baseline['model'],candidate['model'],alpha))
        policies.append(policy)
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    tick = time.perf_counter()
    vec = VectorRacing(len(policies),baseline['env_config'],0)
    try:
        rows = evaluate_ensemble(policies,vec,args.device,range(args.seed_start,args.seed_start+args.episodes))
    finally:
        vec.close()
    summaries = []
    for alpha, episodes in zip(args.alphas,rows):
        result = dict(alpha=alpha,**summary(episodes))
        summaries.append(result)
        append_csv(out/'validation.csv',result)
        for episode in episodes:
            append_csv(out/'episodes.csv',dict(alpha=alpha,**episode))
        print(f"alpha={alpha}: completion={result['completion_rate']:.1%} coverage={result['mean_coverage']:.1%}",flush=True)
    selected = max(range(len(policies)),key=lambda i: validation_rank(summaries[i],'completion'))
    metadata = dict(method='aligned_head_update_backtracking',selected_alpha=args.alphas[selected],
                    baseline_steps=baseline['steps'],candidate_training_steps=candidate['steps'],
                    baseline=str(Path(args.baseline).resolve()),candidate=str(Path(args.candidate).resolve()),
                    baseline_sha256=hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),
                    candidate_sha256=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),
                    validation_seed_start=args.seed_start,validation_episodes=args.episodes,
                    validation=summaries,device=args.device,evaluation_seconds=time.perf_counter()-tick,
                    selection_criterion='completion, then coverage, then native return')
    checkpoint = dict(model={k:v.cpu() for k,v in policies[selected].state_dict().items()},
                      config=candidate['config'],env_config=candidate['env_config'],
                      steps=baseline['steps'] if args.alphas[selected] == 0 else candidate['steps'],
                      train_seconds=candidate['train_seconds'],
                      warm_start=candidate.get('warm_start'),postprocessing=metadata,inference_only=True)
    torch.save(checkpoint,out/'best_policy.pt')
    save_json(out/'selection.json',metadata)
    print(f"Selected alpha={args.alphas[selected]}; checkpoint is inference-only",flush=True)


if __name__ == '__main__':
    main()
