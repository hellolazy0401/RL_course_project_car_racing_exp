"""Paired native-reward evaluation of a fixed baseline and fixed candidate.

Choose the candidate on validation tracks before using this script. The script
does not select a policy or change training based on the comparison tracks.
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from racing.evaluate import evaluate, evaluate_ensemble
from racing.envs import VectorRacing
from racing.model import Agent
from racing.utils import append_csv, save_json, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--seed-start', type=int, default=1_000_100)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--parallel', action='store_true', help='Run the two fixed policies concurrently on identical tracks')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.episodes < 1 or args.seed_start < 0:
        p.error('episodes must be positive and seed-start nonnegative')
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    checkpoints = {label: torch.load(path, map_location=args.device, weights_only=False)
                   for label, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    if checkpoints['baseline']['env_config'] != checkpoints['candidate']['env_config']:
        p.error('Both checkpoints must use identical environment protocols')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    seeds = range(args.seed_start, args.seed_start + args.episodes)
    all_rows = {}
    results = {}
    agents = {}
    for label, ck in checkpoints.items():
        agent = Agent(with_critic=ck['config']['algo'] not in ('grpo', 'grpo-c'), road_state=ck["env_config"].get("road_state", False), road_scale=ck["config"].get("grpo_c_road_scale", .25)).to(args.device)
        agent.load_state_dict(ck['model'])
        agents[label] = agent
    if args.parallel:
        tick = time.perf_counter()
        vec = VectorRacing(2, checkpoints['baseline']['env_config'], 0)
        all_rows = {label: [] for label in agents}
        try:
            for seed in seeds:
                paired = evaluate_ensemble(list(agents.values()), vec, args.device, [seed])
                for label, rows in zip(agents, paired):
                    row = rows[0]
                    all_rows[label].append(row)
                    append_csv(out / f'{label}_episodes.csv', row)
                    print(f"{label} track={seed} completed={row['completed']} coverage={row['coverage']:.1%}", flush=True)
        finally:
            vec.close()
        elapsed = {label: time.perf_counter()-tick for label in agents}
    else:
        elapsed = {}
        for label, agent in agents.items():
            tick = time.perf_counter()
            rows = []
            for seed in seeds:
                row = evaluate(agent, args.device, checkpoints[label]['env_config'], [seed])[0]
                rows.append(row)
                append_csv(out / f'{label}_episodes.csv', row)
                print(f"{label} track={seed} completed={row['completed']} coverage={row['coverage']:.1%}", flush=True)
            all_rows[label] = rows
            elapsed[label] = time.perf_counter()-tick
    for label, ck in checkpoints.items():
        results[label] = dict(**summary(all_rows[label]), checkpoint=str(Path(getattr(args,label)).resolve()),
                              checkpoint_steps=ck['steps'], warm_start=ck.get('warm_start'),
                              evaluation_seconds=elapsed[label], shared_parallel_wall_time=args.parallel)
        save_json(out / f'{label}_summary.json', results[label])
    gained = lost = both = neither = 0
    for a, b in zip(all_rows['baseline'], all_rows['candidate']):
        assert a['track_seed'] == b['track_seed']
        gained += int(b['completed'] and not a['completed'])
        lost += int(a['completed'] and not b['completed'])
        both += int(a['completed'] and b['completed'])
        neither += int(not a['completed'] and not b['completed'])
        append_csv(out / 'paired.csv', dict(track_seed=a['track_seed'],
                   baseline_completed=a['completed'], candidate_completed=b['completed'],
                   baseline_coverage=a['coverage'], candidate_coverage=b['coverage'],
                   coverage_delta=b['coverage']-a['coverage']))
    result = dict(**results, device=args.device, deterministic=True, env_config=checkpoints['candidate']['env_config'],
                  seed_start=args.seed_start, episodes=args.episodes,
                  newly_completed=gained, lost_completions=lost, both_completed=both, neither_completed=neither,
                  completion_delta=results['candidate']['completion_rate']-results['baseline']['completion_rate'],
                  coverage_delta=results['candidate']['mean_coverage']-results['baseline']['mean_coverage'])
    save_json(out / 'comparison.json', result)
    print(f"Completion: {results['baseline']['completion_rate']:.1%} -> {results['candidate']['completion_rate']:.1%}; "
          f"new={gained}, lost={lost}", flush=True)


if __name__ == '__main__':
    main()
