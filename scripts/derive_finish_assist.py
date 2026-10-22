"""Create an explicitly derived learned-policy + physical finish controller."""
import argparse
import hashlib
from pathlib import Path
import torch
from racing.evaluate import evaluate
from racing.recurrent_ppo import agent_from_config, completion_summary, selection_key
from racing.utils import save_json, append_csv, provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    source = Path(args.checkpoint).resolve()
    ck = torch.load(source, map_location="cpu", weights_only=False)
    if ck["config"]["rp_observation"] != "state":
        p.error("Requires a state policy")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    ck["config"] = dict(ck["config"], rp_finish_assist=True, name=out.name)
    ck["env_config"] = dict(ck["env_config"], finish_assist=True)
    ck.pop("optimizer", None)  # Explicitly evaluation-only; no fictitious resume.
    ck["derivation"] = dict(source_checkpoint=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        additional_training_steps=0, change="Physical finish alignment controller; identical learned weights and finish gate")
    save_json(out/"derivation.json", ck["derivation"])
    save_json(out/"config.json", ck["config"])
    save_json(out/"provenance.json", provenance())
    agent = agent_from_config(ck["config"])
    agent.load_state_dict(ck["model"])
    rows = evaluate(agent, "cpu", ck["env_config"], range(1000000,1000010))
    metrics = completion_summary(rows)
    ck["best_key"] = selection_key(metrics)
    for name in ("best.pt", "last.pt"):
        torch.save(ck, out/name)
    save_json(out/"best_metrics.json", dict(steps=ck["steps"], **metrics))
    save_json(out/"finished.json", dict(steps=ck["steps"], train_seconds=ck["train_seconds"],
                                      additional_training_steps=0, derived=True))
    append_csv(out/"validation.csv", dict(steps=ck["steps"],
               **{k:v for k,v in metrics.items() if k != "completion_wilson95"}))
    for row in rows:
        append_csv(out/"validation_episodes.csv", dict(steps=ck["steps"], **row))
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
