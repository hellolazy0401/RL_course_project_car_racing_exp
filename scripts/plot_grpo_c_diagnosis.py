"""Plot raw validation points across consecutive GRPO-C resume directories.

From the project directory, for example::

    python scripts/plot_grpo_c_diagnosis.py --runs runs_grpo_c_v2_check/grpo-c_s0 runs_grpo_c_v2_continue_check/grpo-c_s0 --out runs_grpo_c_v2_check/diagnosis

Pass runs in chronological order. Duplicate step/member records are replaced by
the later run. Every subsequent run must resume a checkpoint in the immediately
previous run, with unchanged training and validation settings. Relative resume
paths are resolved against the current working directory: launch from the same
project directory used for training. The output directory must not contain any
of the output filenames; use a new output directory when adding a continuation.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


PROTOCOL_KEYS = ("algo", "seed", "num_envs", "completion_mode", "eval_episodes",
                 "eval_seed_start", "grpo_c_variant", "batch_size", "actor_lr",
                 "clip", "entropy_coef", "gamma", "max_grad_norm", "grpo_c_clusters",
                 "grpo_c_state_eps", "grpo_c_state_min_samples", "grpo_c_epochs",
                 "grpo_c_beta", "grpo_c_smooth", "grpo_c_diversity",
                 "grpo_c_diversity_threshold", "grpo_c_reference_tau", "grpo_c_init",
                 "grpo_c_state_fallback_clusters", "grpo_c_episodes_per_update",
                 "grpo_c_selection", "grpo_c_completion_bonus", "grpo_c_trainable", "warm_start_sha256", "grpo_c_road_state", "grpo_c_road_scale", "grpo_c_offroad_penalty")
OUTPUT_NAMES = ("selected_validation.csv", "member_validation.csv", "metadata.json",
                "validation_learning_curves.png", "validation_learning_curves.svg")


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Empty or incomplete CSV: {path}")
    return rows


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge_runs(paths):
    protocol = None
    selected, members = {}, {}
    replaced_selected = replaced_members = 0
    for index, path in enumerate(paths):
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        config.setdefault("grpo_c_episodes_per_update", 1)
        config.setdefault("grpo_c_selection", "return")
        config.setdefault("grpo_c_completion_bonus", 0.)
        config.setdefault("grpo_c_trainable", "full")
        config.setdefault("grpo_c_road_state", False)
        config.setdefault("grpo_c_road_scale", .25)
        config.setdefault("grpo_c_offroad_penalty", 0.)
        lineage_path = path / "warm_start.json"
        config["warm_start_sha256"] = (json.loads(lineage_path.read_text(encoding="utf-8"))["sha256"]
                                        if lineage_path.exists() else None)
        if index:
            resume = config.get("resume")
            if not resume:
                raise ValueError(f"{path} is an independent run, not a continuation: config.resume is missing")
            checkpoint = Path(resume).resolve()
            if checkpoint.parent != paths[index - 1].resolve():
                raise ValueError(f"{path} resumes {checkpoint}, not the immediately previous run. "
                                 "Pass runs in chronological order and launch from the training working directory.")
            if checkpoint.suffix != ".pt" or not checkpoint.is_file():
                raise ValueError(f"Cannot confirm resume checkpoint: {checkpoint}")
        missing = [key for key in PROTOCOL_KEYS if key not in config]
        if missing:
            raise ValueError(f"Cannot confirm evaluation protocol in {path}: missing {missing}")
        current = {key: config[key] for key in PROTOCOL_KEYS}
        if current["algo"] != "grpo-c":
            raise ValueError(f"Expected grpo-c, got {current['algo']} in {path}")
        if protocol is None:
            protocol = current
        differences = [key for key in PROTOCOL_KEYS if current[key] != protocol[key]]
        if differences:
            raise ValueError(f"Cannot merge different training/evaluation protocols: {differences} in {path}")
        for filename, target, member_rows in (("validation.csv", selected, False),
                                               ("member_validation.csv", members, True)):
            for row in read_rows(path / filename):
                if int(row["episodes"]) != protocol["eval_episodes"]:
                    raise ValueError(f"Evaluation episode count differs inside {path / filename}")
                step = int(row["steps"])
                key = (step, int(row["member"])) if member_rows else step
                if key in target:
                    if member_rows:
                        replaced_members += 1
                    else:
                        replaced_selected += 1
                row["source_run"] = str(path.resolve())
                target[key] = row
    selected_rows = [selected[key] for key in sorted(selected)]
    member_records = [members[key] for key in sorted(members)]
    expected_ids = set(range(protocol["num_envs"]))
    for row in selected_rows:
        step = int(row["steps"])
        same_step = {member: record for (record_step, member), record in members.items()
                     if record_step == step}
        if set(same_step) != expected_ids:
            raise ValueError(f"Missing ensemble validation records at step {step}")
        selected_member = int(row["selected_member"])
        if selected_member not in same_step:
            raise ValueError(f"Selected member is invalid at step {step}")
        rank_metrics = (("completion_rate", "mean_coverage", "mean_return")
                        if protocol["grpo_c_selection"] == "completion" else ("mean_return",))
        def rank(record):
            return tuple(float(record[key]) for key in rank_metrics)
        if rank(same_step[selected_member]) != max(rank(record) for record in same_step.values()):
            raise ValueError(f"Selected validation member does not maximize {rank_metrics} at step {step}")
        for metric in ("mean_return", "mean_coverage", "completion_rate"):
            if abs(float(row[metric]) - float(same_step[selected_member][metric])) > 1e-5:
                raise ValueError(f"Selected member's {metric} differs at step {step}")
    metadata = dict(protocol=protocol, source_runs=[str(path.resolve()) for path in paths],
                    duplicate_selected_rows_replaced=replaced_selected,
                    duplicate_member_rows_replaced=replaced_members,
                    first_step=int(selected_rows[0]["steps"]),
                    last_step=int(selected_rows[-1]["steps"]),
                    evaluation_points=len(selected_rows), smoothing="none",
                    selection=f"At each checkpoint rank by {protocol['grpo_c_selection']}; all metrics refer to that selected member. This is not a running maximum or held-out test result.",
                    step_accounting=f"Cumulative training decisions summed across {protocol['num_envs']} policies; resumed counters are not added twice.")
    return selected_rows, member_records, metadata


def plot_rows(selected, members, metadata, out):
    protocol = metadata["protocol"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    metrics = (("completion_rate", 100, "Completion rate (%)"),
               ("mean_coverage", 100, "Mean track coverage (%)"),
               ("mean_return", 1, "Mean validation return"))
    colors = plt.get_cmap("tab10")
    for ax, (metric, scale, label) in zip(axes, metrics):
        for member in range(protocol["num_envs"]):
            data = [row for row in members if int(row["member"]) == member]
            ax.plot([int(row["steps"]) for row in data],
                    [scale * float(row[metric]) for row in data],
                    color=colors(member % 10), alpha=.35, linewidth=1.2,
                    marker=".", markersize=4, label=f"Member {member}")
        ax.plot([int(row["steps"]) for row in selected],
                [scale * float(row[metric]) for row in selected],
                color="#182f5a", linewidth=2.2, marker="o", markersize=5,
                label=f"Selected by {protocol['grpo_c_selection']}")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="best", ncol=2)
    axes[0].set_ylim(0, 105)
    axes[1].set_ylim(0, 105)
    budget_label = "GRPO-C fine-tuning decisions" if protocol['warm_start_sha256'] else "Cumulative training decisions"
    axes[-1].set_xlabel(f"{budget_label} (sum across {protocol['num_envs']} policies)")
    axes[-1].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    first_seed = protocol["eval_seed_start"]
    last_seed = first_seed + protocol["eval_episodes"] - 1
    method = "PPO -> GRPO-C" if protocol['warm_start_sha256'] else "GRPO-C"
    if protocol["grpo_c_road_state"]:
        method += " (image + privileged road state)"
    fig.suptitle(f"{method} validation | training seed {protocol['seed']} | {protocol['completion_mode']}\n"
                 f"{protocol['eval_episodes']} fixed tracks (seeds {first_seed}-{last_seed}); raw points, no smoothing",
                 fontsize=12)
    fig.text(.5, .018, "Selection is repeated at every checkpoint; coverage uses the same selected member.\n"
                      "Validation selection is optimistic; these curves are not a held-out test or multi-seed algorithm comparison.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .065, 1, .94))
    fig.savefig(out / "validation_learning_curves.png", dpi=170)
    fig.savefig(out / "validation_learning_curves.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="Consecutive run directories containing config.json and both validation CSVs, earlier first")
    parser.add_argument("--out", type=Path, required=True, help="Directory for plots, merged CSVs, and protocol metadata")
    args = parser.parse_args()
    try:
        selected, members, metadata = merge_runs(args.runs)
        existing = [name for name in OUTPUT_NAMES if (args.out / name).exists()]
        if existing:
            raise ValueError(f"Output files already exist: {existing}; choose a new --out directory")
        args.out.mkdir(parents=True, exist_ok=True)
        write_rows(args.out / OUTPUT_NAMES[0], selected)
        write_rows(args.out / OUTPUT_NAMES[1], members)
        (args.out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        plot_rows(selected, members, metadata, args.out)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f"Merged {len(selected)} validation checkpoints through {metadata['last_step']:,} decisions.")
    print(f"Plots and data: {args.out.resolve()}")


if __name__ == "__main__":
    main()
