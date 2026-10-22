import csv
import json
from racing.sac_experiment import metrics


def write_segment(path, vals, finished=True):
    path.mkdir(parents=True, exist_ok=True)
    for name, data in [("config.json", {"sac_arch": "gru", "seed": 0}),
                       ("model_info.json", {"architecture": "gru", "parameters": 10, "context_length": 8})]:
        (path/name).write_text(json.dumps(data))
    if finished:
        (path/"finished.json").write_text(json.dumps({"steps": vals[-1][0], "train_seconds": 3600}))
    with (path/"validation.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["steps", "mean_return", "completion_rate"])
        for step, reward in vals:
            w.writerow([step, reward, 0.])
    (path/"best.pt").write_text("fixture")


def test_study_aggregates_resume_without_losing_previous_best(tmp_path):
    write_segment(tmp_path, [(0, -30), (25000, 200), (50000, 100)], finished=False)
    write_segment(tmp_path/"segment_001", [(50000, 100), (75000, 150), (100000, 50)])
    result = metrics(tmp_path)
    assert result["best_validation"] == 200
    assert result["best_checkpoint"] == str(tmp_path/"best.pt")
    assert result["best_steps"] == 25000
    assert result["last_three_validation"] == 100
    assert result["total_steps"] == 100000
    assert result["architecture"] == "gru"
