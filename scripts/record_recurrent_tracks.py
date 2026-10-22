"""Record every held-out track, preserving failures and checking prior results."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import multiprocessing as mp
from pathlib import Path
import sys
import zipfile


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def initialize(checkpoint):
    import cv2
    import torch
    from racing.recurrent_ppo import agent_from_config
    global agent, env_config
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = agent_from_config(ck["config"])
    agent.load_state_dict(ck["model"])
    env_config = ck["env_config"]


def record(seed, output, reference):
    import cv2
    import math
    from racing.evaluate import evaluate
    row = evaluate(agent, "cpu", env_config, [seed], output / "videos", 1)[0]
    for key in ("raw_frames", "decisions"):
        if row[key] != int(reference[key]):
            raise AssertionError(f"Track {seed}: changed {key}")
    for key in ("completed", "terminated", "truncated", "score_ge_900"):
        if bool(row[key]) != (reference[key] == "True"):
            raise AssertionError(f"Track {seed}: changed {key}")
    for key in ("coverage", "return_raw"):
        if not math.isclose(row[key], float(reference[key]), rel_tol=1e-10, abs_tol=1e-8):
            raise AssertionError(f"Track {seed}: changed {key}")
    video = output / "videos" / f"track_{seed}.mp4"
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    if fps != 50 or decoded != row["raw_frames"]:
        raise AssertionError(f"Track {seed}: video frames {decoded}/{row['raw_frames']}, fps={fps}")
    return dict(**row, seconds=row["raw_frames"] / fps, video=f"videos/{video.name}",
                video_frames=decoded, video_fps=fps, width=width, height=height,
                bytes=video.stat().st_size, sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                matches_original_evaluation=True)


def gallery(output, rows, summary):
    cards = []
    for row in rows:
        status = "完赛" if row["completed"] else "未完赛 · 超时" if row["truncated"] else "未完赛"
        cards.append(f'''<article data-seed="{row['track_seed']}" data-completed="{str(row['completed']).lower()}">
<h2>赛道 {row['track_seed']} <span class="{'ok' if row['completed'] else 'fail'}">{status}</span></h2>
<video controls preload="none" width="600" height="400" src="{row['video']}"></video>
<p>{row['seconds']:.2f} 秒 · 覆盖率 {row['coverage']:.2%} · {row['video_frames']} 帧 · 50 FPS
<a href="{row['video']}" download>下载 MP4</a></p></article>''')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>100 条赛道完整录像</title><style>
body{{font:16px/1.6 system-ui,sans-serif;margin:30px auto;padding:0 24px;max-width:1280px;background:#f5f7fa;color:#243444}}
h1{{margin-bottom:8px}}h2{{font-size:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,430px),1fr));gap:20px}}
article{{background:white;padding:16px;border-radius:12px;box-shadow:0 2px 8px #20304012}}article[hidden]{{display:none}}video{{width:100%;height:auto;background:#111}}
.ok{{color:#138263}}.fail{{color:#b43d31}}span{{font-size:14px;float:right}}a{{color:#1765a1}}input,select{{font:inherit;padding:8px;margin:8px 12px 20px 0;max-width:100%}}
.summary{{padding:18px;background:#e5f2ee;border-radius:12px}}p{{margin:8px 0}}</style>
<h1>100 条新赛道 · 完整回合录像</h1>
<div class="summary"><b>{sum(r['completed'] for r in rows)}/{len(rows)} 条完赛</b>，成功平均用时 {summary['mean_finish_seconds']:.2f} 秒。
使用 Recurrent PPO 终点辅助候选，赛道编号 {rows[0]['track_seed']}–{rows[-1]['track_seed']}。
每条录像均为完整物理帧，未加速、未跳帧；未完赛回合如实保留。</div>
<p>这是 2026-10-20 实验中已测试的同一组赛道，2026-10-21 补录视频，并逐项核对原始结果。</p>
<p><a href="episodes.csv">全部回合与录像索引</a> · <a href="summary.json">汇总与验证信息</a> · <a href="model_reference.json">模型来源</a></p>
<label>赛道编号 <input id="search" placeholder="例如 4000010" inputmode="numeric"></label>
<label>显示 <select id="filter"><option value="all">全部</option><option value="true">仅完赛</option><option value="false">仅未完赛</option></select></label>
<div class="grid">{''.join(cards)}</div>
<script>const search=document.querySelector('#search'),filter=document.querySelector('#filter');
function update(){{document.querySelectorAll('article').forEach(card=>card.hidden=!card.dataset.seed.includes(search.value.trim())||(filter.value!=='all'&&card.dataset.completed!==filter.value));}}
search.addEventListener('input',update);filter.addEventListener('change',update);
document.querySelectorAll('video').forEach(video=>video.addEventListener('play',()=>document.querySelectorAll('video').forEach(other=>{{if(other!==video)other.pause();}})));
</script></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True, help="Original episodes.csv for these exact tracks")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    checkpoint, reference, output = map(lambda s: Path(s).resolve(), (args.checkpoint, args.reference, args.out))
    if args.workers < 1:
        parser.error("workers must be positive")
    with reference.open(encoding="utf-8-sig", newline="") as file:
        originals = list(csv.DictReader(file))
    seeds = [int(row["track_seed"]) for row in originals]
    if len(set(seeds)) != len(seeds) or not seeds:
        parser.error("reference must contain unique track seeds")
    output.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    save(output / "model_reference.json", dict(checkpoint=str(checkpoint), sha256=source_hash,
        reference_episodes=str(reference), reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        model="Recurrent PPO finish-assist candidate", recorded_date="2026-10-21",
        inference="deterministic CPU, one recurrent state per episode", argv=sys.argv))
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=initialize, initargs=(checkpoint,)) as pool:
            pending = [pool.submit(record, seed, output, original) for seed, original in zip(seeds, originals)]
            for future in as_completed(pending):
                row = future.result()
                rows.append(row)
                with (output / "progress.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row) + "\n")
                save(output / "status.json", dict(status="recording", videos_done=len(rows), total=len(seeds),
                     updated_at=datetime.now(timezone.utc).isoformat()))
                print(f"Recorded {len(rows)}/{len(seeds)}: track {row['track_seed']}, completed={row['completed']}, {row['seconds']:.2f}s", flush=True)
        rows.sort(key=lambda r: r["track_seed"])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_hash
        from racing.recurrent_ppo import completion_summary
        summary = dict(**completion_summary(rows), videos=len(rows), all_videos_fully_decoded=True,
                       all_video_frame_counts_match=True, all_results_match_original=True,
                       checkpoint_sha256=source_hash, video_fps=50, total_playback_seconds=sum(r["seconds"] for r in rows))
        with (output / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        save(output / "summary.json", summary)
        gallery(output, rows, summary)
        save(output / "status.json", dict(status="complete", videos_done=len(rows), total=len(seeds)))
        archive = output.with_suffix(".zip")
        if archive.exists():
            raise FileExistsError(archive)
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zipped:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    zipped.write(path, output.name + "/" + path.relative_to(output).as_posix())
            zipped.write(Path(__file__), output.name + "/record_recurrent_tracks.py")
        with zipfile.ZipFile(archive) as zipped:
            assert zipped.testzip() is None
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        archive.with_suffix(".zip.sha256").write_text(digest + "  " + archive.name + "\n", encoding="ascii")
        print(json.dumps(dict(archive=str(archive), summary=summary)), flush=True)
    except BaseException as error:
        save(output / "status.json", dict(status="failed", videos_done=len(rows), total=len(seeds), error=repr(error)))
        raise


if __name__ == "__main__":
    main()
