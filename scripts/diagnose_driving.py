"""Read-only model diagnosis: replay validation seeds; no training changes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import pandas as pd
import torch
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from racing.envs import Racing
from racing.model import Agent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--limit',type=int,default=2000)
    args=p.parse_args()
    run=Path(args.run)
    out=run/('driving_diagnosis_'+args.device)
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    ck=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    agent=Agent().to(args.device);agent.load_state_dict(ck['model']);agent.eval()
    summaries=[]
    for seed in range(1000000,1000005):
        video=run/'validation_video_best'/'videos'/f'track_{seed}.mp4'
        reader=imageio.get_reader(str(video))
        fig,axs=plt.subplots(3,4,figsize=(16,8.5))
        for ax,t in zip(axs.flat,[0,2,4,6,8,10,12,14,16,18,19,19.98]):
            ax.imshow(reader.get_data(round(t*50)));ax.set_title(f'{t:.2f} s');ax.axis('off')
        fig.suptitle(f'Original validation video - track {seed}')
        fig.tight_layout();fig.savefig(out/f'frames_{seed}.jpg',dpi=110);plt.close(fig);reader.close()
        env=Racing(**ck['env_config'])
        # Diagnostic counterfactual ONLY. Formal environment remains 1000 frames.
        env.env._max_episode_steps=args.limit
        obs,_=env.reset(seed=seed)
        base=env.unwrapped
        track=np.array([[t[2],t[3]] for t in base.track])
        original_step=env.env.step
        rows=[]
        total=0.
        def traced_step(action):
            nonlocal total
            result=original_step(action)
            total+=result[1]
            v=np.array(base.car.hull.linearVelocity)
            xy=np.array(base.car.hull.position)
            wheels=sum(bool(w.tiles) for w in base.car.wheels)
            rows.append(dict(frame=len(rows)+1,seconds=(len(rows)+1)/50,
                steer=float(action[0]),gas=float(action[1]),brake=float(action[2]),
                speed=float(np.linalg.norm(v)),wheels_on_road=wheels,
                x=float(xy[0]),y=float(xy[1]),tiles=base.tile_visited_count,
                coverage=base.tile_visited_count/len(base.track),return_raw=total,
                terminated=bool(result[2]),truncated=bool(result[3]),
                lap_finished=bool(result[4].get('lap_finished',False))))
            return result
        env.env.step=traced_step
        with torch.no_grad():
            while True:
                a=agent.action(torch.as_tensor(obs[None],device=args.device),deterministic=True)[0][0].cpu().numpy()
                obs,_,term,trunc,_=env.step(a)
                if term or trunc:break
        df=pd.DataFrame(rows);df.to_csv(out/f'telemetry_{seed}.csv',index=False)
        first=df[df.frame<=1000]
        off=(first.wheels_on_road==0).to_numpy()
        def longest(mask):
            best=cur=0
            for v in mask:
                cur=cur+1 if v else 0;best=max(best,cur)
            return best/50
        no_new=first.tiles.diff().fillna(1).eq(0).to_numpy()
        s=dict(seed=seed,coverage_1000=float(first.coverage.iloc[-1]),return_1000=float(first.return_raw.iloc[-1]),
            mean_speed=float(first.speed.mean()),mean_gas=float(first.gas.mean()),mean_brake=float(first.brake.mean()),
            gas_brake_both_gt_02=float(((first.gas>.2)&(first.brake>.2)).mean()),
            all_wheels_offroad_fraction=float(off.mean()),longest_all_offroad_s=longest(off),
            longest_no_new_tiles_s=longest(no_new),end_speed=float(first.speed.iloc[-1]),
            extended_frames=len(df),extended_coverage=float(df.coverage.iloc[-1]),
            extended_completed=bool(df.lap_finished.iloc[-1]),
            extended_return=float(df.return_raw.iloc[-1]))
        summaries.append(s)
        fig,axs=plt.subplots(4,1,figsize=(11,10),sharex=True)
        axs[0].plot(df.seconds,df.coverage,label='Tile coverage');axs[0].legend()
        axs[1].plot(df.seconds,df.speed,label='Speed (Box2D units/s)');axs[1].legend()
        axs[2].plot(df.seconds,df.gas,label='Gas');axs[2].plot(df.seconds,df.brake,label='Brake');axs[2].legend()
        axs[3].plot(df.seconds,df.steer,label='Steer');axs[3].plot(df.seconds,df.wheels_on_road/4,label='Fraction of wheels on road');axs[3].legend()
        for ax in axs:ax.axvline(20,color='red',ls='--');ax.grid(alpha=.2)
        axs[-1].set_xlabel('Seconds; after red line = diagnostic extension, NOT formal score')
        fig.suptitle(f'Frozen best policy - {seed}');fig.tight_layout();fig.savefig(out/f'trace_{seed}.png',dpi=130);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,7));ax.plot(track[:,0],track[:,1],color='lightgray',lw=10,label='Track centerline')
        ax.plot(first.x,first.y,lw=1,label='First 1000 frames');ax.plot(df.x.iloc[1000:],df.y.iloc[1000:],ls='--',label='Diagnostic extension')
        ax.scatter([first.x.iloc[-1]],[first.y.iloc[-1]],c='red',label='Original time limit')
        ax.set_aspect('equal');ax.legend();fig.tight_layout();fig.savefig(out/f'path_{seed}.png',dpi=130);plt.close(fig)
        env.close();print(json.dumps(s),flush=True)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')


if __name__=='__main__':main()
