"""Compose already recorded same-track MP4s, keeping original timing."""
import argparse
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",nargs="+",required=True)
    p.add_argument("--labels",nargs="+",required=True,help="ASCII labels for OpenCV font")
    p.add_argument("--out",required=True)
    args=p.parse_args()
    if len(args.inputs)!=len(args.labels):
        p.error("Each video needs one label")
    out=Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    readers=[imageio.get_reader(p) for p in args.inputs]
    iterators=[iter(r) for r in readers]
    rates=[r.get_meta_data()["fps"] for r in readers]
    if not all(abs(x-rates[0])<.01 for x in rates):
        raise ValueError("Videos must share FPS")
    writer=imageio.get_writer(str(out),fps=rates[0],codec="libx264",quality=7,macro_block_size=2)
    ended=[False]*len(readers)
    count=0
    try:
        while True:
            panels=[]
            for i,iterator in enumerate(iterators):
                try:
                    frame=next(iterator) if not ended[i] else None
                except StopIteration:
                    ended[i]=True
                    frame=None
                canvas=np.zeros((440,600,3),np.uint8)
                if frame is not None:
                    canvas[40:]=cv2.resize(frame,(600,400))
                label=args.labels[i]+(" - ended" if ended[i] else "")
                cv2.putText(canvas,label,(12,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                panels.append(canvas)
            if all(ended):
                break
            writer.append_data(np.concatenate(panels,axis=1))
            count+=1
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    print(f"Saved {out}: {count} frames at {rates[0]} FPS")


if __name__ == "__main__":
    main()
