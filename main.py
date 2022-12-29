import multiprocessing
from cv2 import cv2
from pathlib import Path
import h5py
import pickle
import lzma
import skvideo.io
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from string import ascii_lowercase

DIR = Path(r"C:\Users\Tom\Videos\clips")
no_label = []
now_data = []
now_label = ""

size = 700
dx = 580
dy = 300
dsize = 175


def callback_fn(result):
    global now_label, now_data, no_label
    # print(123)
    try:
        label, vid_data = result
        if label == "no_label":
            no_label.append(vid_data)
        else:
            if now_label != label:
                print("this should not happen, overwrite it anyways")
                now_label = label
                now_data = [vid_data]
            else:
                now_data.append(vid_data)
    except Exception as e:
        print(e, 123)


def load_clip(index, clip, d, sub):
    try:
        video_data = skvideo.io.vread(str(d / clip), as_grey=True)[:, dy:dy + size, dx:dx + size, :]
        video_data = np.array([cv2.resize(src, (dsize, dsize)) for src in video_data])
        if index % 2 == 0:
            return "no_label", video_data
        else:
            return sub, video_data
    except Exception as e:
        print(e, 123)


def main():
    global now_data, now_label
    Path("lzma_compressed/").mkdir(parents=True, exist_ok=True)
    for sub in ascii_lowercase:
        d = DIR / sub
        pool = Pool(6)
        now_data = []
        now_label = sub
        print(sub)
        for index, clip in tqdm(enumerate(d.iterdir())):
            pool.apply_async(load_clip, args=(index, clip, d, sub), callback=callback_fn)
        pool.close()
        pool.join()

        with lzma.open(f"lzma_compressed/{sub}.p.xz", "w") as f:
            print(sub, "now start compressing")
            pickle.dump(now_data, f)
            print(sub, "compression done")

    with lzma.open("lzma_compressed/no_label.p.xz", "w") as f:
        pickle.dump(no_label, f)


if __name__ == "__main__":
    main()
