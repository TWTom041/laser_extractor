from cv2 import cv2
from pathlib import Path
import h5py
import pickle
import lzma
import skvideo.io
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

DIR = Path(r"C:\Users\Tom\Videos\clips")
no_label = []

size = 700
dx = 580
dy = 300
dsize = 175

for sub in "qwertyuiopasdfghjklzxcvbnm":
    d = DIR / sub
    now_data = []
    print(sub)
    for index, clip in tqdm(enumerate(d.iterdir())):
        video_data = skvideo.io.vread(str(d / clip), as_grey=True)[:, dy:dy + size, dx:dx + size, :]
        video_data = np.array([cv2.resize(src, (dsize, dsize)) for src in video_data])
        if index % 2 == 0:
            no_label.append(video_data)
        else:
            now_data.append(video_data)

    with lzma.open(f"lzma_compressed/{sub}.p.xz", "w") as f:
        pickle.dump(now_data, f)

with lzma.open("lzma_compressed/no_label.p.xz", "w") as f:
    pickle.dump(no_label, f)
