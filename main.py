from cv2 import cv2
from pathlib import Path
import h5py
import skvideo.io
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

DIR = Path(r"C:\Users\Tom\Videos\clips")
no_label = []
f = h5py.File("data_cut.hdf5", "w")

size = 700
dx = 580
dy = 300

for sub in "qwertyuiopasdfghjklzxcvbnm":
    d = DIR / sub
    now_data = []
    print(sub)
    for index, clip in tqdm(enumerate(d.iterdir())):
        video_data = skvideo.io.vread(str(d / clip), as_grey=True)[0, dy:dy+size, dx:dx+size, :]
        if index % 2 == 0:
            no_label.append(video_data)
        else:
            now_data.append(video_data)

    f.create_dataset(sub, data=now_data)

f.create_dataset("no_label", np.array(no_label))



