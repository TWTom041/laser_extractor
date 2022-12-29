from cv2 import cv2
import pickle
import lzma

with lzma.open(r"lzma_compressed/a.p.xz", "r") as f:
    a = pickle.load(f)

cv2.imshow("", a[0][0])
cv2.waitKey(0)
