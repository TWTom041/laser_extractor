from cv2 import cv2
import skvideo.io

size = 700
dx = 580
dy = 300
video_data = skvideo.io.vread(r"C:\Users\Tom\Videos\clips\g\g_clips00000109.mov", as_grey=True)
cv2.imshow("", video_data[0, dy:dy+size, dx:dx+size, :])

cv2.waitKey(0)
