import cv2
import matplotlib.pyplot as plt
# import cv2.ximgproc as xip
import numpy as np
import os
prefix = "./data/hybrid_train"
# Sobel Edge Detection
def sobel_transform(fn):
    img_blur = cv2.imread(os.path.join('./data/synthetics_train/', fn))
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1) # Combined X and Y Sobel Edge Detection
    cv2.imwrite(os.path.join(prefix, "sobel_"+fn), sobelxy)

table = np.array([((i / 255.0) ** 0.2) * 255
		for i in np.arange(0, 256)]).astype("uint8")
# Gamma Correction
def gamma(fn):
    img_blur = cv2.imread(os.path.join('./data/synthetics_train/', fn))
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    out = cv2.LUT(img_blur, table)
    cv2.imwrite(os.path.join(prefix, "gamma_"+fn), out)

from tqdm import tqdm
fns = os.listdir("./data/synthetics_train/")

for fn in tqdm(fns):
    if fn.split(".")[1] != "pkl":
        sobel_transform(fn)
        gamma(fn)