import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from argparse import ArgumentParser
import glob
import shutil
import pickle
# prefix = "./data/hybrid_test"
# Sobel Edge Detection
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--source_dir",
        help="Directory to original training dataset.",
        default="../data/synthetics_train/"
    )
    parser.add_argument(
        "--dest_dir",
        help="Dirrectory to destination.",
        default="../data/test_gen_hybrid/"
    )   

    args = parser.parse_args()

    return args

def sobel_transform(prefix, fn):
    """
    input: image filename
    feature: Apply sobel edge detector on input image and save to destination directory
    """
    img_blur = cv2.imread(fn)
    gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1) # Combined X and Y Sobel Edge Detection
    cv2.imwrite(os.path.join(prefix, "sobel_"+fn.split("/")[3]), sobelxy)

# Gamma Correction
def gamma(prefix, fn):
    """
    input: image filename
    feature: Apply gamma correction on input image and save to destination directory
    """
    # img_blur = cv2.imread(os.path.join('../data/synthetics_train/', fn))
    img_blur = cv2.imread(fn)
    # gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    out = cv2.LUT(img_blur, table)
    cv2.imwrite(os.path.join(prefix, "gamma_"+fn.split("/")[3]), out)


if __name__ == '__main__':
    from tqdm import tqdm
    args = parse_args()
    if not os.path.isdir(args.dest_dir):
        os.mkdir(args.dest_dir)
    fns = os.listdir(args.source_dir)
    # look up table for gamma correction
    table = np.array([((i / 255.0) ** 0.2) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    
    # Move original training data to destination directory
    # Start transform images and move to destination directory
    prefix = args.dest_dir
    """
    
    for jpgfile in glob.iglob(os.path.join(args.source_dir, "*.jpg")):
        shutil.copy(jpgfile, args.dest_dir)

    for fn in tqdm(fns):
        if fn.split(".")[1] != "pkl":
            org_path = os.path.join(args.source_dir, fn)
            sobel_transform(prefix, org_path)
            gamma(prefix, org_path)
            # os.rename(org_path, os.path.join(args.dest_dir, fn))
    """
    with open(os.path.join(args.source_dir, "annot.pkl") , 'rb') as f:
        imgs, landmarks = pickle.load(f)
    sobel_imgs = ["sobel_"+fn for fn in imgs]
    gamma_imgs = ["gamma"+fn for fn in imgs]
    all_imgs = [*imgs, *sobel_imgs, *gamma_imgs]
    all_landmarks = [*landmarks, *landmarks, *landmarks]

    with open(os.path.join(args.dest_dir, "annot.pkl"), 'wb') as f:
        pickle.dump((all_imgs, all_landmarks), f)

    with open(os.path.join(args.dest_dir, "annot.pkl") , 'rb') as f:
        imgs, landmarks = pickle.load(f)

    imgs, landmarks = np.array(imgs), np.array(landmarks)
    assert imgs.shape == (299268, )
    assert landmarks.shape == (299268, 68, 2)
    # print(imgs.shape)
    # print(landmarks.shape)
    