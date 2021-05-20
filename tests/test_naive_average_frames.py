import numpy as np
import cv2
import glob
import imageio 
from skimage.util import random_noise
from motioncompensation import *

dir2imgs = "/home/abel/Desktop/repo_history/k_dvdnet/data/DAVIS/JPEGImages/480p/car-race"
list2imgs = sorted(glob.glob(dir2imgs+"/*jpg"))

print("Total image number=", len(list2imgs))
img0 = imageio.imread(list2imgs[0])/255.0
img1 = imageio.imread(list2imgs[1])/255.0
img2 = imageio.imread(list2imgs[2])/255.0

sigma = 0.1
img0 = random_noise(img0, var=sigma**2)
img1 = random_noise(img1, var=sigma**2)
img2 = random_noise(img2, var=sigma**2)

warped_img0 = align_frames(img0, img1, mc_alg='DeepFlow')/255.0
warped_img2 = align_frames(img2, img1, mc_alg='DeepFlow')/255.0

average3 = (warped_img0 + img1 + warped_img2)/3

imageio.imwrite("img1_vs_average3.jpg", np.hstack((img1, average3)))