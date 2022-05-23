import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
# import open3d as o3d


# sift
sift = cv2.SIFT_create()

# matcher
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)


color_r = "/data/Wanqing/YCB_Video_Dataset/data/0000/000001-color.png"
color_f = "/data/Wanqing/YCB_Video_Dataset/data_gen/0000/000001-color.png"
depth_r = "/data/Wanqing/YCB_Video_Dataset/data/0000/000001-depth.png"
depth_f = "/data/Wanqing/YCB_Video_Dataset/data_gen/0000/000001-depth.png"

img_r = cv2.imread(color_r)
img_f = cv2.imread(color_f)
dep_r = cv2.imread(depth_r)
dep_f = cv2.imread(depth_f)
img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
keypoints_r, descriptors_r = sift.detectAndCompute(img_r,None)
keypoints_f, descriptors_f = sift.detectAndCompute(img_f,None)
dep_r = dep_r[:, :, 0]
dep_f = dep_f[:, :, 0]

arr_r = []
for num_r, keypoint_r in enumerate(keypoints_r):
    y = int(keypoint_r.pt[0])
    x = int(keypoint_r.pt[1])
    if dep_f[x, y] == 0:
        arr_r.append(num_r)
keypoints_r = np.delete(keypoints_r, arr_r, axis=0)
descriptors_r = np.delete(descriptors_r, arr_r, axis=0)

arr_f = []
for num_f, keypoint_f in enumerate(keypoints_f):
    y = int(keypoint_f.pt[0])
    x = int(keypoint_f.pt[1])
    if dep_f[x, y] == 0:
        arr_f.append(num_f)
keypoints_f = np.delete(keypoints_f, arr_f, axis=0)
descriptors_f = np.delete(descriptors_f, arr_f, axis=0)

matches_f = []
matches = bf.match(descriptors_r, descriptors_f)
# matches_f = [match for match in matches if match.distance < 500]
for match in matches:
    point_r = keypoints_r[match.queryIdx]
    point_f = keypoints_f[match.trainIdx]
    if int(point_r.pt[0])-2 <= int(point_f.pt[0]) <= int(point_r.pt[0])+2:
        if int(point_r.pt[1])-2 <= int(point_f.pt[1]) <= int(point_r.pt[1])+2:
            matches_f.append(match)

template_image = cv2.cvtColor(cv2.imread(color_r), cv2.COLOR_BGR2RGB)
target_image = cv2.cvtColor(cv2.imread(color_f), cv2.COLOR_BGR2RGB)
outimage = np.zeros((2000, 2000, 3))
results = cv2.drawMatches(template_image, keypoints_r, target_image, keypoints_f, matches_f, target_image, flags=2)
plt.imshow(results)
plt.show()


