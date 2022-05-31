import time
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io
import png
from PIL import Image
# import pcl
import cv2
import open3d as o3d

"""
This code is currently not working, doesn't do anything

"""

img = plt.imread("/data/Wanqing/YCB_Video_Dataset/data_gen/0000/000001-color.png")

mat = scipy.io.loadmat("/data/Wanqing/YCB_Video_Dataset/data/0000/000001-meta.mat")
cam_int = mat['intrinsic_matrix']
depth_img_r = Image.fromarray(np.array(Image.open("/data/Wanqing/YCB_Video_Dataset/data/0000/000001-depth.png")).astype("uint16"))
depth_img_r = np.asarray(depth_img_r)
depth_img_r = (depth_img_r/mat["factor_depth"]).astype(np.float32)
depth_img_f = Image.fromarray(np.array(Image.open("/data/Wanqing/YCB_Video_Dataset/data_gen/0000/000001-depth.png")).astype("uint16"))
depth_img_f = np.asarray(depth_img_f)
depth_img_f = (depth_img_f/mat["factor_depth"]).astype(np.float32)

intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, cam_int[0, 0], cam_int[1, 1], cam_int[0, 2], cam_int[1, 2])
img_r = o3d.geometry.Image(depth_img_r)
img_f = o3d.geometry.Image(depth_img_f)


pcd_r = o3d.geometry.PointCloud.create_from_depth_image(img_r, intrinsics)
pcd_f = o3d.geometry.PointCloud.create_from_depth_image(img_f, intrinsics)


keypoints_r = o3d.geometry.keypoint.compute_iss_keypoints(pcd_r)
keypoints_f = o3d.geometry.keypoint.compute_iss_keypoints(pcd_f)

kp_r = np.asarray(keypoints_r.points)
kp_f = np.asarray(keypoints_f.points)

points_r = np.zeros((480, 640, 1))
points_f = np.zeros((480, 640, 1))
for kp in kp_r:
    x = int((kp[0] * cam_int[0, 0] / (np.linalg.norm(kp))) + cam_int[0, 2])
    y = int((kp[1] * cam_int[1, 1] / (np.linalg.norm(kp))) + cam_int[1, 2])
    points_r[y, x] = 1

for kp in kp_f:
    x = int((kp[0] * cam_int[0, 0] / (np.linalg.norm(kp))) + cam_int[0, 2])
    y = int((kp[1] * cam_int[1, 1] / (np.linalg.norm(kp))) + cam_int[1, 2])
    points_f[y, x] = 1

for x in range(points_f.shape[0]):
    for y in range(points_f.shape[1]):
        if points_f[x, y] == 1:
            if np.sum(points_r[x-2:x+2, y-2:y+2]) > 0:
                img[x-1:x+1, y-1:y+1, :] = img[x-1:x+1, y-1:y+1, :] * [0, 255, 0]

pcd_r.paint_uniform_color([0.5, 0.0, 0.0])
pcd_f.paint_uniform_color([0.0, 0.5, 0.0])
keypoints_r.paint_uniform_color([0.0, 0.0, 0.5])
keypoints_f.paint_uniform_color([1.0, 0.0, 0.0])

# points_harris = np.delete(points_harris, slice(num), axis=0)
# what = o3d.geometry.PointCloud()
# what.points = o3d.utility.Vector3dVector(points_harris)
# ori = o3d.io.read_point_cloud("scene.pcd")
# pcd_f.paint_uniform_color([0.0, 0.5, 0.0])
# o3d.io.write_point_cloud("scene.pcd", pcd_r)

# plt.imshow(img)
# plt.show()
# o3d.visualization.draw_geometries([keypoints_r])

depmin = min(i for i in depth_img_r.flatten() if i > 0)
depmax = np.max(depth_img_r)
for x in range(480):
    for y in range(640):
        if depth_img_r[x,y]>0:
            depth_img_r[x,y] = ((depth_img_r[x,y] - depmin) * (255 / (depmax - depmin)))

depmin = min(i for i in depth_img_f.flatten() if i > 0)
depmax = np.max(depth_img_f)
for x in range(480):
    for y in range(640):
        if depth_img_f[x, y] > 0:
            depth_img_f[x, y] = ((depth_img_f[x, y] - depmin) * (255 / (depmax - depmin)))

depth_img_r = np.round(depth_img_r).astype(dtype=np.uint8)
depth_img_f = np.round(depth_img_f).astype(dtype=np.uint8)
fast = cv2.FastFeatureDetector_create()
drawimg_r = cv2.imread("/data/Wanqing/YCB_Video_Dataset/data/0000/000001-color.png")
drawimg_f = cv2.imread("/data/Wanqing/YCB_Video_Dataset/data_gen/0000/000001-color.png")
depth_img_r = np.repeat(depth_img_r[..., np.newaxis], 3, axis=2)
depth_img_f = np.repeat(depth_img_f[..., np.newaxis], 3, axis=2)
kp_r = fast.detect(depth_img_r,None)
kp_f = fast.detect(depth_img_f,None)
arr = []
for k in kp_f:
    for p in kp_r:
        if p.pt[0]-5 < k.pt[0] and p.pt[0]+ 5 > k.pt[0]:
            if p.pt[1]-5 < k.pt[1] and p.pt[1]+ 5 > k.pt[1]:
                arr.append(k)
img1 = cv2.drawKeypoints(drawimg_r, arr, None, color=(255, 0, 0))

plt.imshow(img1)
plt.show()
