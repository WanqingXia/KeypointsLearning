import open3d as o3d
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io
import png
from PIL import Image

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
            if np.sum(points_r[x-1:x+1, y-1:y+1]) > 0:
                img[x-1:x+1, y-1:y+1, :] = img[x-1:x+1, y-1:y+1, :] * [0, 255, 0]

pcd_r.paint_uniform_color([0.5, 0.0, 0.0])
pcd_f.paint_uniform_color([0.0, 0.5, 0.0])
keypoints_r.paint_uniform_color([0.0, 0.0, 0.5])
keypoints_f.paint_uniform_color([0.5, 0.5, 0.0])

plt.imshow(img)
plt.show()
# o3d.visualization.draw_geometries([pcl, points_world1, points_world2, depth_points1, depth_points2])
o3d.visualization.draw_geometries([pcd_f, keypoints_f])