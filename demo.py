import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

img1 = cv2.imread('./output/021_bleach_cleanser/colour/colour_img_0.jpg')
simg1 = cv2.imread('./output/021_bleach_cleanser/colour/colour_img_0.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
msk1 = np.load('./output/021_bleach_cleanser/mask/mask_0')
img2 = cv2.imread('./output/021_bleach_cleanser/colour/colour_img_1.jpg')
simg2 = cv2.imread('./output/021_bleach_cleanser/colour/colour_img_1.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
msk2 = np.load('./output/021_bleach_cleanser/mask/mask_1')

sift = cv2.xfeatures2d.SIFT_create()
kp1, dp1 = sift.detectAndCompute(gray1, None)
kp2, dp2 = sift.detectAndCompute(gray2, None)


simg1 = cv2.drawKeypoints(img1, kp1, simg1)
simg2 = cv2.drawKeypoints(img2, kp2, simg2)

plt.imshow(simg1)
# plt.show()
plt.imshow(simg2)
# plt.show()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
# Match descriptors.
matches = bf.match(dp1, dp2)
# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
# plt.show()

arr1 = []
for num, keypoint in enumerate(kp1):
    y = int(keypoint.pt[0])
    x = int(keypoint.pt[1])
    if not msk1[x, y]:
        arr1.append(num)
kp1 = np.delete(kp1, arr1).tolist()
dp1 = np.delete(dp1, arr1, axis=0)

arr2 = []
for num, keypoint in enumerate(kp2):
    y = int(keypoint.pt[0])
    x = int(keypoint.pt[1])
    if not msk2[x, y]:
        arr2.append(num)
kp2 = np.delete(kp2, arr2).tolist()
dp2 = np.delete(dp2, arr2, axis=0)

matches = bf.match(dp1, dp2)
matches_f = [match for match in matches if match.distance < 500]
img4 = cv2.drawMatches(img1, kp1, img2, kp2, matches_f, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img4)
# plt.show()

mesh = o3d.io.read_triangle_mesh("./bottle.ply")

pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)

mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.5, 0.5, 0.5])
keypoints.paint_uniform_color([1.0, 0.75, 0.0])
o3d.visualization.draw_geometries([mesh, keypoints])