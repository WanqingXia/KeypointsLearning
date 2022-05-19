import open3d as o3d
import time
import numpy as np
import os

mesh = o3d.io.read_triangle_mesh("./models/021_bleach_cleanser/textured.obj")

pcd = o3d.geometry.PointCloud()
pcd.points = mesh.vertices

tic = time.time()
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
toc = 1000 * (time.time() - tic)
ints = np.asarray(keypoints.points)
print("ISS Computation took {:.0f} [ms]".format(toc))

mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.5, 0.5, 0.5])
keypoints.paint_uniform_color([1.0, 0.75, 0.0])

depth_img = np.load("./output/021_bleach_cleanser/depth/depth_0")
depth_img = depth_img * (depth_img < 1000000)
cam2world_matrix = np.load("./output/021_bleach_cleanser/matrix/matrix_0")
camera_intrinsics = np.load("./camera_intrinsics")

depth_img2 = np.load("./output/021_bleach_cleanser/depth/depth_1")
depth_img2 = depth_img2 * (depth_img2 < 1000000)
cam2world_matrix2 = np.load("./output/021_bleach_cleanser/matrix/matrix_1")


intrinsics = o3d.camera.PinholeCameraIntrinsic(480, 480, camera_intrinsics[0, 0], camera_intrinsics[1, 1], 239.5, 239.5)
img1 = o3d.geometry.Image(depth_img)
img2 = o3d.geometry.Image(depth_img2)
# pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

cam2world_matrix = cam2world_matrix * [[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]
cam2world_matrix2 = cam2world_matrix2 * [[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]
pcd1 = o3d.geometry.PointCloud.create_from_depth_image(img1, intrinsics, cam2world_matrix)
pcd2 = o3d.geometry.PointCloud.create_from_depth_image(img2, intrinsics, cam2world_matrix2)
pcd1 = pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # flips to it is right side up
pcd2 = pcd2.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
# points_2d1 = np.load("./arr1.npy")
# points_2d2 = np.load("./arr2.npy")
# points_3d1 = np.zeros((len(points_2d1), 4))
# points_3d2 = np.zeros((len(points_2d2), 4))
# for num in range(len(points_2d1)):
#     # transform form depth to 3D coordinate
#     # get from https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/box_dimensioner_multicam/helper_functions.py
#     point3D = np.ones(4)
#     x = int(points_2d1[num, 0])
#     y = int(points_2d1[num, 1])
#     point3D[0] = (x - 239.5) * depth_img[x, y] / camera_intrinsics[0, 0]  # x location of point
#     point3D[1] = (y - 239.5) * depth_img[x, y] / camera_intrinsics[1, 1]  # y location of point
#     point3D[2] = depth_img[x, y]  # z location of point
#     point_world = np.matmul(point3D, np.linalg.inv(cam2world_matrix))
#     points_3d1[num, :] = point3D
#
# for num in range(len(points_2d2)):
#     # transform form depth to 3D coordinate
#     # get from https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/box_dimensioner_multicam/helper_functions.py
#     point3D = np.ones(4)
#     x = int(points_2d2[num, 0])
#     y = int(points_2d2[num, 1])
#     point3D[0] = (x - 239.5) * depth_img2[x, y] / camera_intrinsics[0, 0]  # x location of point
#     point3D[1] = (y - 239.5) * depth_img2[x, y] / camera_intrinsics[1, 1]  # y location of point
#     point3D[2] = depth_img2[x, y]  # z location of point
#     point_world = np.matmul(point3D, np.linalg.inv(cam2world_matrix2))
#     points_3d2[num, :] = point_world

dp_points1 = np.ones((480*480, 4))
num = 0
for x in range(480):
    for y in range(480):
        if depth_img[x, y] > 0:
            point3D = np.ones(4)
            point3D[0] = (x - 239.5) * depth_img[x, y] / camera_intrinsics[0, 0]  # x location of point
            point3D[1] = (y - 239.5) * depth_img[x, y] / camera_intrinsics[1, 1]  # y location of point
            point3D[2] = depth_img[x, y]  # z location of point
            # point3D[0] = point3D[0] + cam2world_matrix[0, 3]
            # point3D[1] = point3D[1] + cam2world_matrix[1, 3]
            # point3D[2] = point3D[2] + cam2world_matrix[2, 3]
            point_world = np.dot(point3D, cam2world_matrix.transpose())
            dp_points1[num, :] = point3D
            num += 1
        else:
            pass
dp_points1 = np.delete(dp_points1, slice(num, 480*480), axis=0)
#
# dp_points2 = np.zeros((480*480, 4))
# num = 0
# for p in range(480):
#     for q in range(480):
#         if depth_img2[p, q] > 0:
#             point3D = np.ones(4)
#             point3D[0] = (p - 239.5) * depth_img2[p, q] / camera_intrinsics[0, 0]  # x location of point
#             point3D[1] = (q - 239.5) * depth_img2[p, q] / camera_intrinsics[1, 1]  # y location of point
#             point3D[2] = depth_img2[p, q]  # z location of point
#             point_world = np.matmul(point3D, np.linalg.inv(cam2world_matrix2))
#             dp_points2[num, :] = point3D
#             num += 1
#         else:
#             pass
# dp_points2 = np.delete(dp_points2, slice(num+1, 480*480), axis=0)
#
#
# points_world1 = o3d.geometry.PointCloud()
# points_world1.points = o3d.utility.Vector3dVector(points_3d1[:, 0:3])
#
# points_world2 = o3d.geometry.PointCloud()
# points_world2.points = o3d.utility.Vector3dVector(points_3d2[:, 0:3])
#
#
depth_points1 = o3d.geometry.PointCloud()
depth_points1.points = o3d.utility.Vector3dVector(dp_points1[:, 0:3])
#
# depth_points2 = o3d.geometry.PointCloud()
# depth_points2.points = o3d.utility.Vector3dVector(dp_points2[:, 0:3])



location = np.zeros((400, 3))
with open("./output/021_bleach_cleanser/positions.txt" , "r") as f:
    for num, line in enumerate(f.readlines()):
        location[num, :] = [float(x) for x in line.split()]

pcl = o3d.geometry.PointCloud()
pcl.points = o3d.utility.Vector3dVector(location)
what = location[0:1, :]
apoint = o3d.geometry.PointCloud()
apoint.points = o3d.utility.Vector3dVector(what)


apoint.paint_uniform_color([1.0, 1.0, 0.0])
pcd1.paint_uniform_color([0.5, 0.0, 0.0])
pcd2.paint_uniform_color([0.0, 0.5, 0.0])
pcl.paint_uniform_color([0.0, 0.0, 0.5])
# points_world1.paint_uniform_color([0.0, 0.0, 0.5])
# points_world2.paint_uniform_color([1.0, 1.0, 0.0])
depth_points1.paint_uniform_color([0.5, 0.0, 0.0])
# depth_points2.paint_uniform_color([0.0, 0.5, 0.0])

xw = np.zeros((100, 4))
yw = np.zeros((100, 4))
zw = np.zeros((100, 4))
xc = np.zeros((100, 4))
yc = np.zeros((100, 4))
zc = np.zeros((100, 4))

# cam2world_matrix = cam2world_matrix[0:3, :]
for i in range(100):
    xw[i, 0] = i * 0.01
    xw[i, 3] = 1
    yw[i, 1] = i * 0.01
    yw[i, 3] = 1
    zw[i, 2] = i * 0.01
    zw[i, 3] = 1
    xc[i, :] = np.dot(xw[i, :], cam2world_matrix.transpose())
    yc[i, :] = np.dot(yw[i, :], cam2world_matrix.transpose())
    zc[i, :] = np.dot(zw[i, :], cam2world_matrix.transpose())


xwp = o3d.geometry.PointCloud()
xwp.points = o3d.utility.Vector3dVector(xw[:, 0:3])
xwp.paint_uniform_color([1.0, 0.0, 0.0])
ywp = o3d.geometry.PointCloud()
ywp.points = o3d.utility.Vector3dVector(yw[:, 0:3])
ywp.paint_uniform_color([0.0, 1.0, 0.0])
zwp = o3d.geometry.PointCloud()
zwp.points = o3d.utility.Vector3dVector(zw[:, 0:3])
zwp.paint_uniform_color([0.0, 0.0, 1.0])

xcp = o3d.geometry.PointCloud()
xcp.points = o3d.utility.Vector3dVector(xc[:, 0:3])
xcp.paint_uniform_color([1.0, 0.0, 0.0])
ycp = o3d.geometry.PointCloud()
ycp.points = o3d.utility.Vector3dVector(yc[:, 0:3])
ycp.paint_uniform_color([0.0, 1.0, 0.0])
zcp = o3d.geometry.PointCloud()
zcp.points = o3d.utility.Vector3dVector(zc[:, 0:3])
zcp.paint_uniform_color([0.0, 0.0, 1.0])



# o3d.visualization.draw_geometries([pcl, points_world1, points_world2, depth_points1, depth_points2])
o3d.visualization.draw_geometries([pcl, xwp, ywp, zwp, pcd1, pcd2])