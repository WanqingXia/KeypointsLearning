import blenderproc as bproc
import numpy as np
import os
import cv2
import h5py
import shutil
import scipy.io

# functions for debug, only uncomment this when debugging is needed
# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()


mat = scipy.io.loadmat('000001-meta.mat')

bproc.init()

# load the objects into the scene
obj1 = bproc.loader.load_obj("./models/009_gelatin_box/textured.obj")
obj2 = bproc.loader.load_obj("./models/025_mug/textured.obj")
obj3 = bproc.loader.load_obj("./models/061_foam_brick/textured.obj")
# define a light and set its location and energy level
light = bproc.types.Light()
light.set_location(mat['rotation_translation_matrix'][:, 3].flatten())
light.set_type("POINT")
light.set_energy(100)

cam2world = np.asarray(mat['rotation_translation_matrix'])

# Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
cam2world = np.vstack((cam2world, [0, 0, 0, 1.0]))

for obj in obj1:
    arr = np.asarray(mat['poses'][:, :, 0])
    rot = arr[:, 0:3]
    arr = np.vstack((arr, [0, 0, 0, 1.0]))
    what = np.matmul(arr[:, 3], cam2world.transpose())
    rot =  np.matmul(cam2world[0:3,0:3],rot)
    arr[:, 3] = what.transpose()
    arr[0:3, 0:3] = rot
    obj.set_local2world_mat(arr)

for obj in obj2:
    arr = np.asarray(mat['poses'][:, :, 1])
    rot = arr[:, 0:3]
    arr = np.vstack((arr, [0, 0, 0, 1.0]))
    what = np.matmul(arr[:, 3], cam2world.transpose())
    rot =  np.matmul(cam2world[0:3,0:3],rot)
    arr[:, 3] = what.transpose()
    arr[0:3, 0:3] = rot
    obj.set_local2world_mat(arr)

for obj in obj3:
    arr = np.asarray(mat['poses'][:, :, 2])
    rot = arr[:, 0:3]
    arr = np.vstack((arr, [0, 0, 0, 1.0]))
    what = np.matmul(arr[:, 3], cam2world.transpose())
    rot =  np.matmul(cam2world[0:3,0:3],rot)
    arr[:, 3] = what.transpose()
    arr[0:3, 0:3] = rot
    obj.set_local2world_mat(arr)

bproc.camera.set_intrinsics_from_K_matrix(mat['intrinsic_matrix'], 640, 480)
# Set camera pose via cam-to-world transformation matrix

cam2world = np.asarray(mat['rotation_translation_matrix'])
# Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])
cam2world = np.vstack((cam2world, [0, 0, 0, 1.0]))
bproc.camera.add_camera_pose(cam2world)
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# render the whole pipeline
data = bproc.renderer.render()
# write the data to a .hdf5 container
bproc.writer.write_hdf5("./output", data)
