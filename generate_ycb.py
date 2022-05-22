import blenderproc as bproc
import numpy as np
import os
import cv2
import h5py
import shutil
import scipy.io
import glob
import png

# functions for debug, only uncomment this when debugging is needed
# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()


def render(folder_name, model_paths, out_folder, count):
    bproc.init()

    # create output folder
    out_subfolder = os.path.join(out_folder,folder_name.split("/")[-1] )
    if os.path.exists(out_subfolder):
        shutil.rmtree(out_subfolder)
    os.mkdir(out_subfolder)

    if count == 0:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)

    mat_files = sorted(glob.glob(os.path.join(folder_name, "*.mat")))

    for num, name in enumerate(mat_files):
        # read mat file
        mat = scipy.io.loadmat(name)

        # define a light and set its location and energy level
        light = bproc.types.Light()
        light.set_location(mat['rotation_translation_matrix'][:, 3].flatten())
        light.set_type("POINT")
        light.set_energy(80)

        # load the objects into the scene
        if num == 0:
            objs = []
            for obj_num in mat["cls_indexes"]:
                objs.append(bproc.loader.load_obj(model_paths[int(obj_num)-1]))

        
        cam2world = np.asarray(mat['rotation_translation_matrix'])
        cam2world = np.vstack((cam2world, [0, 0, 0, 1.0]))

        for count, obj in enumerate(objs):
            for object in obj:
                pose = np.asarray(mat['poses'][:, :, count])
                pose_n = np.vstack((pose, [0, 0, 0, 1.0]))
                # Calculate translation in world
                pose_n[:, 3] = np.matmul(cam2world, pose_n[:, 3])
                # Calculate rotation in world
                pose_n[0:3, 0:3] = np.matmul(cam2world[0:3,0:3], pose[:, 0:3])
                object.set_local2world_mat(pose_n)

        # Set camera intrinsics with K matrix
        bproc.camera.set_intrinsics_from_K_matrix(mat['intrinsic_matrix'], 640, 480)
        # Read cam-to-world transformation matrix (extrincis)
        cam2world = np.asarray(mat['rotation_translation_matrix'])
        # Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
        cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])
        cam2world = np.vstack((cam2world, [0, 0, 0, 1.0]))
        bproc.camera.add_camera_pose(cam2world)

        # render the whole pipeline
        data = bproc.renderer.render()
        # write the data to a .hdf5 container
        bproc.writer.write_hdf5(os.path.join(out_subfolder, str(num)), data)

        # reset the scene (clear the camera and light)
        bproc.utility.reset_keyframes()

        with h5py.File(os.path.join(out_subfolder, (str(num) + '/0.hdf5')),'r') as h5f:
            print(os.path.join(out_subfolder, (str(num) + '/0.hdf5')))
            colour_file = os.path.join(out_subfolder, (name.split("/")[-1].split("-")[0]+ "-color.png"))
            print(colour_file)

            depth_file = os.path.join(out_subfolder, (name.split("/")[-1].split("-")[0]+ "-depth.png"))
            print(depth_file)
            colours = np.array(h5f["colors"])[...,::-1].copy()
            cv2.imwrite(colour_file,colours)
            with open((depth_file), 'wb') as im:
                float_arr = np.array(h5f["depth"])
                mask = np.array(h5f["depth"]) < 100
                int_arr = float_arr*mask*10000
                writer = png.Writer(width=640, height=480, bitdepth=16, greyscale=True)
                writer.write(im, int_arr.astype(np.int16))
        shutil.rmtree(os.path.join(out_subfolder, str(num)))

    return count+1

if __name__ == "__main__":
    count = 0

    data_folder = "/data/Wanqing/YCB_Video_Dataset/data"
    model_folder = "/data/Wanqing/YCB_Video_Dataset/models"
    out_folder = "/data/Wanqing/YCB_Video_Dataset/data_gen"
    data_paths = sorted(os.listdir(data_folder))
    model_paths = sorted(os.listdir(model_folder))
    for num, name in enumerate(data_paths):
        data_paths[num] = os.path.join(data_folder, name)
    for num, name in enumerate(model_paths):
        model_paths[num] = os.path.join(model_folder, name, "textured.obj")

    if os.path.exists(out_folder):
        pass
        # shutil.rmtree(out_folder)
    else:
        os.mkdir(out_folder)

    for folder_name in data_paths:
        count = render(folder_name, model_paths, out_folder, count)
