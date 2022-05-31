import blenderproc as bproc
import numpy as np
import os
import cv2
import h5py
import shutil
import png

"""
This is the main script to generate all the images with BlenderProc
The camera position and light position will be reset for each image
Run the script by calling "blenderproc run generate.py" in terminal
"""

# functions for debug, only uncomment this when debugging is needed
#import debugpy
#debugpy.listen(5678)
#debugpy.wait_for_client()

# exception handler
def handler(func, path, exc_info):
    print("Inside handler")
    print(exc_info)


# Get obj file path and corresponding length from "diameters.txt"
def get_path_and_length(file, filepaths, diag_length):
    with open(file, "r") as f:
        for line in f.readlines():
            strs = line.split(" ")
            filepaths.append(strs[0])
            diag_length.append(float(strs[1]))


# Create the output path for each object
def create_output_path(filepath, output_folder):
    obj_name = filepath.split("/")[-2]
    out_parent = os.path.join(output_folder, obj_name)
    out_colour = os.path.join(out_parent, "colour")
    out_depth = os.path.join(out_parent, "depth")
    out_mask = os.path.join(out_parent, "mask")
    out_matrix = os.path.join(out_parent, "matrix")
    if os.path.exists(output_folder)==False:
        os.mkdir(output_folder)

    if os.path.exists(out_parent):
        shutil.rmtree(out_parent, onerror=handler)

    os.mkdir(out_parent)
    os.mkdir(out_colour)
    os.mkdir(out_depth)
    os.mkdir(out_mask)
    os.mkdir(out_matrix)
    return out_parent, out_colour, out_depth, out_mask, out_matrix

# Sample camera points and save to txt
def sample_points(path, radius, sample):
    points = [[0, 0, 0] for _ in range(sample)]
    for n in range(sample):formula
        phi = np.arccos(-1.0 + (2.0 * (n + 1) - 1.0) / sample)
        theta = np.sqrt(sample * np.pi) * phi
        points[n][0] = radius * np.cos(theta) * np.sin(phi)
        points[n][1] = radius * np.sin(theta) * np.sin(phi)
        points[n][2] = radius * np.cos(phi)

    points = np.array(points)
    savepath = os.path.join(path,"positions.txt")
    np.savetxt(savepath, points, fmt='%1.4f')
    return savepath

# The core function for rendering images, render a colour and depth image
# for each camera location
def render(obj_path, pos_file, parent_path, colour_path, depth_path, mask_path, matrix_path, count):
    bproc.init()

    # load the objects into the scene
    objs = bproc.loader.load_obj(obj_path)

    # define a light and set its location and energy level
    light = bproc.types.Light()

    # define the camera resolution
    bproc.camera.set_resolution(640, 480)

    # locate object center
    # poi = bproc.object.compute_poi(objs)
    poi = np.array([0, 0, 0])
    # activate depth rendering
    if count == 0:
        bproc.renderer.enable_depth_output(activate_antialiasing=False)
        K = bproc.camera.get_intrinsics_as_K_matrix()
        with open(os.path.join("./camera_intrinsics"), 'wb') as cam:
                np.save(cam, K)

    else:
        pass

    with open(pos_file, "r") as f:
        for num, line in enumerate(f.readlines()):

            location = [float(x) for x in line.split()]
            # set light
            light.set_location(location)
            light.set_type("POINT")
            light.set_energy(20)

            # set camera
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location)
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
            bproc.camera.add_camera_pose(cam2world_matrix)

            # render the whole pipeline
            data = bproc.renderer.render()

            # write the data to a .hdf5 container
            bproc.writer.write_hdf5(os.path.join(parent_path, str(num)), data)

            # reset the scene (clear the camera and light)
            bproc.utility.reset_keyframes()

            with h5py.File(os.path.join(parent_path, str(num),'0.hdf5'),'r') as h5f:
                    colours = np.array(h5f["colors"])[...,::-1].copy()
                    cv2.imwrite(os.path.join(colour_path, f'colour_img_{num}.jpg'),colours)
                    with open(os.path.join(depth_path, f'depth_{num}.png'), 'wb') as im:
                            float_arr = np.array(h5f["depth"])
                            mask = np.array(h5f["depth"]) < 100
                            int_arr = float_arr*mask*10000
                            writer = png.Writer(width=640, height=480, bitdepth=16, greyscale=True)
                            writer.write(im, int_arr.astype(np.int16))
                    # with open(os.path.join(mask_path, f'mask_{num}'), 'wb') as dat:
                            # savearr = np.array(h5f["depth"]) < 100
                            # np.save(dat, savearr)
                    with open(os.path.join(matrix_path, f'matrix_{num}'), 'wb') as dat:
                            np.save(dat, cam2world_matrix)
            shutil.rmtree(os.path.join(parent_path, str(num)))

    return count+1

if __name__ == "__main__":
    filepaths = []
    diag_length = []
    get_path_and_length("diameters.txt", filepaths, diag_length)
    count = 0

    for obj_path, length in zip(filepaths, diag_length):
        parent_path, colour_path, depth_path, mask_path, matrix_path = create_output_path(obj_path, output_folder="/data/Wanqing/YCB_Video_Dataset/YCB_objects")
        pos_file = sample_points(parent_path, radius=length*3, sample=400)
        count = render(obj_path, pos_file, parent_path, colour_path, depth_path, mask_path,matrix_path, count)


