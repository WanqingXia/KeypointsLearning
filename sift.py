import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import open3d as o3d

def calc_angels(object_folder, position_file, neighbour_file):
    num_lines = sum(1 for line in open(position_file))
    lines = np.zeros((num_lines, 3))
    out = np.zeros((num_lines, 4))

    with open(position_file, "r") as f:
        for num, line in enumerate(f.readlines()):
            line = [float(x) for x in line.split()]
            lines[num, :] = line

    for num, line in enumerate(lines):
        angles = np.ones(4) * 100
        pos = np.ones(4) * 100
        for count, others in enumerate(lines):
            angle = np.arccos(np.clip(np.dot(line / np.linalg.norm(line), others / np.linalg.norm(others)), -1.0, 1.0))
            if angle < 0.02:
                pass
            elif angle < np.max(angles):
                pos[np.where(angles == np.max(angles))[0][0]] = count
                angles[np.where(angles == np.max(angles))[0][0]] = angle

        out[num] = pos

    np.savetxt(neighbour_file, out, fmt='%i')
    print("Calculated neighbours for " + object_folder)

def read_detect(image_folder, mask_folder, sift):
    # read images and detect keypoints
    image_files = os.listdir(image_folder)
    mask_files = os.listdir(mask_folder)
    for num, name in enumerate(image_files):
        image_files[num] = "colour_img_" + str(num) + ".jpg"
    for num, name in enumerate(mask_files):
        mask_files[num] = "mask_" + str(num)
    keypoints_list = []
    descriptors_list = []

    for image, mask in zip(image_files, mask_files):
        img = cv2.imread(os.path.join(image_folder, image))
        msk = np.load(os.path.join(mask_folder, mask))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)

        arr = []
        for num, keypoint in enumerate(keypoints):
            y = int(keypoint.pt[0])
            x = int(keypoint.pt[1])
            if not msk[x, y]:
                arr.append(num)
        keypoints = np.delete(keypoints, arr).tolist()
        descriptors = np.delete(descriptors, arr, axis=0)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    print("Finished calculate keypoints and descriptors for " + image_folder)
    return keypoints_list, descriptors_list

def filter(keypoints_list, descriptors_list, neighbour_file, bf):
    keypoints_list_f = []
    count =0
    with open(neighbour_file, "r") as n:
        neighbour = n.readlines()
    for keypoint, descriptor, line in tqdm(zip(keypoints_list, descriptors_list, neighbour), total=len(keypoints_list)):
        counter = np.zeros(len(keypoint))
        keypoint_f = []

        neighbour_num = line.strip("\n").split(" ")
        for num in neighbour_num:
            kp, dp = keypoints_list[int(num)], descriptors_list[int(num)]
            matches = bf.match(descriptor, dp)
            matches_f = [match for match in matches if match.distance < 500]
            for match in matches_f:
                counter[match.queryIdx] += 1
        indices = np.asarray(np.where(counter == 4)).flatten()
        for idx in indices:
            keypoint_f.append(keypoint[idx])
        # This part is used to visualise keypoints
        # img = cv2.imread("output/021_bleach_cleanser/colour/colour_img_0.jpg")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # imgWithKeypoints = cv2.drawKeypoints(img, keypoint_f, 0)
        # plt.imshow(imgWithKeypoints)
        # plt.show()
        arr = np.zeros((len(keypoint_f), 2))
        for num, point in enumerate(keypoint_f):
            arr[num, 0] = int(point.pt[1])
            arr[num, 1] = int(point.pt[0])
        if count == 0:
            np.save("./arr1", arr)
            count+=1
        elif count == 1:
            np.save("./arr2", arr)

        ############

        keypoints_list_f.append(keypoint_f)

    return keypoints_list_f

def map_to_model(model_folder, keypoints_list_f, depth_folder, matrix_folder, camera_in):
    for num, keypoints in enumerate(keypoints_list_f):
        depth_img = np.load(os.path.join(depth_folder, "depth_" + str(num)))
        cam2world_matrix = np.load(os.path.join(matrix_folder, "matrix_" + str(num)))
        for keypoint in keypoints:
            # transform form depth to 3D coordinate
            # get from https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/box_dimensioner_multicam/helper_functions.py
            point3D = np.ones(4)
            y = int(keypoint.pt[0])
            x = int(keypoint.pt[1])
            point3D[0] = (x - camera_in[0, 2]) * depth_img[x, y] / camera_in[0, 0]    # x location of point
            point3D[1] = (y - camera_in[1, 2]) * depth_img[x, y] / camera_in[1, 1]    # y location of point
            point3D[2] = depth_img[x, y]                                              # z location of point
            point_world = np.dot(point3D, cam2world_matrix[0:3, :].transpose())




if __name__ == "__main__":
    output_folder = "./output"
    object_folders = os.listdir(output_folder)
    camera_intrinsics = np.load("./camera_intrinsics")

    # sift
    sift = cv2.xfeatures2d.SIFT_create()
    # matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    for folder in object_folders:
        image_folder = os.path.join(output_folder, folder, "colour")
        depth_folder = os.path.join(output_folder, folder, "depth")
        mask_folder = os.path.join(output_folder, folder, "mask")
        matrix_folder = os.path.join(output_folder, folder, "matrix")
        position_file = os.path.join(output_folder, folder, "positions.txt")
        neighbour_file = os.path.join(output_folder, folder, "neighbours.txt")
        model_folder = os.path.join("./models", folder)


        if os.path.exists(neighbour_file) == False:
            calc_angels(os.path.join(output_folder, folder), position_file, neighbour_file)

        keypoints_list, descriptors_list = read_detect(image_folder, mask_folder, sift)
        keypoints_list_f = filter(keypoints_list, descriptors_list, neighbour_file, bf)
        # map_to_model(model_folder, keypoints_list_f, depth_folder, matrix_folder, camera_intrinsics)
