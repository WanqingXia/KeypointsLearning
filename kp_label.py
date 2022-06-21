import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import glob


def label(folder_name, gen_folder_name, sift, fast, bf):

    ori_color_files = sorted(glob.glob(os.path.join(folder_name, "*-color.png")))
    gen_color_files = sorted(glob.glob(os.path.join(gen_folder_name, "*-color.png")))

    for num, color_name in tqdm(enumerate(ori_color_files), desc="Calculating keypoints for" + folder_name, total=len(ori_color_files)):
        color_r = os.path.join(folder_name, color_name)
        color_f = os.path.join(gen_folder_name, gen_color_files[num])
        depth_r = color_r.split("-")[0] + "-depth.png"
        depth_f = color_f.split("-")[0] + "-depth.png"
        if color_r.split("/")[5:] != color_f.split("/")[5:] or depth_r.split("/")[5:] != depth_f.split("/")[5:]:
            raise Exception("File name not match! Program aborted.")

        img_r = cv2.imread(color_r)
        img_f = cv2.imread(color_f)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2GRAY)
        depth_img_r = Image.fromarray(np.array(Image.open(depth_r)).astype("uint16"))
        depth_img_r = (np.asarray(depth_img_r)/10000).astype(np.float32)
        depth_img_f = Image.fromarray(np.array(Image.open(depth_f)).astype("uint16"))
        depth_img_f = (np.asarray(depth_img_f)/10000).astype(np.float32)


        '''
        Calculating colour image keypoints with sift
        '''

        keypoints_r, descriptors_r = sift.detectAndCompute(img_r, None)
        keypoints_f, descriptors_f = sift.detectAndCompute(img_f, None)

        # arr_r = []
        # for num_r, keypoint_r in enumerate(keypoints_r):
        #     y = int(keypoint_r.pt[0])
        #     x = int(keypoint_r.pt[1])
        #     if depth_img_f[x, y] == 0:
        #         arr_r.append(num_r)
        # keypoints_r = np.delete(keypoints_r, arr_r, axis=0)
        # descriptors_r = np.delete(descriptors_r, arr_r, axis=0)

        arr_f = []
        for num_f, keypoint_f in enumerate(keypoints_f):
            y = int(keypoint_f.pt[0])
            x = int(keypoint_f.pt[1])
            if depth_img_f[x, y] == 0:
                arr_f.append(num_f)
        keypoints_f = np.delete(keypoints_f, arr_f, axis=0)
        descriptors_f = np.delete(descriptors_f, arr_f, axis=0)

        matches_f = []
        matches = bf.match(descriptors_r, descriptors_f)
        # matches_f = [match for match in matches if match.distance < 500]
        for match in matches:
            point_r = keypoints_r[match.queryIdx]
            point_f = keypoints_f[match.trainIdx]
            if np.linalg.norm(np.array([int(point_r.pt[0]), int(point_r.pt[1])]) - np.array([int(point_f.pt[0]), int(point_f.pt[1])]))<=3:
                    matches_f.append(match)

        '''
        Calculating depth image keypoints with fast
        '''
        depmin = min(i for i in depth_img_r.flatten() if i > 0)
        depmax = np.max(depth_img_r)

        for x in range(480):
            for y in range(640):
                if depth_img_r[x, y] > 0:
                    depth_img_r[x, y] = ((depth_img_r[x, y] - depmin) * (255 / (depmax - depmin)))

        depmin = min(i for i in depth_img_f.flatten() if i > 0)
        depmax = np.max(depth_img_f)
        for x in range(480):
            for y in range(640):
                if depth_img_f[x, y] > 0:
                    depth_img_f[x, y] = ((depth_img_f[x, y] - depmin) * (255 / (depmax - depmin)))

        depth_img_r = np.round(depth_img_r).astype(dtype=np.uint8)
        depth_img_f = np.round(depth_img_f).astype(dtype=np.uint8)
        depth_img_r = np.repeat(depth_img_r[..., np.newaxis], 3, axis=2)
        depth_img_f = np.repeat(depth_img_f[..., np.newaxis], 3, axis=2)

        kp_r = fast.detect(depth_img_r, None)
        kp_f = fast.detect(depth_img_f, None)

        kp_rn = []
        kp_fn = []
        for num, f in enumerate(kp_f):
            kp_dist = 100
            for r in kp_r:
                cur_dist = np.linalg.norm(np.array([f.pt[0], f.pt[1]]) - np.array([r.pt[0], r.pt[1]]))
                if cur_dist <= 3 and cur_dist < kp_dist:
                    if kp_dist == 100:
                        kp_rn.append(r)
                        kp_fn.append(f)
                    else:
                        kp_rn.pop()
                        kp_fn.pop()
                        kp_rn.append(r)
                        kp_fn.append(f)
                    kp_dist = cur_dist

        save_points = np.zeros((len(matches_f)+len(kp_fn), 4))
        for num, match in enumerate(matches_f):
            # trainidx: generated queryidx: real
            save_points[num, 0] = int(keypoints_f[match.queryIdx].pt[1]) #real_row
            save_points[num, 1] = int(keypoints_f[match.queryIdx].pt[0]) #real_col
            save_points[num, 2] = int(keypoints_r[match.trainIdx].pt[1]) #gene_row
            save_points[num, 3] = int(keypoints_r[match.trainIdx].pt[0]) #gene_col
        pre_num = len(matches_f)
        for count in range(len(kp_fn)):
            cur_pos = pre_num + count
            save_points[cur_pos, 0] = kp_fn[count].pt[1]
            save_points[cur_pos, 1] = kp_fn[count].pt[0]
            save_points[cur_pos, 2] = kp_rn[count].pt[1]
            save_points[cur_pos, 3] = kp_rn[count].pt[0]
        save_points = save_points.astype(dtype=np.uint16)

        unique_points = np.unique(save_points[:, 2:], axis=0)
        save_points_n = np.zeros((len(unique_points), 4)).astype(np.uint16)
        for count, line_u in enumerate(unique_points):
            for num, line_sec in enumerate(save_points):
                if np.linalg.norm(line_u - line_sec[2:]) == 0:
                    save_points_n[count, :] = line_sec
                    break

        np.save(color_f.split("-")[0] + "-np.npy", save_points_n)

        # code for visualising the keypoints
        # img1 = plt.imread(color_f)
        # img2 = plt.imread(color_r)
        # out_img = np.concatenate((img1, img2), axis=1)
        # for line in save_points_n:
        #     img1[line[0]-2:line[0]+3, line[1]-2: line[1]+3, :] = [0, 1, 0]
        #     img2[line[2]-2:line[2]+3, line[3]-2: line[3]+3, :] = [0, 1, 0]
        #     out_img = cv2.line(out_img, (line[0], line[1]), (line[2] + img1.shape[1], line[3]),
        #                        (0, 255, 0), 1)
        # plt.figure()
        # plt.imshow(img1)
        # plt.figure()
        # plt.imshow(img2)
        # plt.figure()
        # plt.imshow(out_img)
        # plt.show()
        # stop = 1

if __name__ == "__main__":
    data_folder = "/data/Wanqing/YCB_Video_Dataset/data"
    data_gen_folder = "/data/Wanqing/YCB_Video_Dataset/data_gen"
    data_paths = sorted(os.listdir(data_folder))
    data_gen_paths = sorted(os.listdir(data_gen_folder))

    # sift
    sift = cv2.SIFT_create()
    # fast
    fast = cv2.FastFeatureDetector_create()
    # matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    for num, name in enumerate(data_paths):
        data_paths[num] = os.path.join(data_folder, name)
    for num, name in enumerate(data_gen_paths):
        data_gen_paths[num] = os.path.join(data_gen_folder, name)
    for folder_name, gen_folder_name in zip(data_paths, data_gen_paths):
        label(folder_name, gen_folder_name, sift, fast, bf)

