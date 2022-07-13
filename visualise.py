import os
import numpy as np
import cv2
from PIL import Image
import PIL
import matplotlib.pyplot as plt
import glob
"""

This code is currently not working, doesn't do anything

"""

data_folder = "/data/Wanqing/YCB_Video_Dataset/data_gen"
data_paths = sorted(os.listdir(data_folder))
for path in data_paths:
    folder_name = os.path.join(data_folder, path)
    np_file = sorted(glob.glob(os.path.join(folder_name, "*-np.npy")))
    for file in np_file:
        points = np.load(os.path.join(folder_name, file))


