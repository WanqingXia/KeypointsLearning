import os
import numpy as np

"""
This script is used to calculate the diagonal length of the bounding boxes
of each object, the diagonal length will be used to control the camera distance
in generate.py, the calculated length and object path will be saved to "diameters.txt"
"""

class Point(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class File(object):
    count = 0
    filepaths = []
    def get_file_paths(self, base_path):
        folders = sorted(os.listdir(base_path))
        for folder in folders:
            folder_path = os.path.join(base_path, folder)
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and file[-3:] == "obj" and file[-10:-4]!="simple":
                   #print("file path:", file_path)
                   self.filepaths.append(file_path)
                   self.count += 1

class Normalize(object):
    minP = Point(1000, 10000, 10000)
    maxP = Point(0, 0, 0)

    def reset_points(self):
        self.minP = Point(1000, 10000, 10000)
        self.maxP = Point(0, 0, 0)

    def get_bounding_box(self, p):
    # Get min and max for x, y, z of an object
        self.minP.x = p.x if p.x < self.minP.x else self.minP.x
        self.minP.y = p.y if p.y < self.minP.y else self.minP.y
        self.minP.z = p.z if p.z < self.minP.z else self.minP.z
        self.maxP.x = p.x if p.x > self.maxP.x else self.maxP.x
        self.maxP.y = p.y if p.y > self.maxP.y else self.maxP.y
        self.maxP.z = p.z if p.z > self.maxP.z else self.maxP.z

    def get_bounding_box_length(self):
    # Get the length for bounding box
        length = self.maxP.x - self.minP.x
        width = self.maxP.y - self.minP.y
        height = self.maxP.z - self.minP.z
        return length, width, height


    def get_bounding_box_diag_length(self, length, width, height):
    # Get the diagonal length
        diag_rect = np.sqrt(length**2 + width**2)
        diag_box = np.sqrt(diag_rect**2 + height**2)
        return diag_box


    def read_points(self, filename):
    # read all points in an obj file
        with open(filename) as file:
            points = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    points.append(Point(float(strs[1]), float(strs[2]), float(strs[3])))
                if strs[0] == "vt":
                    break
        return points


if __name__ == "__main__":
    myFile = File()
    basePath = "/data/Wanqing/YCB_Video_Dataset/models"
    myFile.get_file_paths(basePath)
    myNormalize = Normalize()

    with open("./diameters.txt", "w") as f:
        for file in myFile.filepaths:
            # read points from obj file
            points = myNormalize.read_points(file)
            for point in points:
                myNormalize.get_bounding_box(point)
                # get the length and diagnoal length of bounding box
            length, width, height = myNormalize.get_bounding_box_length()
            diag = myNormalize.get_bounding_box_diag_length(length, width, height)
            myNormalize.reset_points()
            f.write(f"{file} {round(diag,3)}\n")
            #print("the diag length for", file, "is", diag)

    print("Finished calculating diagonal length")
