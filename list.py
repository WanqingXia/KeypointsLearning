import os
import glob
"""
This script creates a txt file for ycb video data set to show the objects in every video
"""
def main():
    d = open("list.txt", "w")
    model_name = sorted(os.listdir("/data/Wanqing/YCB_Video_Dataset/models"))
    # take the first txt file in every folder
    for folder in sorted(os.listdir("/data/Wanqing/YCB_Video_Dataset/data")):
        txt_path = os.path.join("/data/Wanqing/YCB_Video_Dataset/data",folder, "000001-box.txt")
        f = open(txt_path, "r")

        lines = f.readlines()
        d.write('\n' + folder.split("/")[-1])
        for index, line in enumerate(lines):
            d.write(" " + line.split(" ")[0])

if __name__ == "__main__":
    main()
    print("list created, open with txt or excel")
