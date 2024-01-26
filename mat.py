from scipy.io import savemat
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil


output_loc = "/HOMES/yigao/Pictures/test _toolboxnyu/depth_image2mat"
image_files = [ f for f in os.listdir("/HOMES/yigao/Pictures/test _toolboxnyu/rgb") if '.jpg' in f.lower() ]

print(image_files)
sorted_image_files = sorted(image_files, key=lambda x: int(x.split('_', 3)[-1].split(".",2)[0]))
print(sorted_image_files)

depth_files = [ f for f in os.listdir("/HOMES/yigao/Pictures/test _toolboxnyu/depth") if '.png' in f.lower() ]

print(depth_files)
sorted_depth_files = sorted(depth_files, key=lambda x: int(x.split('_', 3)[-1].split(".",2)[0]))
print(sorted_depth_files)

for i_file in sorted_image_files:
    for d_file in sorted_depth_files:
        if i_file.split(".", 2)[0].split("_", 3)[-1] == d_file.split(".", 2)[0].split("_", 3)[-1]:
            print(i_file)
            print(d_file)
            img=mpimg.imread("/HOMES/yigao/Pictures/test _toolboxnyu/rgb/" + i_file)
            depth=mpimg.imread("/HOMES/yigao/Pictures/test _toolboxnyu/depth/" + d_file)
            dic = {"images": img, "rawDepths": depth}
            savemat("/HOMES/yigao/Pictures/test _toolboxnyu/depth_image2mat/" + "test" + i_file.split(".", 2)[0].split("_", 3)[-1] + ".mat", dic)

