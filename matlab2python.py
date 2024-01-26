from PIL import Image
import numpy as np
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

files = [ f for f in os.listdir("/HOMES/yigao/Pictures/test _toolboxnyu/toolboxed")]
print(files)
sorted_files = sorted(files, key=lambda x: int(x))
print(sorted_files)
count = 0

for f in sorted_files:
    with open("/HOMES/yigao/Pictures/test _toolboxnyu/toolboxed/"+f, 'rb') as ff:
        img = mpimg.imread(ff)
        plt.imshow(img)
        plt.show()
        img = cv2.normalize(img, img, 0.1, 10, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        plt.imsave("/HOMES/yigao/Pictures/test _toolboxnyu/toolboxed_pythoned/" + "depth_" + "%#05d" % (count + 1) + ".png", img)
        count += 1