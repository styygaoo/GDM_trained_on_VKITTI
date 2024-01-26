import numpy as np                        # fundamental package for scientific computing
from PIL import Image
import os
from torchvision import transforms
import torch



# def inverse_depth_norm(depth):
#     depth = maxDepth / depth
#     depth = torch.clamp(depth, maxDepth / 100, maxDepth)
#     return depth
#
#
# maxDepth = 10

test_rgb_path = "/HOMES/yigao/Pictures/September/LMS/combined_up/rgb"
files = [ f for f in os.listdir(test_rgb_path) if '.jpg' in f.lower() ]
sorted_rgb_files = sorted(files, key=lambda x: int(x.split('.')[0]))


test_depthGT_path = "/HOMES/yigao/Pictures/September/LMS/combined_up/depth"
files = [ f for f in os.listdir(test_depthGT_path) if '.png' in f.lower() ]
sorted_depthGT_files = sorted(files, key=lambda x: int(x.split('.')[0]))
# print(sorted_depthGT_files)


trans = transforms.Compose([transforms.Resize(size=(480, 640))])

output_loc = "/HOMES/yigao/Pictures/September/LMS/combined_up/LMS_com_upstairs"      # for GDM

count = 1
for rgb, depthgt in zip(sorted_rgb_files, sorted_depthGT_files):
    # print(rgb)
    # print(depthgt)
    rgb_img = Image.open("/HOMES/yigao/Pictures/September/LMS/combined_up/rgb/" + rgb)
    depth_img = Image.open("/HOMES/yigao/Pictures/September/LMS/combined_up/depth/" + depthgt)

    rgb_img = trans(rgb_img)
    depth_img = trans(depth_img)
    rgb_img = np.asarray(rgb_img)
    depth_img = np.asarray(depth_img)
    # print(depth_img)

    # buffer = np.copy(depth_img)
    # buffer = torch.from_numpy(buffer)
    # depth_img = inverse_depth_norm(buffer)
    # torch.set_printoptions(threshold=10_000)
    # print(depth_img)
    # if not os.path.exists(output_loc + "/%#05d" % (count)):
    #     os.mkdir(output_loc + "/%#05d" % (count))
    # np.save(output_loc + "/%#05d" % (count) + "/image" + "%#05d" % (count), rgb_img)
    # np.save(output_loc + "/%#05d" % (count) + "/depth" + "%#05d" % (count), depth_img)

    np.savez_compressed(output_loc + "/%#05d" % (count), image=rgb_img, depth=depth_img)


    count += 1