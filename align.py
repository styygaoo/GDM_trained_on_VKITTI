import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API


import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import os


print("Environment Ready")

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("/HOMES/yigao/Downloads/GuidedDecoding/GuidedDecoding-main/test.bag")



# output_loc = "/HOMES/yigao/Downloads/eval_testset/forZOE/"      # for ZOE
output_loc = "/HOMES/yigao/Downloads/eval_testset/test/"      # for GDM
count = 0
transform = T.Resize(size=(480, 640))

profile = pipe.start(cfg)



while True:
    if count == 154:
        break
# Skip 5 first frames to give the Auto-Exposure time to adjust

    # pipe.wait_for_frames()

    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Cleanup:
    # pipe.stop()
    # print("Frames Captured")

    color = np.asanyarray(color_frame.get_data())
    plt.rcParams["axes.grid"] = False
    plt.rcParams['figure.figsize'] = [12, 6]

    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 0) # white to black
    # colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
    # plt.imshow(colorized_depth)
    # plt.show()

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    # colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())



    print("colorized_depth.shape:, ", colorized_depth.shape)
    img = Image.fromarray(colorized_depth)
    img = img.resize((480, 640))
    plt.imshow(colorized_depth)
    plt.show()


    ''' for ZOE '''
    # if not os.path.exists(output_loc + "/%#05d" % (count+1)):
    #     os.mkdir(output_loc + "/image_" + "%#05d" % (count+1))

    # plt.imsave(output_loc + "/image_" + "%#05d" % (count+1) + ".jpg", color)
    # plt.imsave(output_loc + "/depth_" + "%#05d" % (count+1) + ".png", colorized_depth, cmap='viridis')


    ''' for GDM '''
    if not os.path.exists(output_loc + "/%#05d" % (count+1)):
        os.mkdir(output_loc + "/%#05d" % (count+1))
    print(colorized_depth.shape)
    np.save(output_loc + "/%#05d" % (count+1) + "/depth" + "%#05d" % (count+1), colorized_depth)
    np.save(output_loc + "/%#05d" % (count+1) + "/image" + "%#05d" % (count+1), color)

    # np.savez_compressed(output_loc + "/%#05d" % (count+1), depth=colorized_depth, image=color)

    count += 1

    # Show the two frames together:
    # images = np.hstack((color, colorized_depth))
    # plt.imshow(images)
    # plt.show()
