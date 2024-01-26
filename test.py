import numpy as np

b = np.load('/HOMES/yigao/Downloads/eval_testset/NYU_Testset/nyu_test_001.npz')

# print(b['depth'])
depth = b['depth']
print(depth.shape)

# print(b['image'])
rgb = b['image']
print(rgb.shape)

# pd = np.load('/HOMES/yigao/Documents/Below_knee/go_upstairs/4/forGDM/00001.npz')
#
# # print(pd['depth'])
# depth = pd['depth']
# print(depth.shape)
#
# # print(pd['image'])
# rgb = pd['image']
# print(rgb.shape)