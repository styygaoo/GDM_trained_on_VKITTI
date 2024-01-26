import numpy as np
import torch



# img = torch.rand(1, 3, 5, 8)
# gt = torch.rand(3, 5, 8)
# print("img: ", img)
#
# img = img[0].permute(1, 2, 0)
# print("img: ", img)


# gt = gt[0, 0].permute(0, 1)



i = 100
print(type(i))
# <class 'int'>

print(np.iinfo(i))


fi32 = np.finfo(np.float32)
print(type(fi32))
print(fi32)