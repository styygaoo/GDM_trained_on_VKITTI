import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# read the input image
img = Image.open("/HOMES/yigao/Downloads/GuidedDecoding/GuidedDecoding-main/eval_testset/Picfromknee/lms.png")

# compute the size(width, height) of image
size = img.size
print("Size of the Original image:", size)

# define transformt o resize the image with given size
transform = T.Resize(size=(480, 640))
# apply the transform on the input image
img = transform(img)
print("Size after resize:", img.size)
plt.imshow(img)
plt.show()
img.save('lms_m.png')