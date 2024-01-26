# "/HOMES/yigao/Downloads/GuidedDecoding/GuidedDecoding-main/results/nyu_reduced_full_GuideDepth.engine"
# "/HOMES/yigao/Downloads/GuidedDecoding/GuidedDecoding-main/eval_testset/Picfromknee/image.jpg"


import io, os
import torch
import tensorrt as trt
from torch2trt import torch2trt
import time
from PIL import Image
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit



engine_path="/HOMES/yigao/Downloads/GuidedDecoding/GuidedDecoding-main/results/nyu_reduced_full_GuideDepth.engine"



image_path = "/HOMES/yigao/Downloads/GuidedDecoding/GuidedDecoding-main/eval_testset/Picfromknee/image.jpg"
img = Image.open(io.BytesIO(open(image_path, 'rb').read()))

print('Image', img.transpose([2, 0, 1]))
print(img.shape)

device = torch.device("cuda:0")
image_tensor = torch.from_numpy(np.array([img.transpose([2, 0, 1])])).to(torch.float).cuda()

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

# Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
h_input = cuda.pagelocked_empty(engine.get_binding_shape(0).volume(), dtype=np.float32)
h_output = cuda.pagelocked_empty(engine.get_binding_shape(1).volume(), dtype=np.float32)
# Allocate device memory for inputs and outputs.
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
# Create a stream in which to copy inputs/outputs and run inference.
stream = cuda.Stream()

with engine.create_execution_context() as context:

    result = engine.infer(img)

    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()
    # Return the host output.
    print('output',h_output)

