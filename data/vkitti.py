# -*- coding: utf-8 -*-
import numpy as np
import data.transforms_vkitti as transforms
from data.dataloader_vkitti import MyDataloader
from data.transforms import Resize, RandomHorizontalFlip, RandomChannelSwap, ToTensor, CenterCrop, RandomRotation, RandomVerticalFlip
import torch


class vKITTIDataset(MyDataloader):
    def __init__(self, root, src_file, transform, output_size=(256, 512), modality='rgb'):
        self.src_file = src_file
        self.output_size = output_size

        super(vKITTIDataset, self).__init__(root, src_file, transform, modality)

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)  # random scaling
        # depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        transform = transforms.Compose([
            # # transforms.Resize(300.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            # transforms.Rotate(angle),
            # # transforms.Resize(s),
            # transforms.CenterCrop((300, 728)),
            # transforms.HorizontalFlip(do_flip),
            transforms.Resize(self.output_size),
        ])
        # print(rgb)

        rgb_np = transform(rgb)

        # rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255

        rgb_np = np.clip(rgb_np, 0.0, 1.0)
        # print(rgb_np)

        depth_np = depth.astype(np.float32)
        zero_mask = depth_np == 0.0

        depth_np = transform(depth_np)
        # depth is given in cm
        # set maximum depth of 80m / 8000 cm
        depth_np[depth_np > 8000] = 8000

        # scale depth
        # depth_np = depth_np / 8000
        depth_np = depth_np / 100


        """For train use DepthNorm"""

        depth_np = np.clip(depth_np, 80 / 100.0, 80)

        depth_np = 80 / depth_np

        depth[zero_mask] = 0.0
        # print(rgb_np)
        return rgb_np, depth_np

    def valid_transform(self, rgb, depth):
        transform = transforms.Compose([
            # transforms.Resize(300.0 / iheight),
            # transforms.CenterCrop((300, 728)),
            transforms.Resize(self.output_size)
        ])

        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        rgb_np = np.clip(rgb_np, 0.0, 1.0)

        depth_np = transform(depth)
        # depth is given in cm
        # set maximum depth of 80m / 8000 cm
        depth_np[depth_np > 8000] = 8000
        # scale to m
        depth_np = depth_np * 1e-2

        return rgb_np, depth_np
