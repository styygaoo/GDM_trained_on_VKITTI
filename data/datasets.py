import os

from torch.utils.data import DataLoader

from data.kitti import KITTIDataset
from data.vkitti import vKITTIDataset

from data.nyu_reduced import get_NYU_dataset

"""
Preparation of dataloaders for Datasets
"""

resolution_dict = {
    'full' : (384, 1280),
    'tu_small' : (128, 416),
    'tu_big' : (228, 912),
    'half' : (192, 640)}

def get_dataloader(dataset_name, 
                   path,
                   split='train', 
                   resolution='full',
                   augmentation='alhashim',
                   interpolation='linear', 
                   batch_size=1,
                   workers=4, 
                   uncompressed=False):
    if dataset_name == 'kitti':
        dataset = KITTIDataset(path, 
                split, 
                resolution=resolution)

    elif dataset_name == 'nyu_reduced':
        dataset = get_NYU_dataset(path, 
                split, 
                resolution=resolution, 
                uncompressed=True)

    elif dataset_name == 'vkitti':

        split_dir = os.path.join(os.getcwd(), 'Splits', "vkitti")
        if split == "train":
            train_dir = "/HOMES/yigao/KITTI/vkitti_data/vkitti"
            dataset = vKITTIDataset(train_dir, src_file=os.path.join(split_dir, 'train_split.pickle'),
                                          transform='train', output_size=resolution_dict['half'])

        elif split == "val":
            valid_dir = "/HOMES/yigao/KITTI/vkitti_data/vkitti"
            dataset = vKITTIDataset(valid_dir, src_file=os.path.join(split_dir, 'val_split.pickle'),
                                          transform='valid', output_size=resolution_dict[resolution])


    else:
        print('Dataset not existant')
        exit(0)

    dataloader = DataLoader(dataset, 
            batch_size=batch_size, 
            shuffle=(split=='train'),
            num_workers=workers, 
            pin_memory=True)
    return dataloader
