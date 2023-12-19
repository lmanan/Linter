import os
import random
from glob import glob

import numpy as np
import tifffile
from csbdeep.utils import normalize
from torch.utils.data import Dataset


class CTCDataset(Dataset):
    def __init__(self, data_dir, patch_size=256, size=None):
        print(f"{self.__class__.__name__} created! Accessing data from {data_dir}")
        self.image_list = glob(os.path.join(data_dir, "images", "*.tif"))
        self.mask_list = glob(os.path.join(data_dir, "masks", "*.tif"))
        assert len(self.image_list) == len(self.mask_list)
        self.size = size
        self.real_size = len(self.image_list)
        self.patch_size = patch_size

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        sample = {}
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        image = tifffile.imread(self.image_list[index]).astype(np.float32)
        image = normalize(image, 1, 99.8, axis=(0, 1))
        mask = tifffile.imread(self.mask_list[index])

        H, W = mask.shape
        ids = np.unique(mask)
        ids = ids[ids != 0]
        bbox_outside = True
        while bbox_outside:
            id_chosen = np.random.choice(ids)
            y, x = np.where(mask == id_chosen)
            ym, xm = int(np.mean(y)), int(np.mean(x))
            top_left = (ym - self.patch_size // 2, xm - self.patch_size // 2)
            bottom_right = (ym + self.patch_size // 2, xm + self.patch_size // 2)
            if top_left[0] < 0 or top_left[1] < 0:
                pass
            elif bottom_right[0] > H or bottom_right[1] > W:
                pass
            else:
                sample["image_crop"] = image[
                    np.newaxis,
                    top_left[0] : bottom_right[0],
                    top_left[1] : bottom_right[1],
                ]
                sample["mask_crop"] = mask[
                    np.newaxis,
                    top_left[0] : bottom_right[0],
                    top_left[1] : bottom_right[1],
                ]
                bbox_outside = False
        return sample
