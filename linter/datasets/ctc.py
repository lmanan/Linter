import glob
import os
import random

import numpy as np
import tifffile
from csbdeep.utils import normalize
from torch.utils.data import Dataset


class CTCDataset(Dataset):
    def __init__(
        self,
        data_dir="./",
        type="train",
        bg_id=0,
        size=None,
        transform=None,
        crop_size=256,
    ):
        print(
            "CTC `{}` dataloader created! Accessing data from {}/{}/".format(
                type, data_dir, type
            )
        )

        # get image and instance list
        image_list = glob.glob(os.path.join(data_dir, f"{type}/", "images/*.tif"))
        image_list.sort()
        print(f"Number of images in `{type}` directory is {len(image_list)}")
        self.image_list = image_list

        instance_list = glob.glob(os.path.join(data_dir, f"{type}/", "masks/*"))
        instance_list.sort()
        print(
            "Number of instances in `{}` directory is {}".format(
                type, len(instance_list)
            )
        )
        self.instance_list = instance_list

        print("*************************")

        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform
        self.type = type
        self.crop_size = crop_size

    def __len__(self):
        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        example = {}

        # load image
        image = tifffile.imread(self.image_list[index]).astype(np.float32)  # YX
        image = normalize(image, 1, 99.8, (0, 1))
        mask = tifffile.imread(self.instance_list[index])  # YX

        ids = np.unique(mask)
        ids = ids[ids != self.bg_id]

        # Randomly pick one object centered crop
        outside = True
        while outside:
            random_id = np.random.choice(ids)
            y, x = np.where(mask == random_id)
            ym, xm = int(np.mean(y)), int(np.mean(x))
            tl_y = ym - self.crop_size // 2
            tl_x = xm - self.crop_size // 2
            br_y = tl_y + self.crop_size
            br_x = tl_x + self.crop_size

            if tl_y < 0 or tl_x < 0 or br_y > mask.shape[0] or br_x > mask.shape[1]:
                pass
            else:
                image_crop = image[tl_y:br_y, tl_x:br_x]
                mask[tl_y:br_y, tl_x:br_x]
                outside = False  # leave the while loop

        example["image"] = image_crop[np.newaxis, ..., ]  # CYX
        # example["im_name"] = (
        #    self.image_list[index][:-4] + "_" + str(random_id).zfill(3) + ".tif"
        # )
        # example["instance"] = mask_crop[np.newaxis, ...]
        return example
