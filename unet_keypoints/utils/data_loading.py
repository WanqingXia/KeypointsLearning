import logging
import os
import pathlib
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
import PIL
from torch.utils.data import Dataset, DataLoader, random_split
import glob
import torch.distributed as dist
from contextlib import contextmanager
import random


def create_dataloader(ycb_dir, d_type, test_folders, batch_size=1, augment=False,
                      sample_num=0, rank=-1, num_workers=8, pin_memory=True):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(ycb_dir, d_type, test_folders, augment=augment)

    if d_type == "train":
        if sample_num == 0:
            sampler = None
        else:
            random_indices = torch.randperm(len(dataset))[:sample_num]
            random_samples = torch.utils.data.RandomSampler(random_indices)
            sampler = random_samples
        # Use torch.utils.data.DataLoader()
        train_dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=sampler,
                            pin_memory=pin_memory,
                            collate_fn=LoadImagesAndLabels.collate_fn)
        return train_dataloader
    elif d_type == "val":
        if sample_num == 0:
            sampler = None
        else:
            # using random seed to keep validation set always the same
            g = torch.Generator()
            g.manual_seed(10)
            random_indices = torch.randperm(len(dataset), generator=g)[:sample_num]
            random_samples = torch.utils.data.RandomSampler(random_indices)
            sampler = random_samples
        val_dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            sampler=sampler,
                            pin_memory=pin_memory,
                            collate_fn=LoadImagesAndLabels.collate_fn)
        return val_dataloader
    elif d_type == "test":
        test_dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=False,
                            collate_fn=LoadImagesAndLabels.collate_fn)
        return test_dataloader
    else:
        raise RuntimeError('The dataset type can only be "train" or "test"')


class LoadImagesAndLabels(Dataset):
    def __init__(self, ycb_dir: str, d_type: str, test_folders: list, augment=False):
        self.real_dir = ycb_dir / 'data'
        self.gen_dir = ycb_dir / 'data_gen'
        self.type = d_type
        self.augment = augment
        self.real_folders = sorted(listdir(self.real_dir))
        self.gen_folders = sorted(listdir(self.gen_dir))
        self.color_dir = []
        self.label_dir = []

        for real_folder, gen_folder in zip(self.real_folders, self.gen_folders):
            if real_folder == gen_folder:
                if self.type == "test":
                    if real_folder in test_folders:
                        self.fill_names(self.color_dir, self.label_dir, self.real_dir, real_folder, self.gen_dir)
                elif self.type == "train" or self.type == "val":
                    if real_folder not in test_folders:
                        self.fill_names(self.color_dir, self.label_dir, self.real_dir, real_folder, self.gen_dir)
                else:
                    raise RuntimeError('The dataset type can only be "train" or "test"')
            else:
                raise RuntimeError(f'The real image folder {real_folder} is different from the generated folder {gen_folder}')

        if not self.color_dir:
            raise RuntimeError(f'No input file found in {ycb_dir}, make sure you put your images there')
        if len(self.color_dir) != len(self.label_dir):
            raise RuntimeError(f'Got {len(self.color_dir)} images but {len(self.label_dir)} labels')
        # logging.info(f'New dataset created with {len(self.color_dir)} examples')

    def __len__(self):
        return len(self.color_dir)

    @staticmethod
    def fill_names(color_list, label_list, real_dir, real_folder, gen_dir):
        # fill the out_list with all color image names in the real folder
        # the name should be the same for real and gen folder
        for file in sorted(glob.glob(os.path.join(real_dir, real_folder, "*-color.png"))):
            color_list.append(file.split('data/')[-1])
        for label in sorted(glob.glob(os.path.join(gen_dir, real_folder, "*-np.npy"))):
            label_list.append(label.split('data_gen/')[-1])

    @staticmethod
    def augment_img(image):
        # no hue gain since we don't want to change the actually color, only brightness to mimic lighting condition
        bright_gain = random.uniform(0.5, 2.0) # saturation gain
        contrast_gain = random.uniform(0.5, 2.0) # value gain

        # adjust the brightness
        img_b = ImageEnhance.Brightness(image).enhance(bright_gain)
        # adjust the contrast
        img_bc = ImageEnhance.Contrast(img_b).enhance(contrast_gain)

        return img_bc

    def load_process(self, filename):
        file_type = filename.stem.split('-')[-1]
        # numpy image: H x W x C
        # torch image: C x H x W
        if file_type == 'color':
            if self.augment:
                real_img = Image.open(filename)
                real_img = self.augment_img(real_img)
                data = np.array(real_img).astype("uint8")
                data = (data.transpose((2, 0, 1))/255).astype(np.float32)
                return torch.as_tensor(data).float().contiguous()
            else:
                data = np.array(Image.open(filename)).astype("uint8")
                data = (data.transpose((2, 0, 1))/255).astype(np.float32)
                return torch.as_tensor(data).float().contiguous()
        elif file_type == 'depth':
            data = np.array(Image.open(filename)).astype("uint16")
            data = (data/10000).astype(np.float32)
            return torch.as_tensor(data).float().contiguous()
        elif file_type == 'np':
            data = np.array(np.load(filename)).astype(np.float32)
            return data


    def __getitem__(self, idx):
        name = self.color_dir[idx]
        label_name = self.label_dir[idx]
        real_img = self.real_dir / name
        real_dep = self.real_dir / name.replace('color', 'depth')
        gen_img = self.gen_dir / name
        gen_dep = self.gen_dir / name.replace('color', 'depth')
        label = self.gen_dir / label_name

        assert Path(real_img).is_file(), f'No color image found for the: {real_img}'
        assert Path(real_dep).is_file(), f'No depth image found for the: {real_dep}'
        assert Path(gen_img).is_file(), f'No color image found for the: {gen_img}'
        assert Path(gen_dep).is_file(), f'No depth image found for the: {gen_dep}'
        assert Path(label).is_file(), f'No depth image found for the: {label}'

        real_img = self.load_process(real_img)
        real_dep = self.load_process(real_dep)
        gen_img = self.load_process(gen_img)
        gen_dep = self.load_process(gen_dep)
        # label has 4 cols, [real_row, real_col, gene_row, gene_col]
        label = self.load_process(label)

        real_dep = real_dep[None, :, :]
        gen_dep = gen_dep[None, :, :]

        return {
            'real': torch.cat((real_img, real_dep), 0),
            'gene': torch.cat((gen_img, gen_dep), 0)
        }, torch.from_numpy(label), self.real_dir / name, self.gen_dir / name

    @staticmethod
    def collate_fn(batch):
        images, labels, real_dir, gen_dir = zip(*batch)  # transposed
        max_size = 0
        labels = list(labels)
        for i, label in enumerate(labels):
            if label.shape[0] > max_size:
                max_size = label.shape[0]
        for num, l in enumerate(labels):
            labels[num] = torch.cat((l, torch.zeros((max_size - l.shape[0], l.shape[1]))), 0)

        return images, tuple(labels), real_dir, gen_dir



@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        dist.barrier()
    yield
    if local_rank == 0:
        dist.barrier()
