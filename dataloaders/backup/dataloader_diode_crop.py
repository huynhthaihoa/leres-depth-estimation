import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os
import random

from utils import DistributedSamplerNoEvenlyDivisible#, crop_center
# from scipy.ndimage.interpolation import rotate
# import cv2

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cut_choice = ["depth", "flip", "none"]

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class NewDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=True,#(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)
        
        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))
            
            
class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()
    
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        self.max_depth = args.max_depth
        self.min_depth = args.min_depth
        self.min_col_bound = None
        self.min_row_bound = None
    
    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        if self.mode == 'train':
            image_path, depth_path, mask_path, width_index, height_index = sample_path.split()
        else:
            image_path, depth_path, mask_path = sample_path.split()
        image_path = os.path.join(self.args.data_path, image_path)
        depth_path = os.path.join(self.args.data_path, depth_path)
        mask_path = os.path.join(self.args.data_path, mask_path)
        image = Image.open(image_path)
        depth_gt = np.load(depth_path)
        mask_gt = np.load(mask_path)
        mask_gt = mask_gt.astype(bool)
        depth_gt[mask_gt == False] = 0
        
        if self.mode == 'train':
            depth_gt = depth_gt.squeeze()
            depth_gt = Image.fromarray(depth_gt)
            width_index = eval(width_index)
            height_index = eval(height_index)
            # crop the input image, depth map, and mask into the size of (input_width, input_height)
            image = image.crop((width_index, height_index, width_index + self.args.input_width, height_index + self.args.input_height))
            depth_gt = depth_gt.crop((width_index, height_index, width_index + self.args.input_width, height_index + self.args.input_height))
            
            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
            
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            
            # if image_path.find("indoor") != -1: # cutdepth augmentation for indoor dataset
            #     image, depth_gt = self.cut_depth(image, depth_gt)
            # else: # cutflip augmentation for outdoor dataset
            #     image, depth_gt = self.cut_flip(image, depth_gt)
            
            image, depth_gt = self.train_augment(image, depth_gt, image_path)
            sample = {'image': image, 'depth': depth_gt}
            
        else:
            image = np.asarray(image, dtype=np.float32) / 255.0
            if self.mode == 'online_eval':
                try:
                    has_valid_depth = True
                except IOError:
                    has_valid_depth = False
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                # print("online eval: ", mask_gt.shape)
                sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': has_valid_depth, 'name': image_path}
            else:
                sample = {'image': image, 'name': image_path}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def cut_depth(self, image, depth):
        a, b, c, d = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
        l, u = int(a * self.args.input_width), int(b * self.args.input_height)
        w, h = int(max((self.args.input_width - a * self.args.input_width) * c * 0.75, 1)), int(max((self.args.input_height - b * self.args.input_height) * d * 0.75, 1))
        depth_copied = np.repeat(depth, 3, axis=2)
        M = np.ones(image.shape)
        M[l : l + h, u : u + w, :] = 0
        image = M * image + (1 - M) * depth_copied
        image = image.astype(np.float32)   
        return image, depth 
    
    def cut_flip(self, image, depth):
        image_copy = image.copy()
        depth_copy = depth.copy()
        h, _, _ = image.shape

        N = 2     
        h_list = []
        h_interval_list = []   # high interval
        for i in range(N - 1):
            h_list.append(random.randint(int(0.2 * h), int(0.8 * h)))
        h_list.append(h)
        h_list.append(0)  
        h_list.sort()
        h_list_inv = np.array([h] * (N + 1)) - np.array(h_list)
        for i in range(len(h_list) - 1):
            h_interval_list.append(h_list[i + 1] - h_list[i])
        for i in range(N):
            image_copy[h_list[i]: h_list[i + 1], :, :] = image[h_list_inv[i] - h_interval_list[i]:h_list_inv[i], :, :]
            depth_copy[h_list[i]: h_list[i + 1], :, :] = depth[h_list_inv[i] - h_interval_list[i]:h_list_inv[i], :, :]

        return image_copy, depth_copy 
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result
    
    def train_augment(self, image, depth_gt, image_path):
        
        # Random cut
        do_cut = random.choice(cut_choice)
        if do_cut == "depth":
            image, depth_gt = self.cut_depth(image, depth_gt)
        elif do_cut == "flip":
            image, depth_gt = self.cut_flip(image, depth_gt)
        #             if image_path.find("indoor") != -1: # cutdepth augmentation for indoor dataset
        #         image, depth_gt = self.cut_depth(image, depth_gt)
        #     else: # cutflip augmentation for outdoor dataset
        #         image, depth_gt = self.cut_flip(image, depth_gt)

        
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image, image_path)
    
        return image, depth_gt
    
    def augment_image(self, image, image_path):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if image_path.find("indoor") != -1:
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
            
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
    
    def __len__(self):
        return len(self.filenames)

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)
        depth = sample['depth']
        
        if self.mode != 'train':
            name = sample['name']
        
        if self.mode == 'test':
            return {'image': image, 'name': name}

        # depth = self.to_tensor(sample['depth'])
        # mask = self.to_tensor(sample['mask'])
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'has_valid_depth': has_valid_depth, 'name': name}
    
    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
        
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img
        
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
