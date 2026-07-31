import os
import torch
import cv2 
# import albumentations as A
import random
import numpy as np
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import DistributedSamplerNoEvenlyDivisible

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
need_shuffle = False

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
        
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples, shuffle=need_shuffle)
            else:
                self.train_sampler = None
    
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=need_shuffle,
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
            with open(args.filenames_file_eval, "r") as f:
                self.samples = f.readlines()
        else:
            with open(args.filenames_file, "r") as f:
                self.samples = f.readlines()
    
        self.mode = mode
        
        self.transform = transform
        # self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
                    
    def __getitem__(self, idx):
        if self.mode == "test":
            image_path = self.samples[idx]#[:-1]
            if idx < len(self.samples) - 1:
                image_path = image_path[:-1]
        else:
            image_path, depth_path = self.samples[idx].split()
        
        full_image_path = f"{self.args.data_path}/{image_path}"
        image = cv2.imread(full_image_path)[:, :, ::-1]
        depth_gt = np.load(f"{self.args.data_path}/{depth_path}")
                
        if self.mode == "train":            
            
            if self.args.do_random_rotate:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=cv2.INTER_NEAREST)#, cv2.INTER_NEAREST)
            
            image = self.normalize_image(image)
            depth_gt = self.normalize_depth(depth_gt)  
            image = self.cut_depth(image, depth_gt)
            
            image, depth_gt = self.train_preprocess(image, depth_gt)

            # if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
            #     image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width) 
            
            sample = {'image': image, 'depth': depth_gt} 
        
        elif self.mode == "online_eval":
            has_valid_depth = True
            if has_valid_depth:
                image = self.normalize_image(image)
                depth_gt = self.normalize_depth(depth_gt)  
            sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': has_valid_depth, 'name': full_image_path} 
        
        else:
            image = self.normalize_image(image)
            sample = {'image': image, 'name': full_image_path}
        
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
        return image  
    
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=flag)
        return result
    
    def normalize_image(self, image):
        return np.asarray(image, dtype=np.float32) / 255.0
    
    def normalize_depth(self, depth_gt):
        depth_gt = np.array(depth_gt, dtype=np.float32)
        
        # Expand dim
        depth_gt = np.expand_dims(depth_gt, axis=2)
        return depth_gt
    
    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
    
        return image, depth_gt
    
    def augment_image(self, image):
        
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        brightness = random.uniform(0.75, 1.25)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.samples)

class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize([352, 1120])
        self.padding_value = None
        
    def add_padding(self, image):
        """Add padding to the image to ensure both width and height are divisible by 32 (use after converting the image to tensor)
        """        
        if self.padding_value is None:
            H = image.shape[1]
            W = image.shape[2]
            offset_W = (32 - W % 32) % 32
            offset_H = (32 - H % 32) % 32
            offset_L = offset_W // 2
            offset_R = offset_W - offset_L
            offset_U = offset_H // 2
            offset_D = offset_H - offset_U
            self.padding_value = (offset_L, offset_U, offset_R, offset_D)
        return transforms.Pad(self.padding_value)(image)
        
    def __call__(self, sample):
        image = sample['image']
        image = self.to_tensor(image)
        if self.mode == 'test':
            image = self.add_padding(image)
        image = self.normalize(image)
        
        if self.mode != 'train':
            name = sample['name']
        
        if self.mode == 'test':
            return {'image': image, 'name': name, 'U' : self.padding_value[1], 'D' : self.padding_value[3], 'L' : self.padding_value[0], 'R': self.padding_value[2]}#, 'focal': focal}

        depth = sample['depth']

        # depth = self.to_tensor(sample['depth'])
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth}#, 'focal': focal}
        else:
            has_valid_depth = sample['has_valid_depth']
            # depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'has_valid_depth': has_valid_depth, 'name': name}#, 'focal': focal
    
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
