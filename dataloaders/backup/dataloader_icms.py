import random
import numpy as np

# from torch.utils.data import DataLoader
from PIL import Image

from .dataloader_base import BaseDataset, BaseDataLoader

class ICMSDataLoader(BaseDataLoader):
    def __init__(self, args, mode):
        super().__init__(args, mode)
        
        if mode == "train":
            self.training_samples = ICMSDataset(args, mode, transform=self.transform)
        elif mode == "eval":
            self.testing_samples = ICMSDataset(args, mode, transform=self.transform)       
        else:
            self.testing_samples = ICMSDataset(args, mode, transform=self.transform)
        
        self.setup_loader(args, mode)

class ICMSDataset(BaseDataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        super().__init__(args, mode, transform, is_for_online_eval)   
        self.input_width, self.input_height = self.args.input_width, self.args.input_height

    def __getitem__(self, idx):
        image_path, depth_path = self.samples[idx].split()
        if self.mode == "eval":
            full_image_path = "{}/{}".format(self.args.data_path_eval, image_path)
            full_depth_path = "{}/{}".format(self.args.data_path_eval, depth_path)
        else:
            full_image_path = "{}/{}".format(self.args.data_path, image_path)
            full_depth_path = "{}/{}".format(self.args.data_path, depth_path)
        
        image = Image.open(full_image_path)
        depth_gt = np.load(full_depth_path)
        
        if self.mode == "train":
            if self.args.do_random_rotate:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_depth(depth_gt, random_angle)#, flag=Image.NEAREST)#, cv2.INTER_NEAREST)
            image = self.preprocess_image(image)
            depth_gt = self.preprocess_depth(depth_gt)   

            # apply cutdepth
            image = self.cut_depth(image, depth_gt)
            
            image, depth_gt = self.apply_augmentation(image, depth_gt)
            
            sample = {"image": image, "depth": depth_gt} 
        elif self.mode == "eval":
            has_valid_depth = True
            if has_valid_depth:
                image = self.preprocess_image(image)
                depth_gt = self.preprocess_depth(depth_gt)  
                                
            sample = {"image": image, "depth": depth_gt, "has_valid_depth": has_valid_depth, "name": full_image_path} 
        else:
            image = self.preprocess_image(image)
            sample = {"image": image, "name": full_image_path}
        
        if self.transform:
            sample = self.transform(sample)   
            
        if self.mode == "train" and self.args.use_cutmix:
            return self.cut_mix(sample, idx)          
        
        return sample
    
    def preprocess_depth(self, depth_gt):
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        # Expand dim
        depth_gt = np.expand_dims(depth_gt, axis=2)        
        return depth_gt
            
    # def apply_augmentation(self, image, depth_gt):        
    #     image, depth_gt = super().apply_augmentation(image, depth_gt)
        
    #     # Cutflip augmentation
    #     # image, depth_gt = self.cut_flip_vertical(image, depth_gt)
    
    #     return image, depth_gt
