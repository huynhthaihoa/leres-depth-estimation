import random
import numpy as np

from PIL import Image

from .dataloader_base import BaseDataset, BaseDataLoader

class NyuDDataLoader(BaseDataLoader):
    def __init__(self, args, mode):
        super().__init__(args, mode)
        
        if mode == "train":
            self.training_samples = NyuDDataset(args, mode, transform=self.transform)
            # if args.distributed:
            #     self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples, shuffle=args.shuffle)
            # else:
            #     self.train_sampler = None
    
            # self.data = DataLoader(self.training_samples, args.batch_size,
            #                        shuffle=args.shuffle,
            #                        num_workers=args.num_threads,
            #                        pin_memory=True,
            #                        sampler=self.train_sampler)

        elif mode == "eval":
            self.testing_samples = NyuDDataset(args, mode, transform=self.transform)
            # if args.distributed:
            #     self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            # else:
            #     self.eval_sampler = None
            # self.data = DataLoader(self.testing_samples, 1,
            #                        shuffle=False,
            #                        num_workers=1,
            #                        pin_memory=True,
            #                        sampler=self.eval_sampler)
        
        else:#if mode == "test":
            self.testing_samples = NyuDDataset(args, mode, transform=self.transform)
            # self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)
        self.setup_loader(args, mode)
              
class NyuDDataset(BaseDataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        super().__init__(args, mode, transform, is_for_online_eval)   
       
        if self.mode == "train":
            
            # if self.args.input_width >= 640:
            # self.input_width = 576
            # else:
            #     self.input_width = self.args.input_width
                
            # if self.args.input_height >= 480:
            # self.input_height = 416
            # else:
            #     self.input_height = self.args.input_height  
                          
            # self.input_width, self.input_height = 544, 416
            self.min_width_index = 43
            self.max_width_index = 608 #- self.input_width #+ 1
            self.input_width = 640
            
            self.min_height_index = 45
            self.max_height_index = 472 #- self.input_height #+ 1
            self.input_height = 480

        else:
            self.input_width, self.input_height = self.args.input_width, self.args.input_height
            
    def __getitem__(self, idx):
        if self.mode == "test":
            image_path = self.samples[idx]#[:-1]
            if idx < len(self.samples) - 1:
                image_path = image_path[:-1]
        else:
            image_path, depth_path = self.samples[idx].split()
            if self.mode == "eval":
                depth_gt = np.load("{}/{}".format(self.args.data_path_eval, depth_path))
            else:
                depth_gt = np.load("{}/{}".format(self.args.data_path, depth_path))
        
        if self.mode == "eval":
            full_image_path = "{}/{}".format(self.args.data_path_eval, image_path)
        else:
            full_image_path = "{}/{}".format(self.args.data_path, image_path)
        image = Image.open(full_image_path)
                
        if self.mode == "train":            
            
            image = image.crop((self.min_width_index, self.min_height_index, self.max_width_index, self.max_height_index))
            depth_gt = depth_gt[self.min_height_index: self.max_height_index, self.min_width_index: self.max_width_index]

            if self.args.do_random_rotate:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_depth(depth_gt, random_angle)#, flag=Image.NEAREST)#, cv2.INTER_NEAREST)
            
            image = self.preprocess_image(image)
            depth_gt = self.preprocess_depth(depth_gt) 
              
            # apply cutdepth
            image = self.cut_depth(image, depth_gt)
            
            image, depth_gt = self.apply_augmentation(image, depth_gt)

            padding_left = random.randint(0, self.input_width + self.min_width_index - self.max_width_index)
            padding_top = random.randint(0, self.input_height + self.min_height_index - self.max_height_index)

            image_pad = np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
            depth_gt_pad = np.zeros((self.input_height, self.input_width, 1), dtype=np.float32)
            
            image_pad[padding_top: padding_top + image.shape[0], padding_left: padding_left + image.shape[1], :] = image
            depth_gt_pad[padding_top: padding_top + depth_gt.shape[0], padding_left: padding_left + depth_gt.shape[1], :] = depth_gt
            
            # apply cutdepth
            # image_pad = self.cut_depth(image_pad, depth_gt_pad)
            
            # image_pad, depth_gt_pad = self.apply_augmentation(image_pad, depth_gt_pad)
           
            sample = {"image": image_pad, "depth": depth_gt_pad} 
        
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
    
    # def cut_depth(self, image, depth):
    #     a, b, c, d = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
    #     l, u = int(a * self.input_width), int(b * self.input_height)
    #     w, h = int(max((self.input_width - a * self.input_width) * c * 0.75, 1)), int(max((self.args.input_height - b * self.args.input_height) * d * 0.75, 1))
    #     depth_copied = np.repeat(depth, 3, axis=2)
    #     M = np.ones(image.shape)
    #     M[l : l + h, u : u + w, :] = 0
    #     image = M * image + (1 - M) * depth_copied
    #     image = image.astype(np.float32)   
    #     return image  
    
    # def rotate_image(self, image, angle, flag=Image.BILINEAR):#cv2.INTER_LINEAR):
    #     result = image.rotate(angle, resample=flag)
    #     return result
    
    def preprocess_depth(self, depth_gt):
        """preprocess NYU ground truth depth map

        Args:
            ground truth depth map (PIL image) with value range is (0, 10000)

        Returns:
            normalized depth map as np array shape (H, W, 1) with depth range (0, 10)
        """        
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        # Expand dim
        depth_gt = np.expand_dims(depth_gt, axis=2)
        # depth_gt = depth_gt / 1000.0
        
        return depth_gt
    
    # def random_crop(self, img, depth, height, width):
    #     assert img.shape[0] >= height
    #     assert img.shape[1] >= width
    #     assert img.shape[0] == depth.shape[0]
    #     assert img.shape[1] == depth.shape[1]
    #     x = random.randint(0, img.shape[1] - width)
    #     y = random.randint(0, img.shape[0] - height)
    #     img = img[y:y + height, x:x + width, :]
    #     depth = depth[y:y + height, x:x + width, :]
    #     return img, depth
    
    def augment_flip(self, image, depth):
        image = (image[:, ::-1, :]).copy()
        depth = (depth[:, ::-1, :]).copy()
        return image, depth