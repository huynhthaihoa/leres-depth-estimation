from loguru import logger
import os
import random
import numpy as np

from PIL import Image

from .dataloader_base import BaseDataset, BaseDataLoader

class KittiDataLoader(BaseDataLoader):
    def __init__(self, args, mode):
        super().__init__(args, mode)
        
        dataset = KittiDataset(args, mode, transform=self.transform)
        # if mode == "train":
        #     self.training_samples = KittiDataset(args, mode, transform=self.transform)
        #     # if args.distributed:
        #     #     self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples, shuffle=args.shuffle)
        #     # else:
        #     #     self.train_sampler = None
    
        #     # self.data = DataLoader(self.training_samples, args.batch_size,
        #     #                        shuffle=args.shuffle,
        #     #                        num_workers=args.num_threads,
        #     #                        pin_memory=True,
        #     #                        sampler=self.train_sampler)

        # elif mode == "eval":
        #     self.testing_samples = KittiDataset(args, mode, transform=self.transform)
        #     # if args.distributed:
        #     #     # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
        #     #     self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
        #     # else:
        #     #     self.eval_sampler = None
        #     # self.data = DataLoader(self.testing_samples, 1,
        #     #                        shuffle=False,
        #     #                        num_workers=1,
        #     #                        pin_memory=True,
        #     #                        sampler=self.eval_sampler)
        
        # else:#if mode == "test":
        #     self.testing_samples = KittiDataset(args, mode, transform=self.transform)
            # self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)
        
        self.setup_loader(dataset, args, mode)
    
class KittiDataset(BaseDataset):
    def __init__(self, args, mode, transform=None):#, is_for_online_eval=False):
        super().__init__(args, mode, transform)#, is_for_online_eval)   
        
        # use cache for train + eval
        if self.args.use_cache and self.mode in ["train", "eval"]:
            self.caches = dict()
            self.use_cache = True
        else:
            self.use_cache = False

    def __getitem__(self, idx):
        # paths = super().__getitem__(idx)
        # sample_path = self.samples[idx]
        # focal = float(sample_path.split()[2])
        # focal = 518.8579
        
        if self.use_cache and idx in self.caches.keys():
            if self.mode == "train":
                image = self.caches[idx]["image"]
                depth_gt = self.caches[idx]["depth"]

                # if rotate, cache image and depth have not been preprocessed                
                if self.args.do_random_rotate is True:
                    random_angle = (random.random() - 0.5) * 2 * self.args.degree
                    image = self.rotate_image(image, random_angle)
                    depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                    image = self.preprocess_image(image)#np.asarray(image, dtype=np.float32) / 255.0
                    depth_gt = self.preprocess_depth(depth_gt)
                # otherwise, image and depth have been preprocessed

                if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                    image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
                image, depth_gt = self.apply_augmentation(image, depth_gt)
                sample = {"image": image, "depth": depth_gt} 
            else: 
                sample = self.caches[idx]
        else:
            image_path = super().__getitem__(idx)[0]
            if self.mode == "train":
                # rgb_file = paths[0]
                depth_path = image_path.replace('/image_02/data/', '/proj_depth/groundtruth/image_02/')
                if self.args.use_right is True and random.random() > 0.5:
                    image_path.replace('image_02', 'image_03')
                    depth_path.replace('image_02', 'image_03')

                image_path = os.path.join(self.args.data_path, 'raw_dataset', image_path)
                depth_path = os.path.join(self.args.data_path, "train", depth_path)
        
                image = Image.open(image_path)
                depth_gt = Image.open(depth_path)
            
                if self.args.do_kb_crop is True:
                    height = image.height
                    width = image.width
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                    image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                    
                if self.use_cache:
                    self.caches[idx] = dict()
                
                if self.args.do_random_rotate:
                    
                    # has random rotate, save cache before rotate
                    if self.use_cache:
                        self.caches[idx]["image"] = image
                        self.caches[idx]["depth"] = depth_gt
                        
                    random_angle = (random.random() - 0.5) * 2 * self.args.degree
                    image = self.rotate_image(image, random_angle)
                    depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)
                
                image = self.preprocess_image(image)#np.asarray(image, dtype=np.float32) / 255.0
                depth_gt = self.preprocess_depth(depth_gt)

                # no random rotate, save cache after preprocess
                if self.use_cache and not self.args.do_random_rotate:
                    self.caches[idx]["image"] = image
                    self.caches[idx]["depth"] = depth_gt

                if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                    image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)
                image, depth_gt = self.apply_augmentation(image, depth_gt)
                sample = {"image": image, "depth": depth_gt} 
            
            else:
                data_path = self.args.data_path_eval

                image_path = os.path.join(data_path, "raw_dataset/" + image_path) #sample_path.split()[0])
                image = Image.open(image_path)
                image = self.preprocess_image(image)#np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

                if self.mode == "eval":
                    depth_path = image_path.replace('raw_dataset', 'val').replace('image_02/data', 'proj_depth/groundtruth/image_02')
                    # depth_path = os.path.join(data_path, "val/" + sample_path.split()[1])
                    has_valid_depth = False
                    try:
                        depth_gt = Image.open(depth_path)
                        has_valid_depth = True
                    except IOError:
                        logger.warning('Missing gt for {}'.format(image_path))
                        depth_gt = False
                        # print('Missing gt for {}'.format(image_path))

                    if has_valid_depth:
                        depth_gt = self.preprocess_depth(depth_gt)
                        # depth_gt = np.asarray(depth_gt, dtype=np.float32)
                        # depth_gt = np.expand_dims(depth_gt, axis=2)
                        # depth_gt = depth_gt / 256.0

                if self.args.do_kb_crop is True:
                    height = image.shape[0]
                    width = image.shape[1]
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    if self.mode == "eval" and has_valid_depth:
                        depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                
                if self.mode == "eval":
                    if self.use_cache:
                        self.caches[idx] = dict()
                        self.caches[idx]["image"] = image
                        self.caches[idx]["depth"] = depth_gt
                        self.caches[idx]['has_valid_depth'] = has_valid_depth
                        self.caches[idx]['image_path'] = image_path
                    sample = {"image": image, "depth": depth_gt, 'has_valid_depth': has_valid_depth, "name": image_path} 
                else:
                    sample = {"image": image, "name": image_path}
        
        if self.transform:
            sample = self.transform(sample)
            
        # if self.mode == "train":# and self.args.use_cutmix:
        #     return self.cut_mix(sample, idx)          

        return sample
    
    # def preprocess_image(self, image):
    #     return np.asarray(image, dtype=np.float32) / 255.0

    def preprocess_depth(self, depth):
        depth = np.asarray(depth, dtype=np.float32)
        # Expand dim
        depth = np.expand_dims(depth, axis=2)
        depth = depth / 256.0
        
        return depth
            
    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    # def apply_augmentation(self, image, depth_gt):        
    #     image, depth_gt = super().apply_augmentation(image, depth_gt)
        
    #     # Cutflip augmentation
    #     image, depth_gt = self.cut_flip_horizontal(image, depth_gt)
    
    #     return image, depth_gt
    
    # def augment_image(self, image):
    #     # gamma augmentation
    #     gamma = random.uniform(0.9, 1.1)
    #     image_aug = image ** gamma

    #     # brightness augmentation
    #     brightness = random.uniform(0.9, 1.1)
    #     image_aug = image_aug * brightness

    #     # color augmentation
    #     colors = np.random.uniform(0.9, 1.1, size=3)
    #     white = np.ones((image.shape[0], image.shape[1]))
    #     color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    #     image_aug *= color_image
    #     image_aug = np.clip(image_aug, 0, 1)

    #     return image_aug
    

