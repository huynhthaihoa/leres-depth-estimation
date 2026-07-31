import random
import numpy as np

from PIL import Image

from .dataloader_base import BaseDataset, BaseDataLoader

class NyuDataLoader(BaseDataLoader):
    def __init__(self, args, mode):#, is_for_online_eval=False):
        super().__init__(args, mode)#, is_for_online_eval)
        
        dataset = NyuDataset(args, mode, transform=self.transform)
        # if mode == "train":
        #     self.training_samples = NyuDataset(args, mode, transform=self.transform)
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
        #     self.testing_samples = NyuDataset(args, mode, transform=self.transform)
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
        #     self.testing_samples = NyuDataset(args, mode, transform=self.transform)
            # self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)
        self.setup_loader(dataset, args, mode)
              
class NyuDataset(BaseDataset):
    def __init__(self, args, mode, transform=None):#, is_for_online_eval=False):
        super().__init__(args, mode, transform)#, is_for_online_eval)   
       
        if self.mode == "train":
            
            if self.args.input_width >= 640:
                self.input_width = 544
            else:
                self.input_width = self.args.input_width
                
            if self.args.input_height >= 480:
                self.input_height = 416
            else:
                self.input_height = self.args.input_height  
                          
            # self.input_width, self.input_height = 544, 416
            self.min_width_index = 43
            self.max_width_index = 608 - self.input_width #+ 1
            self.min_height_index = 45
            self.max_height_index = 472 - self.input_height #+ 1
        else:
            self.input_width, self.input_height = self.args.input_width, self.args.input_height

        # use cache for eval only
        if self.args.use_cache and self.mode == "eval":
            self.use_cache = True
            self.caches = dict()
        else:
            self.use_cache = False
           
    def __getitem__(self, idx):
        if self.use_cache and idx in self.caches.keys():
            sample = self.caches[idx]
        else:
            selected_paths = super().__getitem__(idx)
            image_path = selected_paths[0]
            if self.mode == "test":
                # image_path = self.paths[idx]#[:-1]
                # image_path = paths[0]#self.paths.image_path.iloc[idx]
                if idx < len(self.selected_paths) - 1:
                    image_path = image_path[:-1]
            else:
                depth_path = selected_paths[1]
                # image_path, depth_path = self.paths[idx].split()
                if self.mode == "eval":
                    depth_gt =  Image.open("{}/{}".format(self.args.data_path_eval, depth_path)) #cv2.imread("{}/{}".format(self.args.data_path, depth_path), cv2.IMREAD_UNCHANGED)
                else:
                    depth_gt =  Image.open("{}/{}".format(self.args.data_path, depth_path)) #cv2.imread("{}/{}".format(self.args.data_path, depth_path), cv2.IMREAD_UNCHANGED)
            
            if self.mode == "eval":
                full_image_path = "{}/{}".format(self.args.data_path_eval, image_path)
            else:
                full_image_path = "{}/{}".format(self.args.data_path, image_path)
            image = Image.open(full_image_path)
                    
            if self.mode == "train":  # no use cache in train mode          
                width_index = random.randint(self.min_width_index, self.max_width_index)
                height_index = random.randint(self.min_height_index, self.max_height_index)
                
                image = image.crop((width_index, height_index, width_index + self.input_width, height_index + self.input_height))
                depth_gt = depth_gt.crop((width_index, height_index, width_index + self.input_width, height_index + self.input_height))
                
                if self.args.do_random_rotate:
                    random_angle = (random.random() - 0.5) * 2 * self.args.degree
                    image = self.rotate_image(image, random_angle)
                    depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)#, cv2.INTER_NEAREST)
                
                image = self.preprocess_image(image)
                depth_gt = self.preprocess_depth(depth_gt)   
               
                # apply cutdepth
                image = self.augment_cutdepth(image, depth_gt)
                
                image, depth_gt = self.apply_augmentation(image, depth_gt)
                
                sample = {"image": image, "depth": depth_gt} 
            
            elif self.mode == "eval":
                has_valid_depth = True
                if has_valid_depth:
                    image = self.preprocess_image(image)
                    depth_gt = self.preprocess_depth(depth_gt)  
                    
                if self.use_cache:
                    self.caches[idx] = dict()
                    self.caches[idx]["image"] = image
                    self.caches[idx]["depth"] = depth_gt
                    self.caches[idx]["has_valid_depth"] = has_valid_depth
                    self.caches[idx]["image_path"] = image_path

                sample = {"image": image, "depth": depth_gt, "has_valid_depth": has_valid_depth, "name": full_image_path} 
            
            else:
                image = self.preprocess_image(image)
                sample = {"image": image, "name": full_image_path}
        
        if self.transform:
            sample = self.transform(sample)   
            
        # if self.mode == "train":# and self.args.use_cutmix:
        #     return self.cut_mix(sample, idx)          
        
        return sample
    
    # def augment_cutdepth(self, image, depth):
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
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        # Expand dim
        depth_gt = np.expand_dims(depth_gt, axis=2)
        depth_gt = depth_gt / 1000.0
        
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
    
    # def augment_horizontalflip(self, image, depth):
    #     image = (image[:, ::-1, :]).copy()
    #     depth = (depth[:, ::-1, :]).copy()
    #     return image, depth