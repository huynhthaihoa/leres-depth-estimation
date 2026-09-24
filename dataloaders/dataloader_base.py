import copy
import random
import cv2
import torch
import math
import numpy as np
import torch.distributed as dist
import albumentations as A
import pandas as pd

from loguru import logger
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, Sampler, DataLoader
from scipy.ndimage import rotate

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor(object):
    def __init__(self, mode, use_imagenet_normalize: bool):
        """Convert a preprocessed sample into tensors

        Args:
            mode: one of "train", "eval", "test"
            use_imagenet_normalize: if True, apply ImageNet normalization on the image
        """
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if use_imagenet_normalize else None

    def __call__(self, sample):
        # build a new dict so cached samples are never overwritten with tensors
        sample = dict(sample)
        image = self.to_tensor(sample["image"])
        if self.normalize is not None:
            image = self.normalize(image)
        sample["image"] = image
        if "depth" in sample:
            sample["depth"] = self.to_tensor(sample["depth"])
        return sample

    def to_tensor(self, pic):
        """Convert numpy array (H, W, C) or (H, W) into float tensor (C, H, W)"""
        if not _is_numpy_image(pic):
            raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))
        if pic.ndim == 2:
            pic = pic[:, :, np.newaxis]
        return torch.from_numpy(np.ascontiguousarray(pic.transpose((2, 0, 1)))).float()


def preprocessing_transforms(mode, use_imagenet_normalize: bool):
    return transforms.Compose([
        ToTensor(mode=mode, use_imagenet_normalize=use_imagenet_normalize)
    ])

class DistributedSamplerNoEvenlyDivisible(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        num_samples = int(math.floor(len(self.dataset) * 1.0 / self.num_replicas))
        rest = len(self.dataset) - num_samples * self.num_replicas
        if self.rank < rest:
            num_samples += 1
        self.num_samples = num_samples
        self.total_size = len(dataset)
        # self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        # assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class BaseDataset(Dataset):
    def __init__(self, args, mode, transform=None):#, is_for_online_eval=False):      
        self.args = args

        self.use_cache = args.use_cache 
        
        self.original_scale = args.original_scale
        
        self.max_depth = args.max_depth
        
        self.mode = mode
        if self.mode == "eval":
            self.paths = pd.read_csv(args.filenames_file_eval)
        else:
            self.paths = pd.read_csv(args.filenames_file)
        # if self.mode == "eval":
        #     with open(args.filenames_file_eval, "r") as f:
        #         self.paths = f.readlines()
        # else:
        #     with open(args.filenames_file, "r") as f:
        #         self.paths = f.readlines()   
                
        if self.mode == "train" and args.use_albumentations:
            self.albumentations_transform = A.Compose([
                A.ISONoise(p=random.uniform(0, 1)),
                A.RandomBrightnessContrast(p=random.uniform(0, 1)),
                A.RandomGamma(p=random.uniform(0, 1)),
                A.GaussNoise(p=random.uniform(0, 1)),
                A.CLAHE(p=random.uniform(0, 1)),
                A.HueSaturationValue(p=random.uniform(0, 1)),
                A.Blur(p=random.uniform(0, 1)),
                A.ColorJitter(p=random.uniform(0, 1))]) 
        else:
            self.albumentations_transform = None
                
        self.transform = transform
        # self.is_for_online_eval = is_for_online_eval  
            
    def __getitem__(self, idx):
        image_path = self.paths.iloc[idx].image_path
        if self.mode == "test":
            return [image_path]
        depth_path = self.paths.iloc[idx].depth_path
        return [image_path, depth_path]
        
        #raise NotImplementedError
        
    def __len__(self):
        return len(self.paths)

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth
        
    def preprocess_image(self, image):
        """Preprocess image

        Args:
            image: input image (PIL Image)

        Returns:
            image as numpy array (H, W, 3) with value scaled into [0, 1]
        """        
        if self.albumentations_transform is not None:
            image = self.albumentations_transform(image=np.asarray(image, dtype=np.uint8))["image"]
        image_norm = np.asarray(image, dtype=np.float32) #/ 255.0
        if self.args.to_grayscale:
            if self.mode == "train": #augmentation - extract any 1 from 3 input channels R, G, B, or convert them to grayscale using OpenCV flag RGB2GRAY
                channel = random.randint(0, 3)
                if channel == 3:
                    image_norm = cv2.cvtColor(image_norm, cv2.COLOR_RGB2GRAY)
                else:
                    image_norm = image_norm[:, :, channel]
            else:
                image_norm = cv2.cvtColor(image_norm, cv2.COLOR_RGB2GRAY)
            image_norm = np.repeat(image_norm[..., np.newaxis], 3, axis=2)
        if self.original_scale:
            return image_norm
        return image_norm / 255.0
    
    def preprocess_depth(self, depth):
        """Preprocess ground truth depth map

        Args:
            depth: input ground truth depth map

        Raises:
            NotImplementedError
        """        
        raise NotImplementedError

    def apply_augmentation(self, image, depth_gt):
        """Apply augmentation on input data

        Args:
            image: input image
            depth_gt: input ground truth depth map

        Returns:
            augmented (image, depth_gt) pair
        """        
        # Random flipping
        image, depth_gt = self.augment_horizontalflip(image, depth_gt)
        # do_flip = random.random()
        # if do_flip > self.args.flip_prob_thres:
        #     image = (image[:, ::-1, :]).copy()
        #     depth_gt = (depth_gt[:, ::-1, :]).copy()
    
        # Random gamma, brightness, color augmentation
        image = self.augment_imagejitter(image)
        # do_augment = random.random()
        # if do_augment > self.args.image_transform_prob_thres:
            
        # Cut flip
        image, depth_gt = self.augment_cutflip(image, depth_gt)
        # horizontal_delta = random.random() - self.args.cut_flip_horizontal_prob_thres
        # vertical_delta = random.random() - self.args.cut_flip_vertical_prob_thres
        # if horizontal_delta > vertical_delta and horizontal_delta > 0:
        #     image, depth_gt = self.cut_flip_horizontal(image, depth_gt)
        # elif vertical_delta > horizontal_delta and vertical_delta > 0:
        #     image, depth_gt = self.cut_flip_vertical(image, depth_gt)
        # elif horizontal_delta == vertical_delta and horizontal_delta > 0:
        #     choice = random.choice(("horizontal", "vertical"))
        #     if choice == "horizontal":
        #         image, depth_gt = self.cut_flip_horizontal(image, depth_gt)
        #     else:
        #         image, depth_gt = self.cut_flip_vertical(image, depth_gt)
        
        return image, depth_gt   
    
    def augment_imagejitter(self, image):
        """Apply agumentation on input image

        Args:
            image: input image

        Returns:
            augmented image
        """        
        do_jitter = random.random()
        if do_jitter <= self.args.image_transform_prob_thres:
            return image

        if self.albumentations_transform is None:
        #     image_aug = self.albumentations_transform(image=image)["image"]
        # else:
            # gamma augmentation
            gamma = random.uniform(self.args.min_gamma, self.args.max_gamma)
            image_aug = image ** gamma

            # brightness augmentation
            brightness = random.uniform(self.args.min_brightness, self.args.max_brightness)
            image_aug = image_aug * brightness

            # color augmentation
            colors = np.random.uniform(self.args.min_color, self.args.max_color, size=3)
            white = np.ones((image.shape[0], image.shape[1]))
            
            if self.original_scale:
                white = white * 255.0
                
            color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
            image_aug *= color_image
            
            if self.original_scale:
                image_aug = np.clip(image_aug, 0, 255)
            else:
                image_aug = np.clip(image_aug, 0, 1)

            return image_aug
        
        return image

    def rotate_image(self, image, angle, flag=Image.BILINEAR):#cv2.INTER_LINEAR):
        result = image.rotate(angle, resample=flag)
        return result
    
    def rotate_depth(self, depth, angle): #rotate depth (numpy array as shape (H, W))
        return rotate(depth, angle)
    
    def augment_horizontalflip(self, image, depth_gt):
        do_flip = random.random()
        if do_flip > self.args.flip_prob_thres:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()   
        return image, depth_gt     
    
    def augment_cutdepth(self, image, depth):
        do_cutdepth = random.random()
        if do_cutdepth <= self.args.cut_depth_prob_thres:
            return image

        a, b, c, d = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
        l, u = int(a * self.input_width), int(b * self.input_height)
        # w, h = int(max((self.input_width - a * self.input_width) * c * self.args.cut_depth_thres, 1)), int(max((self.args.input_height - b * self.args.input_height) * d * self.args.cut_depth_thres, 1))
        
        w = max(1, int((1 - a) * self.input_width * c * self.args.cut_depth_thres)) #int(max(((1 - a) * self.input_width) * c * self.args.cut_depth_thres, 1))
        h = max(1, int((1 - b) * self.input_height * d * self.args.cut_depth_thres))

        # # print("cutdepth range:", l, u, w, h)
        
        # normalize depth to [0, 1]
        depth_copied = np.repeat(depth, 3, axis=2).astype(np.float32) / self.max_depth
        
        # normalize depth to [0, 255]
        if self.original_scale:
            depth_copied = depth_copied * 255.0
        
        M = np.ones(image.shape)
        M[l : l + h, u : u + w, :] = 0
        image = M * image + (1 - M) * depth_copied
        image = image.astype(np.float32)   
        return image 

    def cut_flip_horizontal(self, image, depth):

        # p = random.random()
        # if p<self.args.cut_flip_horizontal_prop:
        #     return image, depth
        image_copy = copy.deepcopy(image)
        depth_copy = copy.deepcopy(depth)
        h, _, _ = image.shape
        N = 2    # split numbers
        h_list=[]      
        h_interval_list = []        # hight interval
        
        for i in range(N-1):
            h_list.append(random.randint(int(0.2*h),int(0.8*h)))
            
        h_list.append(h)
        h_list.append(0)  
        h_list.sort()
        h_list_inv = np.array([h]*(N+1))-np.array(h_list)
        for i in range(len(h_list)-1):
            h_interval_list.append(h_list[i+1]-h_list[i])

        for i in range(N):
            image[h_list[i]:h_list[i+1],:,:] = image_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i],:,:].copy()
            depth[h_list[i]:h_list[i+1],:,:] = depth_copy[h_list_inv[i]-h_interval_list[i]:h_list_inv[i],:,:].copy()
        return image, depth
    
    def cut_flip_vertical(self, image, depth):

        # p = random.random()
        # if p<self.args.cut_flip_vertical_prop:
        #     return image, depth
        image_copy = copy.deepcopy(image)
        depth_copy = copy.deepcopy(depth)
        _, w, _ = image.shape
        N = 2    # split numbers
        w_list = []      
        w_interval_list = []        # hight interval
        
        for i in range(N-1):
            w_list.append(random.randint(int(0.2*w), int(0.8*w)))
            
        w_list.append(w)
        w_list.append(0)  
        w_list.sort()
        w_list_inv = np.array([w]*(N+1))-np.array(w_list)
        for i in range(len(w_list)-1):
            w_interval_list.append(w_list[i+1] - w_list[i])

        for i in range(N):
            image[:, w_list[i]:w_list[i+1],:] = image_copy[:, w_list_inv[i] - w_interval_list[i] : w_list_inv[i], :].copy()
            depth[:, w_list[i]:w_list[i+1],:] = depth_copy[:, w_list_inv[i] - w_interval_list[i] : w_list_inv[i], :].copy()
        return image, depth
    
    def augment_cutflip(self, image, depth_gt):
        horizontal_delta = random.random() - self.args.cut_flip_horizontal_prob_thres
        vertical_delta = random.random() - self.args.cut_flip_vertical_prob_thres
        if horizontal_delta > vertical_delta and horizontal_delta > 0:
            image, depth_gt = self.cut_flip_horizontal(image, depth_gt)
        elif vertical_delta > horizontal_delta and vertical_delta > 0:
            image, depth_gt = self.cut_flip_vertical(image, depth_gt)
        elif horizontal_delta == vertical_delta and horizontal_delta > 0:
            choice = random.choice(("horizontal", "vertical"))
            if choice == "horizontal":
                image, depth_gt = self.cut_flip_horizontal(image, depth_gt)
            else:
                image, depth_gt = self.cut_flip_vertical(image, depth_gt)
        
        return image, depth_gt             
    # def cut_mix(self, sample, idx):
    #     """apply cutmix augmentation

    #     Args:
    #         sample: current sample
    #         idx: index of current sample to prevent duplication

    #     Returns:
    #         aug_sample: cutmix sample
    #     """        
    #     do_cutmix = random.random()
    #     if do_cutmix <= self.args.cut_mix_prob_thres:
    #         return sample
        
    #     # get random sample
    #     while True:
    #         index = random.choice(range(len(self)))
    #         if index != idx:
    #             second_sample = self[index]#self.paths[index]
    #             break
        
    #     augment_sample = sample.copy() #deep copy
        
    #     second_image = second_sample["image"] #(3, H, W)
    #     second_depth = second_sample["depth"] #(1, H, W)
        
    #     _, input_height, input_width = second_image.shape
        
    #     a, b, c, d = random.uniform(0,1), random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)
    #     l, u = int(a * input_width), int(b * input_height)
    #     w, h = int(max((input_width - a * input_width) * c * 0.75, 1)), int(max((input_height - b * input_height) * d * 0.75, 1))
        
    #     augment_sample["image"][:, l : l + h, u : u + w] = second_image[:, l : l + h, u : u + w]
    #     augment_sample["depth"][:, l : l + h, u : u + w] = second_depth[:, l : l + h, u : u + w]
            
    #     return augment_sample

class CutMixDataset(Dataset):
    def __init__(self, dataset, args):#, mode, transform=None):
        self.dataset = dataset#Dataset(args, mode, transform)
        
        # self.num_classes = len(args.class_list)
        # self.num_mix = args.cut_mix_num_mix
        self.prob = args.cut_mix_prob_thres
        self.size_thres = args.cut_mix_thres
        
    def __getitem__(self, index):
        sample = self.dataset[index]
        do_cutmix = random.random()
        if do_cutmix <= self.prob:
            # print("no cutmix:", index)
            return sample
        
        
        # get random sample
        while True:
            index2 = random.choice(range(len(self)))
            if index2 != index:
                second_sample = self.dataset[index2]#self.paths[index]
                break

        image = sample["image"]
        depth = sample["depth"]
                
        second_image = second_sample["image"] #(3, H, W)
        second_depth = second_sample["depth"] #(1, H, W)
        
        # print("have cutmix:", index, index2)
        
        # # print("image shape:", image.shape, second_image.shape)
        # # print("depth shape:", depth.shape, second_depth.shape)
        _, H, W = second_image.shape # (H, W)
        
        a, b, c, d = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
        l = int(a * W)
        u = int(b * H)
        w = max(1, int((1 - a) * W * c * self.size_thres)) #int(max(((1 - a) * self.input_width) * c * self.args.cut_depth_thres, 1))
        h = max(1, int((1 - b) * H * d * self.size_thres))
        
        # print("cutmix range:", l, u, w, h)
        
        # # print("image before:", image)
        
        image[:, l : l + h, u : u + w] = second_image[:, l : l + h, u : u + w]
        depth[:, l : l + h, u : u + w] = second_depth[:, l : l + h, u : u + w]

        # # print("image after:", image)
        
        # exit(0)
        
        return {"image": image, "depth": depth}

    def __len__(self):
        return len(self.dataset)
       
class BaseDataLoader(object):
    def __init__(self, args, mode):
        if mode not in ("train", "eval", "test"):
            logger.error('mode should be one of \'train, test, eval\'. Got {}'.format(mode))
            raise TypeError
        
        # if args.original_scale:
        #     self.transform = preprocessing_transforms(mode, False)
        # else:
        self.transform = preprocessing_transforms(mode, (args.normalize and not args.original_scale))
            
        # self.training_samples = None
        # self.testing_samples = None

    def __len__(self):
        return len(self.data_loader)
    
    def setup_loader(self, dataset, args, mode):
        if mode == "train":
            dataset = CutMixDataset(dataset, args)
            batch_size = args.batch_size
            shuffle = args.shuffle
            num_workers = args.num_threads
        else:
            batch_size = 1
            shuffle = False
            num_workers = 1            
        
        if mode in ["train", "eval"]:
            pin_memory = True
        else:
            pin_memory = False
        
        if mode in ["train", "eval"] and args.distributed:
            if mode == "train":
                self.data_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            else:
                self.data_sampler = DistributedSamplerNoEvenlyDivisible(dataset, shuffle=shuffle)
        else:
            self.data_sampler = None
        
        self.data_loader = DataLoader(dataset=dataset, 
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                sampler=self.data_sampler)            
        