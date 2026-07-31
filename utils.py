import copy
import random
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms
import torch.amp as amp
import torch.nn.functional as F

import os, sys
import numpy as np
import math

import cv2

try:
    import kornia
except:
    pass

try:    
    import matplotlib
except:
    pass

from collections import OrderedDict

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score#, confusion_matrix
except:
    pass

# constant/global variables

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

classifier_eval_metrics = ['accuracy', 'precision', 'recall', 'f1-score']

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

imagenet_transform = transforms.Compose([transforms.ToTensor(),
		                        transforms.Normalize((0.485, 0.456, 0.406) , (0.229, 0.224, 0.225))])

normal_transform = transforms.Compose([transforms.ToTensor()])

# utility functions

# Function for setting the seed
def set_seeds(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__

def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))

def normalize_result(value, to_tensorboard=False, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.
    
    # if to_tensorboard:
    #     return np.expand_dims(value, 0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    # (H, W, C)
    img = value[:, :, :3]

    if to_tensorboard:
        img = img.transpose((2, 0, 1)) # (C, H, W)
        img = torch.from_numpy(img)
    # print(img.shape)
    
    return img#.transpose((2, 0, 1))
    
    # value = (value * 255).astype(np.uint8)
    # return np.expand_dims(value, 2)

def inv_normalize_image(image, to_tensorboard=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    image = (image * 255)
    if to_tensorboard:
        return image.to(dtype=torch.uint8) # (C, H, W)
    image = image.detach().cpu().numpy() # (C, H, W)
    image = np.transpose(image, (1, 2, 0)) # (H, W, C)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return image

def inv_unnormalize_image(image, to_tensorboard=True):
    image = (image * 255)
    if to_tensorboard:
        return image.to(dtype=torch.uint8) # (C, H, W)
    image = image.detach().cpu().numpy() # (C, H, W)
    image = np.transpose(image, (1, 2, 0)) # (H, W, C)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return image

def inv_standard_image(image, to_tensorboard=True):
    if to_tensorboard:
        return image.to(dtype=torch.uint8) # (C, H, W)
    image = image.detach().cpu().numpy() # (C, H, W)
    image = np.transpose(image, (1, 2, 0)) # (H, W, C)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return image

def inv_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=True, to_tensorboard=False):
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
        # The normalize code -> t.sub_(m).div_(s)
    if scale:
        image = (image * 255)
    if to_tensorboard:
        return image.to(dtype=torch.uint8) # (C, H, W)    
    image = image.detach().cpu().numpy() # (C, H, W)
    image = np.transpose(image, (1, 2, 0)) # (H, W, C)
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return image

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

def compute_classification_metrics(gt, pred, average='weighted'):
    accuracy = accuracy_score(gt, pred)
    precision = precision_score(gt, pred, average=average)
    recall = recall_score(gt, pred, average=average)
    f1 = f1_score(gt, pred, average=average)
    
    return [accuracy, precision, recall, f1]

def flip_lr(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])

def generate_padding(image):
    """Generate padding value to the image to ensure both image width and height are divisible by 32 (padding value is None if the original shape already satisfies the condition)

    Args:
        image: input image

    Returns:
        padding_value: padding value
    """    
    H, W = image.shape[:-1]
    
    offset_W = (32 - W % 32) % 32
    offset_H = (32 - H % 32) % 32
    if offset_W == 0 and offset_H == 0:
        return None
    offset_L = offset_W // 2
    offset_R = offset_W - offset_L
    offset_U = offset_H // 2
    offset_D = offset_H - offset_U
    padding_value = (offset_L, offset_U, offset_R, offset_D)
    
    return padding_value 

def remove_padding(depth, padding_values=None):
    """Remove padding from output depth

    Args:
        depth: output depth as shape [H, W]
        padding_values: padding value (if padding value is None, do nothing)

    Returns:
        _type_: _description_
    """    
    if padding_values is None:
        return depth
    return depth[padding_values[1]: depth.shape[0] - padding_values[3], padding_values[0]: depth.shape[1] - padding_values[2]]

def load_ckpt(checkpoint_path, depth_model):
    """
    Load checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    depth_model.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."), strict=True)
    del checkpoint
    torch.cuda.empty_cache()

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def preprocess_image(image, normalize=True, to_grayscale=False):
    """Preprocess input image before feeding into model

    Args:
        image: input image (numpy/OpenCV array)

    Returns:
        - image as tensor with shape [1, C, H, W]
        - padding_values to ensure both H and W are divisible by 32 (is None if input image shape already satisfies the condition)
    """  
    padding_values = generate_padding(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if to_grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.repeat(image[..., np.newaxis], 3, axis=2)
    
    if padding_values is not None:
        image_zeros = np.zeros((image.shape[0] + padding_values[1] + padding_values[3], image.shape[1] + padding_values[0] + padding_values[2], 3), dtype = np.uint8)
        image_zeros[padding_values[1]: padding_values[1] + image.shape[0], padding_values[0]: padding_values[0] + image.shape[1]] = image
        image = image_zeros
    
    if normalize:
        image = imagenet_transform(image)   
    else:
        image = normal_transform(image) 
        
    return image[None, :, :, :], padding_values

def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))

def post_process_depth(depth, depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    B, C, H, W = depth.shape
    inv_depth_hat = flip_lr(depth_flipped)
    inv_depth_fused = fuse_inv_depth(depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=depth.device,
                        dtype=depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused

'''new utility functions'''
def depth_value_to_depth_image(depth_values, min_depth=None, max_depth=None, cmap='magma'):# applyReverse=False):#, applyColor=False):
    """Convert depth value to depth image for visualization (in magma colormap/grayscale)

    Args:
        depth_values: input depth value
        min_depth (default is None): maximum depth value in visualization range (if max_depth is None, it is set as max value of depth_values)
        max_depth (default is None): maximum depth value in visualization range (if max_depth is None, it is set as max value of depth_values)
        cmap: cmap to display, recommended value is 'magma' for short-range depth (like NYUv2) and 'magma_r' for long-range depth (like KITTI)
    Returns:
        depth visualization
    """
    try:    
        cmapper = matplotlib.cm.get_cmap(cmap)
    except:
        raise TypeError
    
    if min_depth is None:
        min_depth = depth_values.min()  #else min_depth
    if max_depth is None:
        max_depth = depth_values.max() #else max_depth

    depth_values_clip = np.clip(depth_values, min_depth, max_depth)
    
    if max_depth != min_depth:
        depth_values_vis = (depth_values_clip - min_depth) / (max_depth - min_depth)
    else:
        depth_values_vis = depth_values_clip * 0.
        
    depth_values_vis = cmapper(depth_values_vis, bytes=True)

    img = depth_values_vis[:, :, :3]
    
    return np.ascontiguousarray(img[:, :, ::-1], dtype=np.uint8)

def remove_padding(depth, padding_values=None):
    """Remove padding from output depth

    Args:
        depth: output depth value as shape [H, W]
        padding_values: padding value (if padding value is None, do nothing)

    Returns:
        _type_: _description_
    """    
    if padding_values is None:
        return depth
    return depth[padding_values[1]: depth.shape[0] - padding_values[3], padding_values[0]: depth.shape[1] - padding_values[2]]

def pixel_unshuffle(fm, r):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    b, c, h, w = fm.shape
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    fm_view = fm.contiguous().view(b, c, out_h, r, out_w, r)
    fm_prime = fm_view.permute(0,1,3,5,2,4).contiguous().view(b, out_channel, out_h, out_w)

    return fm_prime

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)

def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)#, 1, -2)#(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def up_triu(x):
    # return a flattened view of up triangular elements of a square matrix
    n, m = x.shape
    assert n == m
    _tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
    return x[_tmp]

# @torch.jit.script
def pyrdown(input_tensor: torch.Tensor, num_scales: int = 3):
    """ Creates a downscale pyramid for the input tensor. """
    output = [input_tensor]
    for _ in range(num_scales - 1):
        down = kornia.filters.blur_pool2d(output[-1], 3)
        output.append(down)
    return output

# loss classes

class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)

class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total

class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target, mask):

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)

class NewcrfsSilogLoss(nn.Module):
    def __init__(self, variance_focus):
        super(NewcrfsSilogLoss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask=None):
        # if mask is not None:
        #     mask = mask.to(torch.bool)
        depth_gt = depth_gt[mask]
        depth_est = depth_est[mask]
        d = torch.log(depth_est) - torch.log(depth_gt)
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
  
class ZoeSilogLoss(nn.Module):
    """ZoeDepth SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(ZoeSilogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta  
        
    def forward(self, depth_est, depth_gt, mask=None):
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = mask.to(torch.bool)
            depth_est = depth_est[mask]
            depth_gt = depth_gt[mask]

        # with amp.autocast(enabled=False):  # amp causes NaNs in this loss function
        alpha = 1e-7
        g = torch.log(depth_est + alpha) - torch.log(depth_gt + alpha)

        Dg = torch.var(g) + self.beta * torch.pow(torch.mean(g), 2)

        loss = 10 * torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", depth_gt.shape)
            print("G", torch.sum(torch.isnan(g)))
            print("Input min max", torch.min(input), torch.max(input))
            print("Target min max", torch.min(depth_gt), torch.max(depth_gt))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))

        return loss

class VaeSilogLoss(nn.Module):
    def __init__(self, SI_loss_lambda = 0.85, max_depth = 10):
        super(VaeSilogLoss, self).__init__()

        self.SI_loss_lambda = SI_loss_lambda
        self.max_depth = max_depth

    def forward(self, depth_est, depth_gt, min_depth=0.1):

        scale_factor = int(np.sqrt(depth_est.shape[1]))

        reshaped_gt = pixel_unshuffle(depth_gt, scale_factor)

        diff = torch.log(depth_est) - torch.log(reshaped_gt)

        num_pixels = (reshaped_gt > min_depth) * (reshaped_gt < self.max_depth)

        diff = torch.where(
        (reshaped_gt > min_depth) * (reshaped_gt < self.max_depth) * (torch.abs(diff) > 0.001), diff, torch.zeros_like(diff))
        
        diff = diff.reshape(diff.shape[0], -1)
        num_pixels = num_pixels.reshape(num_pixels.shape[0], -1).sum(dim=-1) + 1e-6

        loss = (diff**2).sum(dim = -1) / num_pixels
        loss = loss - self.SI_loss_lambda * (diff.sum(dim = -1) / num_pixels) ** 2

        total_pixels = reshaped_gt.shape[1] * reshaped_gt.shape[2] * reshaped_gt.shape[3]

        weight = num_pixels.to(diff.dtype) / total_pixels

        loss = (loss * weight).sum()

        return loss
 
class OrdinalEntropyLoss(nn.Module):
    """
    OrdinalEntropyLoss: https://github.com/needylove/OrdinalEntropy/blob/main/OrdinalEntropy.py
    """    
    def __init__(self, use_double=False):
        super(OrdinalEntropyLoss, self).__init__()
        self.use_double = use_double
    
    def forward(self, features, gt, min_depth=0.001):
        """
        Features: a certain layer's features
        gt: pixel-wise ground truth values, in depth estimation, gt.size()= n, h, w
        mask: In case values of some pixels do not exist. For depth estimation, there are some pixels lack the ground truth values
        """
        f_n, f_c, f_h, f_w = features.size()
        
        scale = gt.shape[-1] // f_w

        features = F.interpolate(features, size=[f_h // scale, f_w // scale], mode='nearest')
        features = features.permute(0, 2, 3, 1)  # n, h, w, c
        features = torch.flatten(features, start_dim=1, end_dim=2)
        
        gt = F.interpolate(gt, size=[f_h // scale, f_w // scale], mode='nearest')
        
        loss = 0
        
        for i in range(f_n):
            """
            mask pixels that without valid values
            """
            _gt = gt[i,:].view(-1)
            
            _mask = _gt > min_depth
            _mask = _mask.to(torch.bool)

            _gt = _gt[_mask]
            _features = features[i,:]
            _features = _features[_mask,:]
            
            # _gt = gt[i,:].view(-1)
            # _gt = _gt[mask]
            # _features = _features[mask, :]

            """
            diverse part
            """
            u_value, u_index, u_counts = torch.unique(_gt, return_inverse=True, return_counts =True)
            center_f = torch.zeros([len(u_value), f_c]).cuda()
            if self.use_double:
                center_f = center_f.double()
            center_f.index_add_(0, u_index, _features)
            u_counts = u_counts.unsqueeze(1)
            center_f = center_f / u_counts

            p = F.normalize(center_f, dim=1)
            _distance = euclidean_dist(p, p)
            _distance = up_triu(_distance)

            u_value = u_value.unsqueeze(1)
            _weight = euclidean_dist(u_value, u_value)
            _weight = up_triu(_weight)
            if _weight.shape[0] == 0:
                return 0
            _max = torch.max(_weight)
            _min = torch.min(_weight)
            _weight = ((_weight - _min) / _max)

            _distance = _distance * _weight

            _entropy = torch.mean(_distance)
            if not torch.isnan(_entropy):
                loss = loss - _entropy
            # else:
            #     logger.warning("_entropy is NaN")
                
            """
            tightness part
            """
            _features = F.normalize(_features, dim=1)
            _features_center = p[u_index, :]
            _features = _features - _features_center
            _features = _features.pow(2)
            _tightness = torch.sum(_features, dim=1)
            _mask = _tightness > 0
            _tightness = _tightness[_mask]

            _tightness = torch.mean(_tightness)
            
            if not torch.isnan(_tightness):
                loss = loss + _tightness
            # else:
            #     logger.warning("_tightness is NaN")
            
        return loss/ f_n

class VarLoss(nn.Module):
    def __init__(self, depth_channel, feat_channel):
        super(VarLoss, self).__init__()

        self.att = nn.Sequential(
                nn.Conv2d(feat_channel, depth_channel, kernel_size=3, padding=1),
                nn.Sigmoid())

        self.post = nn.Conv2d(depth_channel, 2, kernel_size=3, padding=1)

        self.r = 10  # repeat sample

    def forward(self, x, d, gts):
        loss = 0.0
        for i in range(self.r):
            loss = loss + self.single(x, d, gts)

        return loss / self.r

    def single(self, feat, d, gts):

        ts_shape = d.shape[2:]
        gt = gts.copy()
        #gt = gts.unsqueeze(1)
        #print(gt.shape)
        os_shape = gt.shape[2:]
        n, c, h, w = d.shape

        reshaped_gt, indices = self.random_pooling(gt, ts_shape)

        bias_x = os_shape[1] // ts_shape[1] // 2
        bias_y = os_shape[0] // ts_shape[0] // 2 * os_shape[1]

        indices = indices + bias_x + bias_y
        ind_x = (indices % os_shape[1]).to(d.dtype) / os_shape[1]
        ind_y = (indices // os_shape[1]).to(d.dtype) / os_shape[0]

        ind_x = 2 * (ind_x - 0.5)
        ind_y = 2 * (ind_y - 0.5)
        grid = torch.cat([ind_x, ind_y], 1)
        grid = grid.permute(0, 2, 3, 1)

        feat = F.grid_sample(input=feat, grid=grid, mode='bilinear', align_corners=True)

        att = self.att(feat)

        ds = att * d
        #att = att.permute(0, 2, 3, 1)

        #ds = F.grid_sample(input=d, grid=grid+att, mode='bilinear', align_corners=True)

        ds = self.post(ds)

        loss = self.loss(ds, reshaped_gt)
        return loss
    
    def random_pooling(self, gt_depth, shape):
        #print(gt_depth.shape)
        n, c, h, w = gt_depth.shape
        rand = torch.rand(n, c, h, w, dtype=gt_depth.dtype, device=gt_depth.device)
        mask = gt_depth > 0.1
        rand = rand * mask

        _, indices = F.adaptive_max_pool2d(rand, shape, return_indices=True)

        reshaped_ind = indices.reshape(n, c, -1)
        reshaped_gt = gt_depth.reshape(n, c, h*w)
        reshaped_gt = torch.gather(input=reshaped_gt, dim=-1, index=reshaped_ind)
        reshaped_gt = reshaped_gt.reshape(n, c, indices.shape[2], indices.shape[3])

        reshaped_gt[reshaped_gt < 0.1] = 0
        return reshaped_gt, indices

    def grad(self, image):
        def gradient_y(img):
            gx = torch.log(img[:,:,1:-1,1:-1]+1e-6) - torch.log(img[:,:,2:,1:-1]+1e-6)

            mask = img > 0.1
            mask = torch.logical_and(mask[:,:,1:-1,1:-1], mask[:,:,2:,1:-1])
            return gx, mask

        def gradient_x(img):
            gy = torch.log(img[:,:,1:-1,1:-1]+1e-6) - torch.log(img[:,:,1:-1,2:]+1e-6)

            mask = img > 0.1
            mask = torch.logical_and(mask[:,:,1:-1,1:-1], mask[:,:,1:-1,2:])
            return gy, mask

        image = F.pad(image, (1,1,1,1), 'constant', 0.0)

        image_grad_x, mask_x = gradient_x(image)
        image_grad_y, mask_y = gradient_y(image)

        return image_grad_x, image_grad_y, mask_x, mask_y

    def loss(self, ds, reshaped_gt):

        gx, gy, mx, my = self.grad(reshaped_gt)
        grad_gt = torch.cat([gx, gy], 1)
        grad_mk = torch.cat([mx, my], 1)

        diff = F.smooth_l1_loss(ds, grad_gt, reduce=False, beta=0.01) * grad_mk

        loss_g =  diff.sum() / grad_mk.sum()

        return loss_g

class MSGradientLoss(nn.Module):
    """
    MSGradientLoss: https://github.com/cake-lab/HybridDepth/blob/main/loss.py#L58
    """
    def __init__(self, num_scales: int = 4):
        super().__init__()

        self.num_scales = num_scales

    def forward(self, depth_pred, depth_gt):
        depth_pred_pyr = pyrdown(depth_pred, self.num_scales)
        depth_gtn_pyr = pyrdown(depth_gt, self.num_scales)
        
        grad_loss = torch.tensor(0, dtype=depth_gt.dtype, device=depth_gt.device)
        
        for depth_pred_down, depth_gtn_down in zip(depth_pred_pyr, depth_gtn_pyr):

            depth_gtn_grad = kornia.filters.spatial_gradient(depth_gtn_down)
            # mask_down_b = depth_gtn_grad.isfinite().all(dim=1, keepdim=True)
            # Mask where depth_gt_grad is not zero
            mask_not_zero = depth_gtn_grad != 0
            
            # Making sure the mask includes all channels
            mask_down_b = mask_not_zero.all(dim=1, keepdim=True)

            depth_pred_grad = kornia.filters.spatial_gradient(
                                    depth_pred_down).masked_select(mask_down_b)

            grad_error = torch.abs(depth_pred_grad - 
                                    depth_gtn_grad.masked_select(mask_down_b))
            grad_loss += torch.mean(grad_error)

        return grad_loss

# utility classes

class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.module = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # if hasattr(model, 'module'):
        #     model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    
class Rectifier:
    def __init__(self, camera_matrix, dist_coeffs, valid_value = 1.):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.map_x = None
        self.map_y = None
        self.valid_value = valid_value
        
    def rectify(self, image):
        if self.map_x is None:
            h, w = image.shape[:2]
            newcamera_mtx, _ = cv2.getOptimalNewCameraMatrix( self.camera_matrix,  self.dist_coeffs, (w, h), self.valid_value, (w, h))
            self.map_x, self.map_y = cv2.initUndistortRectifyMap( self.camera_matrix,  self.dist_coeffs, None,  newcamera_mtx, (w, h), 5)
        dst = cv2.remap(image, self.map_x,  self.map_y, cv2.INTER_LINEAR)
        return dst
    
class ModelEmaV3(nn.Module):
    """ Model Exponential Moving Average V3: https://github.com/huggingface/pytorch-image-models/blob/main/timm/utils/model_ema.py#L133

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V3 of this module leverages for_each and in-place operations for faster performance.

    Decay warmup based on code by @crowsonkb, her comments:
      If inv_gamma=1 and power=1, implements a simple average. inv_gamma=1, power=2/3 are
      good values for models you plan to train for a million or more steps (reaches decay
      factor 0.999 at 31.6K steps, 0.9999 at 1M steps), inv_gamma=1, power=3/4 for models
      you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
      215.4k steps).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(
            self,
            model,
            decay: float = 0.9999,
            min_decay: float = 0.0,
            update_after_step: int = 0,
            use_warmup: bool = False,
            warmup_gamma: float = 1.0,
            warmup_power: float = 2/3,
            device: Optional[torch.device] = None,
            foreach: bool = True,
            exclude_buffers: bool = False,
    ):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_warmup = use_warmup
        self.warmup_gamma = warmup_gamma
        self.warmup_power = warmup_power
        self.foreach = foreach
        self.device = device  # perform ema on different device from model if set
        self.exclude_buffers = exclude_buffers
        if self.device is not None and device != next(model.parameters()).device:
            self.foreach = False  # cannot use foreach methods with different devices
            self.module.to(device=device)

    def get_decay(self, step: Optional[int] = None) -> float:
        """
        Compute the decay factor for the exponential moving average.
        """
        if step is None:
            return self.decay

        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0

        if self.use_warmup:
            decay = 1 - (1 + step / self.warmup_gamma) ** -self.warmup_power
            decay = max(min(decay, self.decay), self.min_decay)
        else:
            decay = self.decay

        return decay

    @torch.no_grad()
    def update(self, model, step: Optional[int] = None):
        decay = self.get_decay(step)
        if self.exclude_buffers:
            self.apply_update_no_buffers_(model, decay)
        else:
            self.apply_update_(model, decay)

    def apply_update_(self, model, decay: float):
        # interpolate parameters and buffers
        if self.foreach:
            ema_lerp_values = []
            model_lerp_values = []
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_lerp_values.append(ema_v)
                    model_lerp_values.append(model_v)
                else:
                    ema_v.copy_(model_v)

            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_lerp_values, model_lerp_values, weight=1. - decay)
            else:
                torch._foreach_mul_(ema_lerp_values, scalar=decay)
                torch._foreach_add_(ema_lerp_values, model_lerp_values, alpha=1. - decay)
        else:
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if ema_v.is_floating_point():
                    ema_v.lerp_(model_v.to(device=self.device), weight=1. - decay)
                else:
                    ema_v.copy_(model_v.to(device=self.device))

    def apply_update_no_buffers_(self, model, decay: float):
        # interpolate parameters, copy buffers
        ema_params = tuple(self.module.parameters())
        model_params = tuple(model.parameters())
        if self.foreach:
            if hasattr(torch, '_foreach_lerp_'):
                torch._foreach_lerp_(ema_params, model_params, weight=1. - decay)
            else:
                torch._foreach_mul_(ema_params, scalar=decay)
                torch._foreach_add_(ema_params, model_params, alpha=1 - decay)
        else:
            for ema_p, model_p in zip(ema_params, model_params):
                ema_p.lerp_(model_p.to(device=self.device), weight=1. - decay)

        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(model_b.to(device=self.device))

    @torch.no_grad()
    def set(self, model):
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            ema_v.copy_(model_v.to(device=self.device))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)