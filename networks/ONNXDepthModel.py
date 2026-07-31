import cv2
import numpy as np
import onnxruntime

from PIL import Image
from typing import Tuple, Union

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class ONNXDepthModel():
    
    IMAGENET_MEAN = (123.68, 116.78, 103.94)
    IMAGENET_SCALE = 0.017
    DEFAULT_SCALE = 1 / 255.    
    
    def __init__(self, weight_path, use_cuda=True, normalize=False, use_tta=False, recover_original=True, original_scale=False):#, width=640, height=480):
        """ONNX depth model inference wrapper:
    
        Args:
            weight_path (str): path to the ONNX weight file
            use_cuda (bool): use CUDA GPU or not (default is False)
            normalize (bool): if True, use ImageNet normalization (default is True)
            use_tta (bool): use test-time augmentation to improve the accuracy but also reduce the FPS (default is False)
            recover_original (bool): if True, the output depth map has the same size with the input image
                                    otherwise, the output depth map size is 1/4 of input image size

        """    
        self.initialize_model(weight_path, use_cuda)
        self.init_mask()
        if normalize: #ImageNet
            self.mean = (123.68, 116.78, 103.94)
            self.scale = 0.017
        else:
            self.mean = 0
            self.scale = 1 / 255.
        self.use_tta = use_tta
        self.recover_original = recover_original
        self.original_scale = original_scale
    
    def initialize_model(self, weight_path, use_cuda):      
        if use_cuda and onnxruntime.get_device() == 'GPU':
            self.session = onnxruntime.InferenceSession(weight_path, providers = [('CUDAExecutionProvider', {'cudnn_conv_algo_search': 'DEFAULT',})])                                                                               
            self.use_cuda = True
        else:
            self.session = onnxruntime.InferenceSession(weight_path, providers=['CPUExecutionProvider'])
            self.use_cuda = False
                                                                                                       
        # Get model info
        self.get_input_details()
        self.get_output_details()
    
    def init_mask(self):
        xs = np.asarray([np.linspace(0., 1., self.model_input_width)])
        xs = np.repeat(xs, self.model_input_height, axis=0)
        xs = xs[None, None]
        self.mask = 1.0 - np.clip(20. * (xs - 0.05), 0., 1.)
        self.mask_hat = np.flip(self.mask, 3)

    def is_use_cuda(self):
        return self.use_cuda
    
    def get_inference_size(self):
        """ONNX model width & height shape
        """        
        return (self.model_input_width, self.model_input_height)
    
    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.model_input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        model_input_shape = model_inputs[0].shape
        self.model_input_height = model_input_shape[2]
        self.model_input_width = model_input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.model_output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def apply_tta(self, input_tensor: np.ndarray, output_tensor: np.ndarray) -> np.ndarray:
        input_tensor_flip = np.flip(input_tensor, axis=3)
        output_tensor_flip = self.predict(input_tensor_flip)
        inv_outputs_hat = np.flip(output_tensor_flip, axis=-1)
        inv_outputs_fused = 0.5 * (output_tensor + inv_outputs_hat)
        return self.mask_hat * output_tensor + self.mask * inv_outputs_hat + \
                (1.0 - self.mask - self.mask_hat) * inv_outputs_fused
        
    def preprocess(self, input_image: Union[Image.Image, np.ndarray], to_grayscale: bool = False) -> Tuple[np.ndarray, int, int]:
        """Convert input image into appropriate input format for ONNX model

        Args:
            input_image: input image (PIL.Image or cv2/np.ndarray)

        Raises:
            TypeError: input_image is neither PIL.Image nor numpy/cv2 ndarray

        Returns:
            input_tensor: input image as ONNX input format (1, 3, H, W)
            original_height: original height of input image 
            orginal_width: original width of input image 
        """        
        need_swap = True
        
        if not (_is_pil_image(input_image) or _is_numpy_image(input_image)):
            raise TypeError(
                'input_image should be PIL Image or ndarray. Got {}'.format(type(input_image)))
            
        if _is_pil_image(input_image):
            input_image = np.asarray(input_image)
            if input_image.ndim == 2: #8-bit image
                input_image = np.repeat(input_image[..., np.newaxis], 3, axis=2)
            need_swap = False
            
        elif input_image.ndim == 2: #8-bit image
            input_image = np.repeat(input_image[..., np.newaxis], 3, axis=2)
            need_swap = True
            
        original_height, orginal_width = input_image.shape[:-1]
        
        if self.original_scale:
            if need_swap:
                input_image = input_image[:, :, ::-1]
            
            if to_grayscale:
                input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
                input_image = np.repeat(input_image[..., np.newaxis], 3, axis=2)

            input_tensor = np.ascontiguousarray(input_image, dtype=np.float32)
            input_tensor = cv2.resize(input_tensor, (self.model_input_width, self.model_input_height), interpolation=cv2.INTER_LINEAR)
            input_tensor = input_tensor.transpose((2, 0, 1))
            input_tensor = input_tensor[None]
        else:
            if to_grayscale:
                if need_swap:
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
                else:
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
                input_image = np.repeat(input_image[..., np.newaxis], 3, axis=2)
            
            input_tensor = cv2.dnn.blobFromImage(input_image, self.scale, (self.model_input_width, self.model_input_height), self.mean, need_swap, False)        
        
        return input_tensor, original_height, orginal_width
    
    def predict(self, input_tensor: np.ndarray) -> np.ndarray:
        """ONNX model prediction

        Args:
            input_tensor: input as appropriate format of ONNX model

        Returns:
            raw output from ONNX model
        """        
        return self.session.run(self.model_output_names, {self.model_input_names[0]: input_tensor})[0]

    def postprocess(self, output_tensor: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
        """Convert ONNX model output to depth map

        Args:
            output_tensor: raw output from ONNX model
            orginal_width: original width of input image
            original_height: original height of input image

        Returns:
            output_depth: output depth map as shape (original_height, orginal_width) if recover_orginal = True, otherwise as shape (model_input_width, model_input_height)
        """        
        output_depth = output_tensor.squeeze()
        if self.recover_original and (original_height != output_tensor.shape[0] or original_width != output_tensor.shape[1]):
            output_depth = cv2.resize(output_depth, (original_width, original_height), cv2.INTER_NEAREST_EXACT)
        return output_depth
   
    def pipeline(self, image: Union[Image.Image, np.ndarray], to_grayscale: bool = False) -> np.ndarray:
        """ONNX inference full pipeline
        Args:
            image: input image (PIL.Image or cv2/np.ndarray) as shape (H, W, 3)
        Returns:
            output depth map as shape (original_height, orginal_width) if recover_orginal = True, otherwise as shape (model_input_width, model_input_height)
        """ 
        input_tensor, original_height, original_width = self.preprocess(image, to_grayscale)
        output_tensor = self.predict(input_tensor)
        if self.use_tta:
            output_tensor = self.apply_tta(input_tensor, output_tensor)
        return self.postprocess(output_tensor, original_width, original_height)
    
    def __call__(self, image: Union[Image.Image, np.ndarray], to_grayscale: bool = False) -> np.ndarray:
        """ONNX inference full pipeline
        Args:
            image: input image (PIL.Image or cv2/np.ndarray) as shape (H, W, 3)
        Returns:
            output depth map as shape (original_height, orginal_width) if recover_orginal = True, otherwise as shape (model_input_width, model_input_height)
        """   
        return self.pipeline(image, to_grayscale)