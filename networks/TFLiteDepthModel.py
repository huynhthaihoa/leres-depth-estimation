import cv2
import numpy as np

# Load the LiteRT model and allocate tensors.
from ai_edge_litert.interpreter import Interpreter

from PIL import Image
from typing import Tuple, Union

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class TFLiteDepthModel():

    IMAGENET_MEAN = (123.68, 116.78, 103.94)
    IMAGENET_SCALE = 0.017
    DEFAULT_SCALE = 1 / 255.
        
    def __init__(self, weight_path, normalize=False, use_tta=False, recover_original=True, original_scale=False):
        """
        Initialize the TFLiteDepthModel.

        Args:
            weight_path (str): Path to the TFLite model weights.
            normalize (bool): Whether to normalize input images using ImageNet mean and scale.
            use_tta (bool): Whether to use test-time augmentation.
            recover_original (bool): Whether to recover the original image dimensions in the output.
            original_scale (bool): Whether to use the original scale for input images.
        """
        try:
            self.interpreter = Interpreter(weight_path)
            self.interpreter.allocate_tensors()
        except Exception as e:
            raise Exception(f"Error loading TFLite model: {e}")        

        # Get model details
        self.num_params = 0
        for tensor in self.interpreter.get_tensor_details():
            shape = tensor['shape']
            if len(shape) > 0:  # Ignore scalars and non-trainable tensors
                self.num_params += np.prod(shape)
        
        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # B, C, H, W
        input_shape = input_details[0]['shape']
        self.model_input_height = input_shape[2]
        self.model_input_width = input_shape[3]
        
        self.input_index = input_details[0]['index']
        
        self.output_index = output_details[0]['index']
        
        if normalize:  # ImageNet
            self.mean = self.IMAGENET_MEAN
            self.scale = self.IMAGENET_SCALE
        else:
            self.mean = 0
            self.scale = self.DEFAULT_SCALE
            
        self.use_tta = use_tta
        self.recover_original = recover_original
        self.original_scale = original_scale

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
        self.interpreter.set_tensor(self.input_index, input_tensor)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)
    
    def postprocess(self, output_tensor: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
        """Convert TFLite model output to depth map

        Args:
            output_tensor: raw output from TFLite model as shape (B, C, H, W)
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
        """TFLite inference full pipeline
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
        """TFLite inference full pipeline
        Args:
            image: input image (PIL.Image or cv2/np.ndarray) as shape (H, W, 3)
        Returns:
            output depth map as shape (original_height, orginal_width) if recover_orginal = True, otherwise as shape (model_input_width, model_input_height)
        """   
        return self.pipeline(image, to_grayscale)
    
    def __len__(self):
        return self.num_params