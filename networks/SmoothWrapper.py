from collections import deque
import numpy as np

class SmoothWrapper():
    """The class that aims to "smooth" the predicted depth map during real-time inference
    As of now, 3 smooth types are supported:
      - 0: cumulative average
      - 1: weighted moving average
      - 2: exponential moving average
    """  
    def __init__(self, smooth_type=0, smooth_factor=0.7, buffer_capacity=15) -> None:
        """Init function

        Args:
            smooth_type (int, optional): smooth type (0: cumulative average, 1: weighted moving average, 2: exponential moving average). Defaults to 0.
            smooth_factor (float, optional): smooth factor to be used if smooth type is 2: exponential moving average. Defaults to 0.7.
            buffer_capacity (int, optional): smooth buffer capacity. Defaults to 15.
        """        
        assert smooth_type in [0, 1, 2]
        self.smooth_type = smooth_type
        self.smooth_factor = smooth_factor
        self.smooth_buffer = deque(maxlen=buffer_capacity)
        self.smooth_cache = None
    
    def pipeline(self, depth):
        self.smooth_buffer.append(depth)
        if self.smooth_type == 0: # CUMULATIVE AVERAGE
            self.cumulative_average()
        elif self.smooth_type == 1: #WEIGHTED MOVING AVERAGE
            self.weighted_moving_average()
        else:
            self.exponential_moving_average()
        return self.smooth_cache
    
    def __call__(self,depth):
        return self.pipeline(depth)
    
    def cumulative_average(self):
        self.smooth_cache = sum(self.smooth_buffer) / len(self.smooth_buffer)
    
    def weighted_moving_average(self):
        n = len(self.smooth_buffer)
        tensors = np.arange(1, n + 1)
        self.smooth_cache = 2 * sum(i[0] * i[1] for i in zip(tensors, self.smooth_buffer)) / (n * (n + 1))#np.dot(tensors, self.smooth_buffer)
    
    def exponential_moving_average(self):
        if self.smooth_cache is None:
            self.smooth_cache = self.smooth_buffer[-1] 
        else:
            self.smooth_cache = self.smooth_factor * self.smooth_buffer[-1] + (1 - self.smooth_factor) * self.smooth_cache