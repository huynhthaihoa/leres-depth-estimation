'''
Reference: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
'''
from loguru import logger
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):

        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0
        self.early_stop = False
        self.min_val_loss = np.inf

    def __call__(self, val_loss):
        if val_loss < self.min_val_loss:
            if self.verbose:
                logger.info(f'Validation loss improved from {self.min_val_loss} to {val_loss}')
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > self.min_val_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.warning(f'Early Stopping triggered!')
            else:
                if self.verbose:
                    logger.warning(f'Validation loss did not improve, counter: {self.counter} out of {self.patience}')
        return self.early_stop
                            
    def percentage(self):
        return (self.counter / self.patience) * 100
                
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count