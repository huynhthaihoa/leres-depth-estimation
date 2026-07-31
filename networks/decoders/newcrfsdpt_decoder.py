"""This implementation is inspired from: https://github.com/isl-org/MiDaS/blob/master/midas/dpt_depth.py
"""

import torch.nn as nn

from ..class_utils import _make_fusion_block, _make_scratch

class NeWCRFsDPTDecoder(nn.Module):
    def __init__(self, in_channels, mid_channel=0, interpolate=False, use_bn=False, sigmoid_last=False):
        """DPT Decoder, inspired from MiDaS

        Args:
            in_channels: input channels
            interpolate (bool, optional): _description_. Defaults to False.
        """
        super(NeWCRFsDPTDecoder, self).__init__()
        
        self.inchannels = in_channels

        if mid_channel == 0:
            self.midchannel = self.inchannels[0] // 2
        else:
            self.midchannel = mid_channel
                
        self.scratch_layer_1 = _make_scratch(self.inchannels[-3], self.midchannel)
        self.scratch_layer_2 = _make_scratch(self.inchannels[-2], self.midchannel)
        self.scratch_layer_3 = _make_scratch(self.inchannels[-1], self.midchannel)
        
        self.fusion_block_1 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)                
        self.fusion_block_2 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)                
        self.fusion_block_3 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)                

        if len(self.inchannels) == 4:
            self.scratch_layer_0 = _make_scratch(self.inchannels[0], self.midchannel)
            self.fusion_block_0 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)  
            self.upsample_factor = 2               
        else:
            self.scratch_layer_0 = None 
            self.fusion_block_0 = None 
            self.upsample_factor = 4  
                    
        if sigmoid_last:
            from ..class_utils import NeWCRFs2DispHead
            self.disp = NeWCRFs2DispHead(self.midchannel)
        else:
            from ..class_utils import NeWCRFsDispHead
            self.disp = NeWCRFsDispHead(self.midchannel)
            
    def forward(self, features):
        
        feature_1_rn = self.scratch_layer_1(features[-3])
        feature_2_rn = self.scratch_layer_2(features[-2])
        feature_3_rn = self.scratch_layer_3(features[-1])   
        
        path_3 = self.fusion_block_3(feature_3_rn)
        path_2 = self.fusion_block_2(path_3, feature_2_rn)
        path_1 = self.fusion_block_1(path_2, feature_1_rn) 
        
        if self.scratch_layer_0 is not None:
            feature_0_rn = self.scratch_layer_0(features[0])
            path_0 = self.fusion_block_0(path_1, feature_0_rn)
            return self.disp(path_0, self.upsample_factor)
        return self.disp(path_1, self.upsample_factor)