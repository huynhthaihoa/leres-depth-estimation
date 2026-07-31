"""This implementation is inspired from: https://github.com/isl-org/MiDaS/blob/master/midas/dpt_depth.py
"""

import torch.nn as nn

from ..class_utils import _make_fusion_block, _make_scratch, upsample


class DPTDecoder(nn.Module):
    def __init__(self, in_channels, mid_channel=0, interpolate=False, use_bn=False, sigmoid_last=False):
        """DPT Decoder, inspired from MiDaS

        Args:
            in_channels: input channels
            interpolate (bool, optional): _description_. Defaults to False.
        """
        super(DPTDecoder, self).__init__()
        
        self.inchannels = in_channels

        if mid_channel == 0:
            self.midchannel = self.inchannels[0] // 2
        else:
            self.midchannel = mid_channel
        
        if len(self.inchannels) == 4:
            self.scratch_layer_1 = _make_scratch(self.inchannels[0], self.midchannel)
            self.fusion_block_1 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)  
            self.upsample_factor = 2 
        else:
            self.scratch_layer_1 = None
            self.fusion_block_1 = None
            self.upsample_factor = 4             

        self.scratch_layer_2 = _make_scratch(self.inchannels[-3], self.midchannel)
        self.scratch_layer_3 = _make_scratch(self.inchannels[-2], self.midchannel)
        self.scratch_layer_4 = _make_scratch(self.inchannels[-1], self.midchannel)
        
        self.fusion_block_2 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)                
        self.fusion_block_3 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)                
        self.fusion_block_4 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)                
            
        self.output_conv1 = nn.Conv2d(self.midchannel, self.midchannel // 2, kernel_size=3, stride=1, padding=1)
        self.output_conv2 = nn.Sequential(
            nn.Conv2d(self.midchannel // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )    
        
        self.sigmoid_last = sigmoid_last        
            
    def forward(self, features):
        feature_2_rn = self.scratch_layer_2(features[-3])
        feature_3_rn = self.scratch_layer_3(features[-2]) 
        feature_4_rn = self.scratch_layer_4(features[-1])  
        
        path_4 = self.fusion_block_4(feature_4_rn)#, size=layer_3_rn.shape[2:])
        path_3 = self.fusion_block_3(path_4, feature_3_rn)#, size=layer_3_rn.shape[2:])
        path_2 = self.fusion_block_2(path_3, feature_2_rn)#, size=layer_3_rn.shape[2:])
        
        if self.scratch_layer_1 is not None:
            feature_1_rn = self.scratch_layer_1(features[0])
            path_1 = self.fusion_block_1(path_2, feature_1_rn)#, size=layer_3_rn.shape[2:]) 
            out = self.output_conv1(path_1)
        else:
            out = self.output_conv1(path_2)
                
        if self.sigmoid_last:
            out = upsample(out, self.upsample_factor)
            return self.output_conv2(out)
        
        out = self.output_conv2(out)
        return upsample(out, self.upsample_factor)