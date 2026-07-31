"""This implementation is inspired from: https://github.com/isl-org/MiDaS/blob/master/midas/dpt_depth.py
"""

import torch.nn as nn

from ..class_utils import _make_fusion_deconv_block, _make_fusion_block, _make_scratch

class DPTDeconvDecoder(nn.Module):
    def __init__(self, in_channels, mid_channel=0, use_fusiondeconv=False, interpolate=False, use_bn=False, upsample_last=False):
        """DPT Decoder, inspired from MiDaS

        Args:
            in_channels: input channels
            interpolate (bool, optional): _description_. Defaults to False.
        """
        super(DPTDeconvDecoder, self).__init__()
        
        self.inchannels = in_channels

        if mid_channel == 0:
            self.midchannel = self.inchannels[0] // 2
        else:
            self.midchannel = mid_channel
        
        if len(self.inchannels) == 4:
            self.scratch_layer_1 = _make_scratch(self.inchannels[0], self.midchannel)
            if use_fusiondeconv:
                self.fusion_block_1 = _make_fusion_deconv_block(self.midchannel, use_bn=use_bn)
            else:
                self.fusion_block_1 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)  
            self.upsample_factor = 2 
        else:
            self.scratch_layer_1 = None
            self.fusion_block_1 = None
            self.upsample_factor = 4             

        self.scratch_layer_2 = _make_scratch(self.inchannels[-3], self.midchannel)
        self.scratch_layer_3 = _make_scratch(self.inchannels[-2], self.midchannel)
        self.scratch_layer_4 = _make_scratch(self.inchannels[-1], self.midchannel)
        
        if use_fusiondeconv:
            self.fusion_block_2 = _make_fusion_deconv_block(self.midchannel, use_bn=use_bn)              
            self.fusion_block_3 = _make_fusion_deconv_block(self.midchannel, use_bn=use_bn)                
            self.fusion_block_4 = _make_fusion_deconv_block(self.midchannel, use_bn=use_bn)               
        else:            
            self.fusion_block_2 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)                
            self.fusion_block_3 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)                
            self.fusion_block_4 = _make_fusion_block(self.midchannel, use_bn=use_bn, interpolate=interpolate)                
            
        self.output_conv1 = nn.Conv2d(self.midchannel, self.midchannel // 2, kernel_size=3, stride=1, padding=1)
        
        self.upsample_last = upsample_last
        
        if self.upsample_last:
            self.output_conv2 = nn.Sequential(
                nn.Conv2d(self.midchannel // 2, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                # nn.Sigmoid()
            )    
            self.upsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=1), \
                    nn.ReLU(inplace=True), \
                    nn.ConvTranspose2d(1, 1, kernel_size=self.upsample_factor, stride=self.upsample_factor), \
                    # nn.Conv2d(in_channels=1, out_channels=self.upsample_factor * self.upsample_factor, kernel_size=3, padding=1, stride=1, bias=False), \
                    # nn.PixelShuffle(self.upsample_factor), \
                    nn.ReLU(inplace=True), \
                    nn.Conv2d(1, out_channels=1, kernel_size=3, padding=1, stride=1, bias=False), \
                    nn.Sigmoid()
                )      
        else:
            self.upsample = nn.Sequential(
                    nn.BatchNorm2d(num_features=self.midchannel // 2), \
                    nn.ReLU(inplace=True), \
                    nn.ConvTranspose2d(self.midchannel // 2, 1, kernel_size=self.upsample_factor, stride=self.upsample_factor), \
                    # nn.Conv2d(in_channels=self.midchannel // 2, out_channels=self.upsample_factor * self.upsample_factor, kernel_size=3, padding=1, stride=1, bias=False), \
                    # nn.PixelShuffle(self.upsample_factor), \
                    nn.ReLU(inplace=True), \
                    nn.Conv2d(1, out_channels=1, kernel_size=3, padding=1, stride=1, bias=False), \
                )      
            self.output_conv2 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
            )                 
        # self.sigmoid_last = sigmoid_last        
            
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
        
        if self.upsample_last:
            out = self.output_conv2(out)
            return self.upsample(out)
        
        out = self.upsample(out)
        return self.output_conv2(out)
