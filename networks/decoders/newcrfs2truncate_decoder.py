import torch.nn as nn

from ..class_utils import Interpolate, FTB, FFM, NeWCRFs2DispHead
from ..function_utils import upsample

# class NeWCRFs2DispHead(nn.Module):
#     def __init__(self, input_dim=100):
#         super(NeWCRFs2DispHead, self).__init__()
#         self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, scale):
#         x = self.conv1(x)
#         if scale > 1:
#             x = upsample(x, scale_factor=scale)
#         return self.sigmoid(x)
            
class NeWCRFs2TruncateDecoder(nn.Module):
    def __init__(self, in_channels, mid_channel=0, interpolate=False):
        """Metric Decoder, which is customized from LeReS decoder for
        a metric depth model
        """        
        super(NeWCRFs2TruncateDecoder, self).__init__()
        self.inchannels =  in_channels
        if mid_channel == 0:
            if len(in_channels) == 3:
                mid_channel = in_channels[0] // 2
            else:
                mid_channel = in_channels[-4]

        self.midchannels = [mid_channel, mid_channel, mid_channel, mid_channel * 2]
        # self.upfactors = [2, 2, 2, 2]
        self.outchannels = 1

        self.conv = FTB(inchannels=self.inchannels[-1], midchannels=self.midchannels[-1]) #self.upfactors[3]            
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[-1], out_channels=self.midchannels[-2], kernel_size=3, padding=1, stride=1, bias=True) #self.upfactors[3]
        
        if interpolate:
            self.upsample = Interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.ffm2 = FFM(inchannels=self.inchannels[-2], midchannels=self.midchannels[-2], outchannels = self.midchannels[-2], interpolate=interpolate) #, upfactor=self.upfactors[2]
        self.ffm1 = FFM(inchannels=self.inchannels[-3], midchannels=self.midchannels[-3], outchannels = self.midchannels[-3], interpolate=interpolate) #, upfactor=self.upfactors[1]
        # self.ffm0 = FFM(inchannels=self.inchannels[-4], midchannels=self.midchannels[-4], outchannels = self.midchannels[-4], interpolate=interpolate) #, upfactor=self.upfactors[0]

        # if len(self.inchannels) == 5:
        #     self.ftb = FTB(inchannels=self.inchannels[0], midchannels=self.midchannels[0])
        
        self.disp = NeWCRFs2DispHead(input_dim=self.midchannels[-3])
        
    def forward(self, features):
        x_32x = self.conv(features[-1])  # 1/32
        x_32 = self.conv1(x_32x)
        x_16 = self.upsample(x_32)  # 1/16
        x_8 = self.ffm2(features[-2], x_16)  # 1/8
        x_4 = self.ffm1(features[-3], x_8)  # 1/4
        # x_2 = self.ffm0(features[-4], x_4)  # 1/2
        # #-----------------------------------------
        # if len(self.inchannels) == 5:
        #     x_2 = self.ftb(features[0]) + x_2

        return self.disp(x_4, 4)  # original size
        
    