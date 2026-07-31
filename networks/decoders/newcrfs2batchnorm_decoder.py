import torch.nn as nn

from ..class_utils import FTB, FFM, Interpolate
from ..function_utils import upsample

class NeWCRFs2BatchNormDispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(NeWCRFs2BatchNormDispHead, self).__init__()
        self.norm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        x = self.norm1(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return self.sigmoid(x)
            

class NeWCRFs2BatchNormDecoder(nn.Module):
    def __init__(self, in_channels, interpolate=False):
        """Metric Decoder, which is customized from LeReS decoder for
        a metric depth model
        """        

        super(NeWCRFs2BatchNormDecoder, self).__init__()
        self.inchannels =  in_channels
        self.midchannels = [in_channels[0], in_channels[0], in_channels[0], in_channels[0] * 2]
        self.upfactors = [2, 2, 2, 2]
        self.outchannels = 1

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        
        if interpolate:
            self.upsample = Interpolate(scale_factor=self.upfactors[3], mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample(scale_factor=self.upfactors[3], mode='bilinear', align_corners=True)

        self.ffm2 = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2], interpolate=interpolate) # , upfactor=self.upfactors[2]
        self.ffm1 = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1], interpolate=interpolate) # , upfactor=self.upfactors[1]
        self.ffm0 = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], interpolate=interpolate) # , upfactor=self.upfactors[0]
        
        self.disp = NeWCRFs2BatchNormDispHead(input_dim=self.midchannels[0])
                    
    def forward(self, features):
        x_32x = self.conv(features[3])  # 1/32
        x_32 = self.conv1(x_32x)
        x_16 = self.upsample(x_32)  # 1/16
        x_8 = self.ffm2(features[2], x_16)  # 1/8
        x_4 = self.ffm1(features[1], x_8)  # 1/4
        x_2 = self.ffm0(features[0], x_4)  # 1/2
        #-----------------------------------------
        x = self.disp(x_2, 2)  # original size
        return x
        
    