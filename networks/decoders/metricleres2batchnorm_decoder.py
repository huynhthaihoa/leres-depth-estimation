import torch.nn as nn

from ..class_utils import Interpolate, FTB, FFM

class MetricAO2BatchNorm(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels, outchannels, interpolate=False): #, scale_factor=2
        super(MetricAO2BatchNorm, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels

        if interpolate:
            self.adapt_conv = nn.Sequential(
                nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels // 2, kernel_size=3, padding=1,
                        stride=1, bias=True), \
                nn.BatchNorm2d(num_features=self.inchannels // 2), \
                nn.ReLU(inplace=True), \
                nn.Conv2d(in_channels=self.inchannels // 2, out_channels=self.outchannels, kernel_size=3, padding=1,
                        stride=1, bias=True), \
                nn.BatchNorm2d(self.outchannels),                
                Interpolate(scale_factor=2, mode='bilinear', align_corners=False), \
                nn.Sigmoid(),
                )
        else:
            self.adapt_conv = nn.Sequential(
                nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels // 2, kernel_size=3, padding=1,
                        stride=1, bias=True), \
                nn.BatchNorm2d(num_features=self.inchannels // 2), \
                nn.ReLU(inplace=True), \
                nn.Conv2d(in_channels=self.inchannels // 2, out_channels=self.outchannels, kernel_size=3, padding=1,
                        stride=1, bias=True), \
                nn.BatchNorm2d(self.outchannels),               
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), \
                nn.Sigmoid(), \
                )

    def forward(self, x):
        x = self.adapt_conv(x)
        return x
                    
class MetricLeRes2BatchNormDecoder(nn.Module):
    def __init__(self, in_channels, interpolate=False):
        """Metric Decoder, which is customized from LeReS decoder for
        a metric depth model
        """        
        super(MetricLeRes2BatchNormDecoder, self).__init__()
        self.inchannels =  in_channels
        self.midchannels = [in_channels[0], in_channels[0], in_channels[0], in_channels[0] * 2]
        #self.upfactors = [2, 2, 2, 2]
        self.outchannels = 1

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3]) #self.upfactors[3]
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True) #self.upfactors[3]
        
        if interpolate:
            self.upsample = Interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.ffm2 = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2], interpolate=interpolate) #, upfactor=self.upfactors[2]
        self.ffm1 = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1], interpolate=interpolate) #, upfactor=self.upfactors[1]
        self.ffm0 = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], interpolate=interpolate) #, upfactor=self.upfactors[0]
        
        self.outconv = MetricAO2BatchNorm(inchannels=self.midchannels[0], outchannels=self.outchannels, interpolate=interpolate)
                    
    def forward(self, features):
        x_32x = self.conv(features[3])  # 1/32
        x_32 = self.conv1(x_32x)
        x_16 = self.upsample(x_32)  # 1/16
        x_8 = self.ffm2(features[2], x_16)  # 1/8
        x_4 = self.ffm1(features[1], x_8)  # 1/4
        x_2 = self.ffm0(features[0], x_4)  # 1/2
        #-----------------------------------------
        x = self.outconv(x_2)  # original size
        return x

        
    