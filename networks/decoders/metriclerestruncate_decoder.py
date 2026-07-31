import torch.nn as nn

from ..class_utils import Interpolate, FTB, FFM

class MetricAO(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels, outchannels, interpolate=False): #, scale_factor=2
        super(MetricAO, self).__init__()
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
                nn.Sigmoid(), \
                Interpolate(scale_factor=4, mode='bilinear', align_corners=False)
                )
        else:
            self.adapt_conv = nn.Sequential(
                nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels // 2, kernel_size=3, padding=1,
                        stride=1, bias=True), \
                nn.BatchNorm2d(num_features=self.inchannels // 2), \
                nn.ReLU(inplace=True), \
                nn.Conv2d(in_channels=self.inchannels // 2, out_channels=self.outchannels, kernel_size=3, padding=1,
                        stride=1, bias=True), \
                nn.Sigmoid(), \
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
                )

    def forward(self, x):
        x = self.adapt_conv(x)
        return x
                    
class MetricLeResTruncateDecoder(nn.Module):
    def __init__(self, in_channels, mid_channel=0, interpolate=False):
        """Metric Decoder, which is customized from LeReS decoder for
        a metric depth model
        """        
        super(MetricLeResTruncateDecoder, self).__init__()
        self.inchannels =  in_channels
        
        if mid_channel == 0:
            if len(in_channels) == 3:
                mid_channel = in_channels[0] // 2
            else:
                mid_channel = in_channels[-4]

        self.midchannels = [mid_channel, mid_channel, mid_channel, mid_channel * 2]
        
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
        
        self.outconv = MetricAO(inchannels=self.midchannels[-3], outchannels=self.outchannels, interpolate=interpolate)
        
        # self._init_params()
        
    # def _init_params(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.normal_(m.weight, std=0.01)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.ConvTranspose2d):
    #             init.normal_(m.weight, std=0.01)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d): #NN.BatchNorm2d
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init.normal_(m.weight, std=0.01)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
                    
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
        
        return self.outconv(x_4)

        
    