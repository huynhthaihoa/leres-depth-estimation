import torch
import torch.nn as nn

from ..class_utils import FTB, FFMDeConv

class PixelFormerDispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(PixelFormerDispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)        
        self.upsample = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)


    def forward(self, x, centers):
        x = self.conv1(x)
        
        x = self.upsample(x)

        x = x.softmax(dim=1)
        
        x = torch.sum(x * centers, dim=1, keepdim=True)
        
        return x

class BCP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, max_depth, in_features=512, hidden_features=512*4, out_features=256, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.max_depth = max_depth

    def forward(self, x):
        x = torch.mean(x.flatten(start_dim=2), dim = 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        bins = torch.softmax(x, dim=1)
        bins = bins / bins.sum(dim=1, keepdim=True)
        bin_widths = self.max_depth * bins
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=0)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.contiguous().view(n, dout, 1, 1)
        return centers
                
class PixelFormerDeconvDecoder(nn.Module):
    def __init__(self, in_channels, max_depth, mid_channel=0):
        """Metric Decoder, which is customized from LeReS decoder for
        a metric depth model
        """        
        super(PixelFormerDeconvDecoder, self).__init__()
        self.inchannels =  in_channels
        if mid_channel == 0:
            mid_channel = in_channels[-4]

        self.midchannels = [mid_channel, mid_channel, mid_channel, mid_channel * 2]
        self.upfactors = [2, 2, 2, 2]
        self.outchannels = 1

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels[0], out_channels=in_channels[0], kernel_size=2, stride=2)

        self.ffm2 = FFMDeConv(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2])#, interpolate=interpolate) #, upfactor=self.upfactors[2]
        self.ffm1 = FFMDeConv(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1])#, interpolate=interpolate) #, upfactor=self.upfactors[1]
        self.ffm0 = FFMDeConv(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0])#, interpolate=interpolate) #, upfactor=self.upfactors[0]
                
        self.disp_head1 = PixelFormerDispHead(input_dim=self.midchannels[0])

        self.bcp = BCP(max_depth=max_depth, in_features=self.inchannels[0])
               
    def forward(self, features):
        x_32x = self.conv(features[3])  # 1/32
        x_32 = self.conv1(x_32x)
        x_16 = self.upsample(x_32)  # 1/16
        x_8 = self.ffm2(features[2], x_16)  # 1/8
        x_4 = self.ffm1(features[1], x_8)  # 1/4
        x_2 = self.ffm0(features[0], x_4)  # 1/2
        
        bin_centers = self.bcp(x_32)
        x = self.disp_head1(x_2, bin_centers)
        return x
    