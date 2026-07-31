import torch
import torch.nn as nn

from ..class_utils import Interpolate

class agg_node(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(agg_node, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.relu(x)
        
class refine(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(refine, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.conv(x))
    
class predict(nn.Module):
    def __init__(self, in_planes, out_planes, type):
        super(predict, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        if type == 0:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(out_planes, out_planes, 2, 2),
                nn.ConvTranspose2d(out_planes, out_planes, 2, 2),
                nn.Sigmoid()
            ) 
        elif type == 1:
            self.block = nn.Sequential(
                nn.Sigmoid(),
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)      
            )             
        elif type == 2:
            self.block = nn.Sequential(
                nn.Sigmoid(),
                Interpolate(scale_factor=4, mode='bilinear', align_corners=False)
            )  
                         
    def forward(self, x):
        return self.block(self.conv(x))
    
class upshuffle(nn.Module):
    def __init__(self, in_planes, out_planes, upscale_factor):
        super(upshuffle, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes*upscale_factor**2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.block(x)

class FPNDeConvDecoder(nn.Module):   
    def __init__(self, in_channels, mid_channel=0, type=0):
        """FPNDeConv Decoder, which is based on Feature Pyramid Network architecture:
        Args:
            type: last layer type:
                - 0 (default): last layer is sigmoid layer
                - 1: last layer is Upsample layer
                - 2: last layer is Interpolate layer
        """        
        super(FPNDeConvDecoder, self).__init__()
        self.inchannels =  in_channels

        if mid_channel == 0:
            mid_channel = self.inchannels[0] // 2
        #     self.midchannel = self.inchannels[0] // 2
        # else:
        self.midchannel = mid_channel
                    
        #top layer
        self.toplayer = nn.Conv2d(self.inchannels[-1], self.inchannels[0], kernel_size=1, stride=1, padding=0)
        
        # Lateral layers
        self.latlayer1 = nn.Conv2d(self.inchannels[-2], self.inchannels[0], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(self.inchannels[-3], self.inchannels[0], kernel_size=1, stride=1, padding=0)
        if len(self.inchannels) == 4:
            self.latlayer3 = nn.Conv2d(self.inchannels[0], self.inchannels[0], kernel_size=1, stride=1, padding=0)
            self.smooth3 = nn.Conv2d(self.inchannels[0], self.inchannels[0], kernel_size=3, stride=1, padding=1)
            self.agg4 = agg_node(self.inchannels[0], self.midchannel)
            self.agg3 = agg_node(self.inchannels[0], self.midchannel)
            self.up3 = upshuffle(self.midchannel, self.midchannel, 2)
        else:
            self.latlayer3 = None
            self.smooth3 = None
            self.agg4 = None
            self.agg3 = agg_node(self.inchannels[0], self.inchannels[0])
            self.up3 = upshuffle(self.inchannels[0], self.midchannel * 2, 2)

        # Refine layers
        self.smooth1 = nn.Conv2d(self.inchannels[0], self.inchannels[0], kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(self.inchannels[0], self.inchannels[0], kernel_size=3, stride=1, padding=1)

        # Aggregate layers
        self.agg1 = agg_node(self.inchannels[0], self.midchannel)
        self.agg2 = agg_node(self.inchannels[0], self.midchannel)
        
        # Upshuffle layers
        self.up1 = upshuffle(self.midchannel, self.midchannel, 8)
        self.up2 = upshuffle(self.midchannel, self.midchannel, 4)
        
        # Depth prediction
        self.refine = refine(self.midchannel * 4, self.midchannel)
        self.predict = predict(self.midchannel, 1, type)
        
        # Upsample layer
        self.upsample = nn.ConvTranspose2d(self.inchannels[0], self.inchannels[0], 2, 2)
        
        # self._init_params()
        
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        return self.upsample(x) + y
    
    def forward(self, features):
        
        #top-down
        p5 = self.toplayer(features[3])
        d5 = self.up1(self.agg1(p5))

        # p4 = self._upsample_add(p5, self.latlayer1(features[2]))
        p4 = self.upsample(p5) + self.latlayer1(features[2])
        p4 = self.smooth1(p4)
        d4 = self.up2(self.agg2(p4))
        
        # p3 = self._upsample_add(p4, self.latlayer2(features[1]))
        p3 = self.upsample(p4) + self.latlayer2(features[1])
        p3 = self.smooth2(p3)
        d3 = self.up3(self.agg3(p3))
        
        if self.latlayer3 is not None:
            #p2 = self._upsample_add(p3, self.latlayer3(features[0]))
            p2 = self.upsample(p3) + self.latlayer3(features[0])
            p2 = self.smooth3(p2)   
            d2 = self.agg4(p2)
            vol = torch.cat( [d for d in [d5, d4, d3, d2]], dim=1)
        else:
            vol = torch.cat( [d for d in [d5, d4, d3]], dim=1)
            
        return self.predict(self.refine(vol))