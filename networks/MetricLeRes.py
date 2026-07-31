import torch.nn as nn
import torch.nn.init as init

class MetricLeRes(nn.Module):
    def __init__(self, encoder_type: str, decoder_type='metricleres', max_depth=10.0, frozen_stages=False, pretrained=True, interpolate=False, replace_silu=False, feature_list=[128,256, 512, 1024], init_type='normal', use_customsilu=False, keepnum_maxpool=[False, False, False], use_5_feat=False, mid_channel=0, input_width=640, input_height=480, **kwargs):
        super().__init__()

        self.pretrained = pretrained
        
        encoder_type = encoder_type.lower()

        if encoder_type.startswith('timm_'): #timm encoders
            from .encoders.timm_encoder import timmEncoder
            # remove "timm_" part
            self.backbone = timmEncoder(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, use_5_feat=use_5_feat, replace_gelu=replace_silu, input_width=input_width, input_height=input_height)
        elif encoder_type.startswith('dinov2'):
            from .encoders.dinov2_encoder import Dinov2
            self.backbone = Dinov2(encoder_type, finetune=(not frozen_stages))
        
        # elif encoder_type.find('fastvit') != -1:
        #     self.backbone = create_model(encoder_type, fork_feat=True)
        #     if self.pretrained:
        #         checkpoint_path = f"{encoder_type}.pth.tar"
        #         if not os.path.exists(checkpoint_path):
        #             os.system(f'wget https://docs-assets.developer.apple.com/ml-research/models/fastvit/image_classification_distilled_models/{checkpoint_path}')
        #         checkpoint = torch.load(checkpoint_path)
        #         self.backbone.load_state_dict(checkpoint['state_dict'], strict=False)
        #     feature_list = [] # update later         
        # elif encoder_type.find('fastervit') != -1:
        #     self.backbone = create_model(encoder_type, pretrained=pretrained)  
        #     feature_list = []   
        # elif encoder_type.find('mobileone') != -1:
        #     from .encoders.mobileone import MobileOne
        #     self.backbone = MobileOne(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, use_5_feat=use_5_feat)
        elif encoder_type.startswith('ultralytics_'):
            from .encoders.ultralytics_encoder import UltralyticsEncoder
            self.backbone = UltralyticsEncoder(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu)
        elif encoder_type.find('convnextv2') != -1:
            from .encoders.convnextv2_encoder import ConvNeXtV2
            self.backbone = ConvNeXtV2(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, replace_gelu=replace_silu)       
        elif encoder_type.find('convnext') != -1:
            from .encoders.convnext_encoder import ConvNeXt
            self.backbone = ConvNeXt(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, replace_gelu=replace_silu)       
        elif encoder_type.find('resnet') != -1:
            from .encoders.torchvision_encoder import ResNet
            self.backbone = ResNet(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, use_5_feat=use_5_feat)  
        elif encoder_type.find('regnet') != -1:
            from .encoders.torchvision_encoder import RegNet
            self.backbone = RegNet(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, use_5_feat=use_5_feat)        
        elif encoder_type.find('resnext') != -1:
            from .encoders.torchvision_encoder import ResNext
            self.backbone = ResNext(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, use_5_feat=use_5_feat)
        elif encoder_type == 'mobilenetv2':
            from .encoders.torchvision_encoder import MobileNetV2
            self.backbone = MobileNetV2(pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, use_5_feat=use_5_feat)
        elif encoder_type.find('mobilenetv3') != -1:
            from .encoders.torchvision_encoder import MobileNetV3
            self.backbone = MobileNetV3(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list)
        elif encoder_type.find('torchefficientnet') != -1:
            from .encoders.torchvision_encoder import TorchEfficientNet
            self.backbone = TorchEfficientNet(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu) 
        elif encoder_type.find('efficientnetap') != -1:
            from .encoders.geffnet_encoder import EfficientNetAdvProp
            self.backbone = EfficientNetAdvProp(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu, use_5_feat=use_5_feat) 
        elif encoder_type.find('efficientnetlite') != -1:
            from .encoders.geffnet_encoder import EfficientNetLite
            self.backbone = EfficientNetLite(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, use_5_feat=use_5_feat)#, replace_silu=replace_silu, use_customsilu=use_customsilu) 
        elif encoder_type.find('efficientnet') != -1:
            from .encoders.geffnet_encoder import EfficientNet
            self.backbone = EfficientNet(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu, use_5_feat=use_5_feat) 
            # feature_list = self.backbone.dimList  
        elif encoder_type.find('yolov5') != -1:
            # is_lite = (encoder_type.find('lite') != -1)           
            from .encoders.nets.yolov5_encoder import YOLOv5                               
            self.backbone = YOLOv5(encoder_type, pretrained=self.pretrained, out_dimList=feature_list, finetune=(not frozen_stages), keepnum_maxpool=keepnum_maxpool, use_customsilu=use_customsilu, replace_silu=replace_silu, use_5_feat=use_5_feat)
        elif encoder_type.find('yolov6') != -1:
            if encoder_type.find('yolov6lite') != -1:
                from .encoders.yolov6lite_encoder import YOLOv6Lite
                self.backbone = YOLOv6Lite(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, use_5_feat=use_5_feat)
            else:
                from .encoders.yolov6_encoder import YOLOv6
                self.backbone = YOLOv6(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu, use_5_feat=use_5_feat)
        elif encoder_type.find('yolov7') != -1:#.startswith('yolov7'):
            from .encoders.yolov7_encoder import YOLOv7
            self.backbone = YOLOv7(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu)
        elif encoder_type.find('yolov8') != -1:
            from .encoders.nets.yolov8_encoder import YOLOv8                    
            self.backbone = YOLOv8(encoder_type, pretrained=self.pretrained, out_dimList=feature_list, finetune=(not frozen_stages), keepnum_maxpool=keepnum_maxpool, use_customsilu=use_customsilu, replace_silu=replace_silu, use_5_feat=use_5_feat)
        elif encoder_type.find('yolov9') != -1 or encoder_type.find('gelan') != -1:#.startswith('yolov9') or encoder_type.startswith('gelan'):
            from .encoders.yolov9_encoder import YOLOv9
            self.backbone = YOLOv9(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu)
        elif encoder_type.find('yolov10') != -1:
            from .encoders.yolov10_encoder import YOLOv10                    
            self.backbone = YOLOv10(encoder_type, pretrained=self.pretrained, out_dimList=feature_list, finetune=(not frozen_stages), use_customsilu=use_customsilu, replace_silu=replace_silu)#, keepnum_maxpool=keepnum_maxpool, use_5_feat=use_5_feat)
        elif encoder_type.find('yolov11') != -1:
            from .encoders.yolov11_encoder import YOLOv11
            self.backbone = YOLOv11(encoder_type, pretrained=self.pretrained, out_dimList=feature_list, finetune=(not frozen_stages), keepnum_maxpool=keepnum_maxpool, use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_type.find('yolov12') != -1:
            from .encoders.yolov12_encoder import YOLOv12
            self.backbone = YOLOv12(encoder_type, pretrained=self.pretrained, out_dimList=feature_list, finetune=(not frozen_stages), use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_type.find('yolox') != -1:
            from .encoders.yolox_encoder import YOLOX
            self.backbone = YOLOX(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu, use_5_feat=use_5_feat)
        elif encoder_type.find('ppyoloe') != -1: #.startswith('ppyoloe') is True:
            from .encoders.ppyoloe_encoder import PPYOLOE
            self.backbone = PPYOLOE(encoder_type, pretrained=pretrained, out_dimList=feature_list, finetune=(not frozen_stages), use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_type.find('damoyolo') != -1:    #.startswith('damoyolo') is True:
            from .encoders.damoyolo_encoder import DAMOYOLO
            self.backbone = DAMOYOLO(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list)
        elif encoder_type.find('rtdetr') != -1:#.startswith('rtdetr') is True:
            from .encoders.rtdetr_encoder import RTDetTR
            self.backbone = RTDetTR(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list)
        elif encoder_type.find('hgnetv2') != -1:
            from .encoders.hgnetv2_encoder import HGNetv2
            self.backbone = HGNetv2(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list)
        elif encoder_type.find('dfine') != -1:
            from .encoders.dfine_encoder import DFINE
            self.backbone = DFINE(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list)
        elif encoder_type.find('deim') != -1:
            from .encoders.deim_encoder import DEIM
            self.backbone = DEIM(encoder_type, pretrained=self.pretrained, finetune=(not frozen_stages), out_dimList=feature_list)
        else:
            raise Exception("Unknown encoder!")
           
        if len(feature_list) == 0:
            feature_list = self.backbone.dimList
        
        decoder_type = decoder_type.lower()
        
        if decoder_type == 'metricleres':
            from .decoders.metricleres_decoder import MetricLeResDecoder
            self.decoder = MetricLeResDecoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        elif decoder_type == 'metriclerestruncate':
            from .decoders.metriclerestruncate_decoder import MetricLeResTruncateDecoder
            self.decoder = MetricLeResTruncateDecoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        # elif decoder_type == 'metricleresbatchnorm':
        #     from .decoders.metricleresbatchnorm_decoder import MetricLeResBatchNormDecoder
        #     self.decoder = MetricLeResBatchNormDecoder(in_channels=feature_list, interpolate=interpolate)
        elif decoder_type == 'metricleres2':
            from .decoders.metricleres2_decoder import MetricLeRes2Decoder
            self.decoder = MetricLeRes2Decoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        elif decoder_type == 'metricleres2truncate':
            from .decoders.metricleres2truncate_decoder import MetricLeRes2TruncateDecoder
            self.decoder = MetricLeRes2TruncateDecoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        # elif decoder_type == 'metricleres2batchnorm':
        #     from .decoders.metricleres2batchnorm_decoder import MetricLeRes2BatchNormDecoder
        #     self.decoder = MetricLeRes2BatchNormDecoder(in_channels=feature_list, interpolate=interpolate)
        elif decoder_type == 'metricleresdeconvupsample':
            from .decoders.metricleresdeconvupsample_decoder import MetricLeResDeConvUpsampleDecoder
            self.decoder = MetricLeResDeConvUpsampleDecoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        elif decoder_type == 'metricleresdeconvupsampletruncate':
            from .decoders.metricleresdeconvupsampletruncate_decoder import MetricLeResDeConvUpsampleTruncateDecoder
            self.decoder = MetricLeResDeConvUpsampleTruncateDecoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        elif decoder_type == 'metricleresdeconv':
            from .decoders.metricleresdeconv_decoder import MetricLeResDeConvDecoder
            self.decoder = MetricLeResDeConvDecoder(in_channels=feature_list, mid_channel=mid_channel)
        elif decoder_type == 'newcrfs':
            from .decoders.newcrfs_decoder import NeWCRFsDecoder
            self.decoder = NeWCRFsDecoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        elif decoder_type == 'newcrfstruncate':
            from .decoders.newcrfstruncate_decoder import NeWCRFsTruncateDecoder
            self.decoder = NeWCRFsTruncateDecoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        # elif decoder_type == 'newcrfsbatchnorm':
        #     from .decoders.newcrfsbatchnorm_decoder import NeWCRFsBatchNormDecoder
        #     self.decoder = NeWCRFsBatchNormDecoder(in_channels=feature_list, interpolate=interpolate)
        elif decoder_type == 'newcrfsdpt':
            from .decoders.newcrfsdpt_decoder import NeWCRFsDPTDecoder
            self.decoder = NeWCRFsDPTDecoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        elif decoder_type == 'newcrfsdpttruncate':
            from .decoders.newcrfsdpt_decoder import NeWCRFsDPTDecoder
            self.decoder = NeWCRFsDPTDecoder(in_channels=feature_list[-3:], interpolate=interpolate, mid_channel=mid_channel)
        elif decoder_type == 'newcrfs2':
            from .decoders.newcrfs2_decoder import NeWCRFs2Decoder
            self.decoder = NeWCRFs2Decoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        elif decoder_type == 'newcrfs2truncate':
            from .decoders.newcrfs2truncate_decoder import NeWCRFs2TruncateDecoder
            self.decoder = NeWCRFs2TruncateDecoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        # elif decoder_type == 'newcrfs2batchnorm':
        #     from .decoders.newcrfs2batchnorm_decoder import NeWCRFs2BatchNormDecoder
        #     self.decoder = NeWCRFs2BatchNormDecoder(in_channels=feature_list, interpolate=interpolate)
        elif decoder_type == 'newcrfs2dpt' or decoder_type == 'newcrfsdpt2' or decoder_type == 'newcrfs2dpttruncate' or decoder_type == 'newcrfsdpt2truncate':
            from .decoders.newcrfsdpt_decoder import NeWCRFsDPTDecoder
            if decoder_type.endswith('truncate') is True:
                feature_list = feature_list[-3:]
        #     from .decoders.newcrfsdpt_decoder import NeWCRFsDPTDecoder
        #     self.decoder = NeWCRFsDPTDecoder(in_channels=feature_list, interpolate=interpolate, sigmoid_last=True, mid_channel=mid_channel)
        # elif decoder_type == 'newcrfs2dpttruncate' or decoder_type == 'newcrfsdpt2truncate':
        #     from .decoders.newcrfsdpt_decoder import NeWCRFsDPTDecoder
            self.decoder = NeWCRFsDPTDecoder(in_channels=feature_list, interpolate=interpolate, sigmoid_last=True, mid_channel=mid_channel)
        elif decoder_type == 'newcrfsdeconv':
            from .decoders.newcrfsdeconv_decoder import NeWCRFsDeconvDecoder
            self.decoder = NeWCRFsDeconvDecoder(in_channels=feature_list, mid_channel=mid_channel)
        elif decoder_type.startswith('newcrfsdptdeconv') or decoder_type.startswith('newcrfsdptdeconv2'):
            from .decoders.newcrfsdptdeconv_decoder import NeWCRFsDPTDeconvDecoder
            if decoder_type.startswith('newcrfsdptdeconv2'):
                use_fusiondeconv = True
            else:
                use_fusiondeconv = False
            
            if decoder_type.endswith('truncate'):
                feature_list = feature_list[-3:]    
            
            self.decoder = NeWCRFsDPTDeconvDecoder(in_channels=feature_list, mid_channel=mid_channel, use_fusiondeconv=use_fusiondeconv, interpolate=interpolate)
        elif decoder_type == 'newcrfsdeconvupsample':
            from .decoders.newcrfsdeconvupsample_decoder import NeWCRFsDeconvUpsampleDecoder
            self.decoder = NeWCRFsDeconvUpsampleDecoder(in_channels=feature_list, mid_channel=mid_channel)
        elif decoder_type == 'newcrfsdeconvupsampletruncate':
            from .decoders.newcrfsdeconvupsampletruncate_decoder import NeWCRFsDeconvUpsampleTruncateDecoder
            self.decoder = NeWCRFsDeconvUpsampleTruncateDecoder(in_channels=feature_list, mid_channel=mid_channel)
        elif decoder_type == 'pixelformer':
            from .decoders.pixelformer_decoder import PixelFormerDecoder
            self.decoder = PixelFormerDecoder(in_channels=feature_list, max_depth=max_depth, interpolate=interpolate, mid_channel=mid_channel)
        elif decoder_type == 'pixelformerdeconv':
            from .decoders.pixelformerdeconv_decoder import PixelFormerDeconvDecoder
            self.decoder = PixelFormerDeconvDecoder(in_channels=feature_list, max_depth=max_depth, mid_channel=mid_channel)
        elif decoder_type == 'pixelformerdeconvupsample':
            from .decoders.pixelformerdeconvupsample_decoder import PixelFormerDeconvUpsampleDecoder
            self.decoder = PixelFormerDeconvUpsampleDecoder(in_channels=feature_list, max_depth=max_depth, mid_channel=mid_channel)
        elif decoder_type.startswith('dpt') is True:
            if decoder_type.endswith('truncate') is True:
                feature_list = feature_list[-3:]
            if decoder_type.find('deconv') != -1:
                if decoder_type.find('deconv2') != -1 or decoder_type.find('deconv4') != -1:
                    upsample_last = True
                else:
                    upsample_last = False
                
                if decoder_type.find('deconv3') != -1 or decoder_type.find('deconv4') != -1:
                    use_fusiondeconv = True
                else:
                    use_fusiondeconv = False
                
                from .decoders.dptdeconv_decoder import DPTDeconvDecoder
                self.decoder = DPTDeconvDecoder(in_channels=feature_list, use_fusiondeconv=use_fusiondeconv, interpolate=interpolate, mid_channel=mid_channel, upsample_last=upsample_last)
            else:
                from .decoders.dpt_decoder import DPTDecoder
                if decoder_type.startswith('dpt2') is True: #last layer is upsampler
                    sigmoid_last = False
                else:
                    sigmoid_last = True
                self.decoder = DPTDecoder(in_channels=feature_list, interpolate=interpolate, sigmoid_last=sigmoid_last, mid_channel=mid_channel)
        elif decoder_type.startswith('fpn') is True:
            if decoder_type.endswith('truncate') is True:
                feature_list = feature_list[-3:]
            if decoder_type.find('deconv') != -1:
                from .decoders.fpndeconv_decoder import FPNDeConvDecoder
                if decoder_type == 'fpndeconvupsample':
                    self.decoder = FPNDeConvDecoder(in_channels=feature_list, type=1, mid_channel=mid_channel)
                elif decoder_type == 'fpndeconvinterpolate':
                    self.decoder = FPNDeConvDecoder(in_channels=feature_list, type=2, mid_channel=mid_channel)
                else:#if decoder_type == 'fpndeconv':
                    self.decoder = FPNDeConvDecoder(in_channels=feature_list, mid_channel=mid_channel)
            else:
                from .decoders.fpn_decoder import FPNDecoder
                if decoder_type.startswith('fpn2') is True:
                    sigmoid_last = False
                else:
                    sigmoid_last = True
                self.decoder = FPNDecoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel, sigmoid_last=sigmoid_last)  
        # elif decoder_type == 'fpn':
        #     from .decoders.fpn_decoder import FPNDecoder
        #     self.decoder = FPNDecoder(in_channels=feature_list, interpolate=interpolate, sigmoid_last=False, mid_channel=mid_channel)
        # elif decoder_type == 'fpntruncate':
        #     from .decoders.fpn_decoder import FPNDecoder
        #     self.decoder = FPNDecoder(in_channels=feature_list[-3:], interpolate=interpolate, sigmoid_last=False, mid_channel=mid_channel)
        # elif decoder_type == 'fpndeconv':
        #     from .decoders.fpndeconv_decoder import FPNDeConvDecoder
        #     self.decoder = FPNDeConvDecoder(in_channels=feature_list, interpolate=interpolate, mid_channel=mid_channel)
        # elif decoder_type == 'fpndeconvtruncate':
        #     from .decoders.fpndeconv_decoder import FPNDeConvDecoder
        #     self.decoder = FPNDeConvDecoder(in_channels=feature_list[-3:], interpolate=interpolate, mid_channel=mid_channel)
        # elif decoder_type.find('fpndeconv') != -1:
        #     from .decoders.fpndeconv_decoder import FPNDeConvDecoder
        #     if decoder_type == 'fpndeconvupsample':
        #         self.decoder = FPNDeConvDecoder(in_channels=feature_list, type=1, mid_channel=mid_channel)
        #     elif decoder_type == 'fpndeconvinterpolate':
        #         self.decoder = FPNDeConvDecoder(in_channels=feature_list, type=2, mid_channel=mid_channel)
        #     else:#if decoder_type == 'fpndeconv':
        #         self.decoder = FPNDeConvDecoder(in_channels=feature_list, mid_channel=mid_channel)
        else:
            raise Exception("Unknown decoder!")

        self.max_depth = max_depth
        if decoder_type.find('pixelformer') != -1:
            self.max_depth = 1
        
        self.init_type = init_type.lower()
        
        # if not pretrained:
        self.init_params()       
                   
    def forward(self, x, get_feat=False):
        lateral_out = self.backbone(x)
        depth = self.decoder(lateral_out) * self.max_depth
        if get_feat:
            # use the feature with largest resolution
            return depth, lateral_out[0]
        return depth
    
    def init_params(self):

        for m in self.decoder.modules():
            self.init_param(m)
        if not self.pretrained: # init both encoder & decoder
            # modules = self.modules()
            for m in self.backbone.modules():
                self.init_param(m)
        elif self.backbone.conv_convert_list is not None:
            for m in self.backbone.conv_convert_list.modules():
                self.init_param(m)

    def init_param(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            # init.kaiming_normal_(m.weight, mode='fan_out')
            if self.init_type == 'normal':
                init.normal_(m.weight, std=0.01)
            elif self.init_type == 'lecun':
                init.normal_(m.weight, mean=0, std=1)
            elif self.init_type == 'uniform':
                init.uniform_(m.weight, -0.02, 0.02)
            elif self.init_type == 'orthogonal':
                init.orthogonal_(m.weight)
            elif self.init_type == 'constant':
                init.constant_(m.weight, 0.1)
            elif self.init_type == 'kaiming':
                init.kaiming_normal_(m.weight, mode='fan_out')
            elif self.init_type == 'he':
                init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            elif self.init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            elif self.init_type == 'xavier':
                init.xavier_normal_(m.weight)                    
            elif self.init_type == 'glorot_uniform':
                init.xavier_uniform_(m.weight, gain=1)
            elif self.init_type == 'sparse':
                init.sparse_(m.weight, sparsity=0.1, std=0.01)
            elif self.init_type == 'identity':
                init.eye_(m.weight)
                # init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
            # elif isinstance(m, nn.ConvTranspose2d):
            #     # init.kaiming_normal_(m.weight, mode='fan_out')
            #     init.normal_(m.weight, std=0.01)
            #     # init.xavier_normal_(m.weight)
            #     if m.bias is not None:
            #         init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):  # NN.Batchnorm2d
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     init.normal_(m.weight, std=0.01)
            #     if m.bias is not None:
            #         init.constant_(m.bias, 0) 
        elif isinstance(m, nn.LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)    