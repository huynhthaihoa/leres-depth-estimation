import torch
import yaml
import argparse
import os
import sys
import numpy as np
import ai_edge_torch

from loguru import logger
from collections import OrderedDict

from networks.MetricLeRes import MetricLeRes
from utils import convert_arg_line_to_args

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Pytorch2TFLite conversion (single model)', fromfile_prefix_chars='@')
    
    parser.add_argument('-c', '--config', help="configuration file *.yml", type=str, required=False, default='')

    parser.add_argument('-w', '--weight_path',             type=str,   help='Pytorch weight file path', default='')
    
    # Model arch setup
    parser.add_argument('-d', '--max_depth',                   type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('-e', '--encoder',                     type=str,   help='type of encoder', default='efficientnetb4')
    parser.add_argument('--decoder',                           type=str,   help='type of decoder', default='metricleres')
    parser.add_argument('-l', '--feature_list', help="Feature list (default is [128,256,512,1024])", type=lambda s: [int(item) for item in s.split(',')], default = []) #128, 256, 512, 1024
    parser.add_argument('--replace_silu', help="if set, replace every SILU with RELU", action='store_true')
    parser.add_argument('--use_customsilu', help="if set, replace every SILU with custom SILU (only when replace_silu = True)", action='store_true')
    parser.add_argument('--interpolate', help="if set, replace every nn.Upsample with custom Interpolate", action='store_true')
    parser.add_argument('--keepnum_maxpool',                       help='Keep number of maxpool layers for each SPP maxpool block (YOLO encoders)', default = [False, False, True], type=lambda s: [bool(item) for item in s.split(',')])
    parser.add_argument('--quantize',                                      help='if set, use quantization-aware-training', action='store_true')
    parser.add_argument('--use_5_feat',                        help="if set, use 5 features",       action='store_true')
    parser.add_argument('--mid_channel',               type=int,   help='# of intermediate channels for decoder', default=0)

    # Output setup
    parser.add_argument('-o', '--output_path',                 type=str,   help='output TFLite file path', default='')
    parser.add_argument('-iw', '--input_width',                type=int,   help='input width', default=640)
    parser.add_argument('-ih', '--input_height',               type=int,   help='input height', default=480)

    if sys.argv.__len__() == 2:
        parser.convert_arg_line_to_args = convert_arg_line_to_args
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
        if args.config != '':
            args = parser.parse_args()
            yaml_data = yaml.safe_load(open(args.config))#, Loader=yaml.FullLoader)
            args_dict = args.__dict__
            for key, value in yaml_data.items():
                if isinstance(value, list):
                    for v in value:
                        args_dict[key].append(v)
                else:
                    args_dict[key] = value 
    
    if type(args.keepnum_maxpool) is str:
        keepnum_maxpool = args.keepnum_maxpool.split()  
        args.keepnum_maxpool = list()
        for elem in keepnum_maxpool:
            if elem == "True":
                args.keepnum_maxpool.append(True)
            elif elem == "False":
                args.keepnum_maxpool.append(False)
        
    if args.weight_path != '':
        if os.path.isfile(args.weight_path):
            logger.info("== Loading weight '{}'".format(args.weight_path))
            
            checkpoint = torch.load(args.weight_path, map_location='cpu')
            
            if "encoder" in checkpoint.keys():
                args.encoder = checkpoint["encoder"]
            
            if "decoder" in checkpoint.keys():
                args.decoder = checkpoint["decoder"]
                
            if "feature_list" in checkpoint.keys():
                args.feature_list = checkpoint["feature_list"]  
            
            if "replace_silu" in checkpoint.keys():
                args.replace_silu = checkpoint["replace_silu"]
                
            if "use_customsilu" in checkpoint.keys():
                args.use_customsilu = checkpoint["use_customsilu"]

            if "keepnum_maxpool" in checkpoint.keys():
                args.keepnum_maxpool = checkpoint["keepnum_maxpool"]
                if type(args.keepnum_maxpool) is str:
                    keepnum_maxpool = args.keepnum_maxpool.split()  
                    args.keepnum_maxpool = list()
                    for elem in keepnum_maxpool:
                        if elem == "True":
                            args.keepnum_maxpool.append(True)
                        elif elem == "False":
                            args.keepnum_maxpool.append(False)  
                                  
            if "interpolate" in checkpoint.keys():
                args.interpolate = checkpoint["interpolate"]
            
            if "max_depth" in checkpoint.keys():
                args.max_depth = checkpoint['max_depth']
                
            if "quantize" in checkpoint.keys():
                args.quantize = checkpoint["quantize"]
                
            if "use_5_feat" in checkpoint.keys():
                args.use_5_feat = checkpoint["use_5_feat"]
                
            if "mid_channel" in checkpoint.keys():
                args.mid_channel = checkpoint["mid_channel"]

            if 'input_height' in checkpoint.keys():
                args.input_height = checkpoint['input_height']
            if 'input_width' in checkpoint.keys():
                args.input_width = checkpoint['input_width']
                                
            model = MetricLeRes(encoder_type=args.encoder, decoder_type = args.decoder, max_depth=args.max_depth, \
                pretrained=False, feature_list=args.feature_list, replace_silu=args.replace_silu, use_customsilu=args.use_customsilu, \
                interpolate=args.interpolate, keepnum_maxpool=args.keepnum_maxpool, use_5_feat=args.use_5_feat, mid_channel=args.mid_channel, \
                    input_width=args.input_width, input_height=args.input_height)#, frozen_stages=args.frozen_stage)    
                             
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k[7:]
                new_state_dict[name] = v
                
            if args.quantize:
                from xnn import quantization
                from xnn.utils import load_weights as load_xnn_weights

                if 'num_epochs' in checkpoint.keys():
                    num_epochs = checkpoint['num_epochs']
                else:
                    num_epochs = 50
                model = quantization.QuantTrainModule(model, dummy_input = torch.zeros(1, 3, args.input_height, args.input_width), total_epochs = num_epochs)           
                load_xnn_weights(model, new_state_dict)
            else:
                model.load_state_dict(new_state_dict)
                    
            model.eval()
            logger.info("== Loaded weight '{}'".format(args.weight_path))
            
            del checkpoint
                
        else:
            logger.error("== No weight found at '{}'".format(args.weight_path))
            raise FileNotFoundError 
    else:
        logger.error("== No weight found")
        raise FileNotFoundError
    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info("== Total number of parameters: {}".format(num_params))
    
    sample_input = (torch.randn(1, 3, args.input_height, args.input_width),)
    
    output_path = args.output_path
    
    if output_path == '':
        if args.weight_path.endswith("ckpt"):
            output_path = args.weight_path.replace("ckpt", "tflite")
        elif args.weight_path.endswith("pth"):
            output_path = args.weight_path.replace("pth", "tflite")
        elif args.weight_path.endswith("pt"):
            output_path = args.weight_path.replace("pt", "tflite")

    # Convert and serialize PyTorch model to a tflite flatbuffer. Note that we
    # are setting the model to evaluation mode prior to conversion.
    edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    edge_model.export(output_path)
    
    logger.info(f"Converting MetricLeReS with encoder {args.encoder}, decoder {args.decoder}, feature list {args.feature_list} to TFLite successfully! Output path is {output_path}")
