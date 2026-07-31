# from networks.custom_encoder import ResNext101, MobileNetV2, MobileNetV3, EfficientNet, ResNet, ConvNeXtV2
import gc
import os
import random
import time
import torch
import numpy as np
import argparse
import sys
try:
    from ptflops import get_model_complexity_info
except:
    pass
# from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm
from loguru import logger
try:
    from xnn import quantization
except:
    pass
from utils import convert_arg_line_to_args
from networks.MetricLeRes import MetricLeRes
from lora.injector import inject_lora_into_model

# Function for setting the seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
set_seed(42)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Model usability check.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('-e', '--encoder',                   type=str,   help='type of encoder', default='efficientnetb0')
    parser.add_argument('-d', '--decoder',                           type=str,   help='type of decoder', default='metricleres')
    parser.add_argument('-l', '--feature_list', help="Feature list (default is [])", type=lambda s: [int(item) for item in s.split(',')], default = [])
    parser.add_argument('-r', '--replace_silu', help="if set, replace every SILU with RELU", action='store_true')
    parser.add_argument('--interpolate', help="if set, replace every nn.Upsample with custom Interpolate", action='store_true')
    parser.add_argument('--use_customsilu', help="if set, use custom SILU (when replace_silu is True)", action='store_true')
    parser.add_argument('--keepnum_maxpool',                       help='Keep number of maxpool layers for each SPP maxpool block (YOLO encoders)', default = [False, False, False], type=lambda s: [bool(item) for item in s.split(',')])
    parser.add_argument('--use_5_feat',                        help="if set, use 5 features",       action='store_true')
    parser.add_argument('--mid_channel',               type=int,   help='# of intermediate channels for decoder', default=0)

    parser.add_argument('-q', '--quantize',                              help='if set, use quantization-aware-training',                  action='store_true')

    parser.add_argument('-iw', '--input_width', help="Input width", type=int, default = 640)
    parser.add_argument('-ih', '--input_height', help="Input height", type=int, default = 480)

    parser.add_argument('--rank', help="LoRA rank", type=int, default = 8)
    parser.add_argument('-a', '--alpha', help="LoRA alpha", type=int, default = 1)
    parser.add_argument('--dropout', help="LoRA dropout", type=float, default = 0.0)
    parser.add_argument('--replace', help="Replacement instead of injection", action='store_true')
    parser.add_argument('--finetune_bias', help="Finetune bias", action='store_true')
    parser.add_argument('-t', '--target_module_name', help="Target module name", type=str, default = None)
    parser.add_argument('--verbose', help="Show injected layer names", action='store_true')
    
    parser.add_argument('-f', '--fps', help="if set, measure FPS by running 10000 inference iterations", action='store_true')
    
    parser.add_argument('-c', '--convert_onnx', help='if set, convert raw model to ONNX for testing the compatibility', action='store_true')
    parser.add_argument('-o', '--opset_version',                 type=int, default = 11)

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    
    # if args.framework == 0:
    #     from networks.PixelFormer import PixelFormer
    #     model = PixelFormer(version=args.encoder, feature_list=args.feature_list)
    # elif args.framework == 1:
    #     from networks.NewCRFDepth import NewCRFDepth
    #     model = NewCRFDepth(version=args.encoder, feature_list=args.feature_list)
    # elif args.framework == 3:
    #     from networks.vadepthnet import VADepthNet
    #     model = VADepthNet(version=args.encoder, pretrained=False, feature_list=args.feature_list)#, interpolate=args.interpolate, replace_silu=args.replace_silu)
    # else:
    model = MetricLeRes(encoder_type=args.encoder, decoder_type=args.decoder, \
        pretrained=False, feature_list=args.feature_list, interpolate=args.interpolate, \
        replace_silu=args.replace_silu, use_customsilu=args.use_customsilu, \
        keepnum_maxpool=args.keepnum_maxpool, use_5_feat=args.use_5_feat, mid_channel=args.mid_channel,\
            input_height=args.input_height, input_width=args.input_width)

    model = inject_lora_into_model(model, r=args.rank, alpha=args.alpha, dropout=args.dropout, replace=args.replace, verbose=args.verbose, target_module_name=args.target_module_name, finetune_bias=args.finetune_bias)
    
    if args.quantize:
        model = quantization.QuantTrainModule(model, dummy_input = torch.zeros(1, 3, args.input_height, args.input_width), total_epochs = 50)

    # try:
    #     macs, _ = get_model_complexity_info(model, (3, args.input_height, args.input_width), as_strings=True, print_per_layer_stat=False, verbose=True)
    # except:
    #     logger.warning("Error when calculating MACs!")
    #     macs = 0
    macs = 0
        
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    num_backbone_params = sum([np.prod(p.size()) for p in model.backbone.parameters()])
    num_decoder_params = sum([np.prod(p.size()) for p in model.decoder.parameters()])

    logger.info("== LoRA MetricLeReS. Encoder: {}. Decoder: {}. Total number of parameters: {} ({} + {}). Computational Complexity: {}.".format(args.encoder, args.decoder, num_params, num_backbone_params, num_decoder_params, macs))#, args.interpolate, args.replace_silu))
    
    # cuDnn configurations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    input = torch.randn((1, 3, args.input_height, args.input_width))

    # flops = FlopCountAnalysis(model, input)
    # logger.info("flops: {} GFLOPs".format(flops.total() / 1000000000))   
    
    if args.fps:
        time_list = list()
    
        with torch.no_grad():
            input = input.cuda()
            model.cuda()
            for i in tqdm(range(10001)):
                torch.cuda.synchronize()
                tic = time.time()
                output = model(input) 
                torch.cuda.synchronize()
                time_list.append(time.time()-tic)                
        output = output.detach().cpu()
        gc.collect()
        torch.cuda.empty_cache()        
    else:
        output = model(input) 

    print(output.shape)
        
    if args.fps:
        time_list = time_list[1:]    
        logger.info("     + Done 10000 inference iterations !")
        # print("     + Total time cost: {}s".format(sum(time_list)))
        logger.info("     + Average time cost: {}s".format(sum(time_list) / 10000))
        logger.info("     + Frame Per Second : {:.2f}".format(1/(sum(time_list) / 10000)))
            
    if args.convert_onnx:
        output_path = f"{args.encoder}_{args.decoder}_lora.onnx"
        output_path_simplify = f"{args.encoder}_{args.decoder}_lora_sim.onnx"
        image_torch = torch.rand((1, 3, args.input_height, args.input_width))
        if args.quantize:
            torch.onnx.export(model.module,                             # model being run
                        image_torch,                                    # model input (or a tuple for multiple inputs)
                        output_path,                                    # where to save the model (can be a file or file-like object)
                        export_params=True,                             # store the trained parameter weights inside the model file
                        opset_version=args.opset_version,               # the ONNX version to export the model to
                        do_constant_folding=True,                       # whether to execute constant folding for optimization
                        input_names = ['input'],                        # the model's input names
                        output_names = ['output'],                      # the model's output names           
            )  
        else:
            torch.onnx.export(model,                                    # model being run
                        image_torch,                                    # model input (or a tuple for multiple inputs)
                        output_path,                                    # where to save the model (can be a file or file-like object)
                        export_params=True,                             # store the trained parameter weights inside the model file
                        opset_version=args.opset_version,               # the ONNX version to export the model to
                        do_constant_folding=True,                       # whether to execute constant folding for optimization
                        input_names = ['input'],                        # the model's input names
                        output_names = ['output'],                      # the model's output names   
            )  

        try:
            os.system(f'onnxsim {output_path} {output_path_simplify}')
        except:
            logger.warning("Cannot simplified the ONNX model")
