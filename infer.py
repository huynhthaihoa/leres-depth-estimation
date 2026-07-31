import cv2
import argparse
import os
import sys
import yaml
import torch
import numpy as np

from loguru import logger
from time import time

from utils import convert_arg_line_to_args, depth_value_to_depth_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MetricLeReS inference (video/webcam)', fromfile_prefix_chars='@')

    parser.add_argument('-c', '--config', help="configuration file *.yml", type=str, required=False, default='')

    # Input
    parser.add_argument('--input', help='Input path (image/video/camera index)', type=str, default='0')
    parser.add_argument('-w', '--weight_path',           type=str,   help='path to a checkpoint to load', default='')

    # Preprocessing
    parser.add_argument('--normalize',                             help='if set, use ImageNet normalization',                        action='store_true')
    parser.add_argument('--original_scale',                        help='if set, use the normal image input value range (0->255) instead of (0->1)',                          action='store_true')
    parser.add_argument('--to_grayscale',                        help='if set, convert input into grayscale domain',                          action='store_true')

    # Inference setup
    parser.add_argument('-g', '--use_cuda',                            help='if set, use CUDA', action='store_true')
    parser.add_argument('-t', '--use_tta',                            help='if set, use TTA', action='store_true')
    parser.add_argument('-m', '--color_map', type=str, help='color map for depth visualization: recommendation for short range is magma (default), recommendation for long range is magma_r', default='magma')

    # Pytorch model architecture setup
    parser.add_argument('-f', '--framework',                   type=int,   help='framework: 0 is PixelFormer (default), 1 is NeWCRFs, 2 is MetricLeReS', default=2)
    parser.add_argument('-d', '--max_depth',                   type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('-e', '--encoder',                     type=str,   help='type of encoder', default='efficientnetb4')
    parser.add_argument('--decoder',                           type=str,   help='type of decoder', default='metricleres')
    parser.add_argument('-l', '--feature_list',                            help="Feature list (default is [128,256,512,1024])", type=lambda s: [int(item) for item in s.split(',')], default = []) #128, 256, 512, 1024
    parser.add_argument('--replace_silu',                                  help="if set, replace every SILU with RELU", action='store_true')
    parser.add_argument('--use_customsilu',                                help="if set, replace every SILU with custom SILU (only when replace_silu = True)", action='store_true')
    parser.add_argument('--interpolate',                                   help="if set, replace every nn.Upsample with custom Interpolate", action='store_true')
    parser.add_argument('--keepnum_maxpool',                               help='Keep number of maxpool layers for each SPP maxpool block (YOLO encoders)', default = [False, False, True], type=lambda s: [bool(item) for item in s.split(',')])
    parser.add_argument('--use_5_feat',                        help="if set, use 5 features",       action='store_true')
    parser.add_argument('--mid_channel',               type=int,   help='# of intermediate channels for decoder', default=0)

    parser.add_argument('--quantize',                                      help='if set, use quantization-aware-training', action='store_true')
    parser.add_argument('--input_height',              type=int,   help='input height', default=480)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640)

    # Log
    parser.add_argument('-od', '--output_depth_path',           type=str,        help='output depth path (NPY format)', default='') # default name is checkpoint file name    
    parser.add_argument('-oi', '--output_image_path',           type=str,        help='output image path', default='') # default name is checkpoint file name
    parser.add_argument('-ov', '--output_video_path',           type=str,        help='output video path', default='') # default name is checkpoint file name
    parser.add_argument('-s', '--show',           help='if set, show the output', action='store_true') # default name is checkpoint file name

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
    
    is_cuda_available = (args.use_cuda and torch.cuda.is_available())
    
    # Load checkpoint
    logger.info("== Loading '{}'".format(args.weight_path))
    if args.weight_path != '':
        if os.path.isfile(args.weight_path):    
            
            is_tflite_model = False            
            is_onnx_model = args.weight_path.endswith('onnx')
            
            if is_onnx_model:
                from networks.ONNXDepthModel import ONNXDepthModel
                try:
                    model = ONNXDepthModel(weight_path=args.weight_path, use_cuda = is_cuda_available, normalize=args.normalize, use_tta=args.use_tta, original_scale=args.original_scale)#, recover_original=True)
                    logger.info("== Loaded ONNX model '{}'".format(args.weight_path))
                except:
                    logger.error("== No usable ONNX model found at '{}'".format(args.weight_path))
                    raise FileNotFoundError                
            else:   
                
                is_tflite_model = args.weight_path.endswith('tflite')
                
                if is_tflite_model:
                    from networks.TFLiteDepthModel import TFLiteDepthModel
                    try:
                        model = TFLiteDepthModel(weight_path=args.weight_path, normalize=args.normalize, use_tta=args.use_tta, original_scale=args.original_scale)#, recover_original=True)
                        logger.info("== Loaded TFLite model '{}'".format(args.weight_path))
                    except:
                        logger.error("== No usable TFLite model found at '{}'".format(args.weight_path))
                        raise FileNotFoundError                     
                else:
                    from networks.MetricLeRes import MetricLeRes
                    from utils import preprocess_image, remove_padding, flip_lr, post_process_depth
            
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
                    if "normalize" in checkpoint.keys():
                        args.normalize = checkpoint["normalize"]
                    if "original_scale" in checkpoint.keys():
                        args.original_scale = checkpoint["original_scale"]
                    else:
                        args.original_scale = False   
                    if "mid_channel" in checkpoint.keys():
                        args.mid_channel = checkpoint["mid_channel"]

                    if 'input_height' in checkpoint.keys():
                        args.input_height = checkpoint['input_height']
                    if 'input_width' in checkpoint.keys():
                        args.input_width = checkpoint['input_width']  
                    
                    model = MetricLeRes(encoder_type=args.encoder, decoder_type = args.decoder, max_depth=args.max_depth, \
                        pretrained=False, feature_list=args.feature_list, replace_silu=args.replace_silu, interpolate=args.interpolate, \
                        use_customsilu=args.use_customsilu, keepnum_maxpool=args.keepnum_maxpool, use_5_feat=args.use_5_feat, mid_channel=args.mid_channel, \
                            input_width=args.input_width, input_height=args.input_height)#, frozen_stages=args.frozen_stage)    

                    if args.quantize:
                        
                        from xnn import quantization
                        from xnn.utils import load_weights as load_xnn_weights

                        if 'num_epochs' in checkpoint.keys():
                            num_epochs = checkpoint['num_epochs']
                        else:
                            num_epochs = 50
                        
                        model = quantization.QuantTrainModule(model, dummy_input = torch.zeros(1, 3, args.input_height, args.input_width), total_epochs = num_epochs)           
                        load_xnn_weights(model, checkpoint['model'])
                        model.eval()

                        if is_cuda_available:
                            model.to("cuda:0")
                    else:
                        if is_cuda_available:                    
                            model = torch.nn.DataParallel(model)
                            model.load_state_dict(checkpoint['model'])
                            model.eval()
                            model.to("cuda:0")
                        else:
                            from collections import OrderedDict
                            new_state_dict = OrderedDict()
                            for k, v in checkpoint['model'].items():
                                name = k[7:]
                                new_state_dict[name] = v
                            model.load_state_dict(new_state_dict)
                            model.eval()
                    logger.info("== Loaded Pytorch weight '{}'".format(args.weight_path))
                    del checkpoint
                    if is_cuda_available:
                        torch.cuda.empty_cache()

        else:
            logger.error("== No usable Pytorch weight found at '{}'".format(args.weight_path))
            raise FileNotFoundError            
    else:
        logger.error("== No Pytorch/ONNX/TFLite model found")
        raise FileNotFoundError  
                
    # Read input
    input = args.input
    if isinstance(input, str) and (input.endswith('png') or input.endswith('jpg') or input.endswith('jpeg') or input.endswith('png')): # input is image
        image = cv2.imread(input)
        if is_onnx_model or is_tflite_model:
            depth = model(image, args.to_grayscale)
        else:
            image_tensor, padding_values = preprocess_image(image, args.normalize, args.to_grayscale)
            
            if is_cuda_available:
                image_tensor = image_tensor.to("cuda:0") 
            depth = model(image_tensor)  
            
            if args.use_tta:
                image_tensor_flipped = flip_lr(image_tensor)
                depth_flipped = model(image_tensor_flipped)
                depth = post_process_depth(depth, depth_flipped)
            
            if is_cuda_available:
                depth = depth.cpu().detach().numpy().squeeze()
            else:
                depth = depth.numpy().squeeze() 
            
            if padding_values is not None:
                depth = remove_padding(depth, padding_values)
        
        if args.output_depth_path != '':
            np.save(args.output_depth_path, depth)
        
        depth_vis = depth_value_to_depth_image(depth, cmap=args.color_map)#, applyColor=args.color_map)  # max_depth=max_depth, applyColor=args.color_map)    
        final_vis = cv2.vconcat([image, depth_vis])

        if args.show:
            cv2.imshow("preview", final_vis)
            cv2.waitKey(0)
        
        if args.output_image_path != '':
            cv2.imwrite(args.output_image_path, depth_vis)
        
        logger.info(f"Running model {args.weight_path} on {args.input} successfully!")                  
    else: # input is video/camera
        if isinstance(input, int):
            try:
                cam = cv2.VideoCapture(input, cv2.CAP_DSHOW)
            except:
                cam = cv2.VideoCapture(input)
        else:
            cam = cv2.VideoCapture(input)
                    
        frameNum = 0
        accumFPS = 0
        pTime = 0
        writer = None
        
        # Inference loop
        with torch.no_grad():
            while True:
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break

                start = time()
                
                ret, frame = cam.read()
                if not ret:
                    break
                                
                if is_onnx_model:
                    depth = model(frame)
                else:
                    image_tensor, padding_values = preprocess_image(frame, args.normalize)
                    if is_cuda_available:
                        image_tensor = image_tensor.to("cuda:0")
                    depth = model(image_tensor)
                    
                    if args.use_tta:
                        image_tensor_flipped = flip_lr(image_tensor)
                        depth_flipped = model(image_tensor_flipped)
                        depth = post_process_depth(depth, depth_flipped)
                    
                    if is_cuda_available:
                        depth = depth.cpu().numpy().squeeze()
                    else:
                        depth = depth.numpy().squeeze()
                    
                    if padding_values is not None:
                        depth = remove_padding(depth, padding_values)
                
                cTime = time()
                fps = int(1/(cTime - pTime))
                pTime = cTime
                accumFPS += fps
                frameNum += 1           
                     
                depth_vis = depth_value_to_depth_image(depth) 
                final_vis = cv2.hconcat([frame, depth_vis])
                
                cv2.putText(final_vis, "FPS: %d" % fps, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                if args.show:
                    cv2.imshow("preview", final_vis)
                        
                if args.output_video_path != '':
                    if writer is None:
                        writer = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (depth_vis.shape[1], depth_vis.shape[0]))
                    writer.write(depth_vis.copy())   
                
        torch.cuda.empty_cache()
        logger.info(f"Running model {args.weight_path} on {args.input} successfully, average FPS: {accumFPS / frameNum}")         