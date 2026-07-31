#!/usr/bin/env python
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import os, sys
import argparse
import numpy as np
import yaml

try:
    from ptflops import get_model_complexity_info
except:
    pass
from tqdm import tqdm
from loguru import logger

from networks.MetricLeRes import MetricLeRes
from utils import depth_value_to_depth_image, post_process_depth, flip_lr, compute_errors, convert_arg_line_to_args
from xnn import quantization
from xnn.utils import load_weights as load_xnn_weights

def eval(args, log_dir, log_file, model, dataloader_eval, post_process=False):
    eval_measures = torch.zeros(10).to("cuda:0")
    
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data_loader)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].to("cuda:0"))
            gt_depth = eval_sample_batched['depth']
            
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                logger.warning(f'Invalid depth on image {eval_sample_batched["name"][0]}. continue.')
                continue

            pred_depth = model(image)
            if post_process:
                image_flipped = flip_lr(image)

                pred_depth_flipped = model(image_flipped)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin : top_margin + 352, left_margin : left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        pred_depth[np.isinf(pred_depth)] = args.max_depth
        pred_depth[np.isnan(pred_depth)] = args.min_depth
        
        image_vis = cv2.imread(eval_sample_batched["name"][0])

        if args.dataset.find('kitti') != -1:
            cmap = 'magma_r'
        else:
            cmap = 'magma'
            
        if args.to_grayscale:
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2GRAY)
            image_vis = np.repeat(image_vis[..., np.newaxis], 3, axis=2)
            
        pred_vis = depth_value_to_depth_image(pred_depth, args.min_depth, args.max_depth, cmap)
        gt_vis = depth_value_to_depth_image(gt_depth, args.min_depth, args.max_depth, cmap)

        if args.do_kb_crop:
            image_vis = image_vis[top_margin:top_margin + 352, left_margin:left_margin + 1216]

        disp_image_flat = np.hstack([image_vis, pred_vis, gt_vis])
        tokens = eval_sample_batched["name"][0].split('/')
        disp_image_path = "{}/{}_{}".format(log_dir, tokens[-2], tokens[-1])
        cv2.imwrite(disp_image_path, disp_image_flat)

        if args.dataset == 'void' or args.dataset == 'nyu_short':
            valid_mask = np.logical_and(gt_depth >= args.min_depth, gt_depth <= args.max_depth)
        else:
            valid_mask = np.logical_and(gt_depth > args.min_depth, gt_depth < args.max_depth)
        
        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset.find('kitti') != -1:
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset.find('nyu') != -1:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).to("cuda:0")
        eval_measures[9] += 1
    
    torch.cuda.empty_cache()
    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    logger.info('== Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                    'd3'))
    log_file.write("silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3\n")

    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        log_file.write('{:7.4f}, '.format(eval_measures_cpu[i]))
    print('{:7.4f}'.format(eval_measures_cpu[8]))
    log_file.write('{:7.4f}'.format(eval_measures_cpu[8]))
    return eval_measures_cpu

def main_worker(args):
    
    logger.info("== Model initialized")

    if args.weight_path != '':
        if os.path.isfile(args.weight_path):

            checkpoint = torch.load(args.weight_path, map_location='cpu')
            
            if "max_depth" in checkpoint.keys():
                args.max_depth = float(checkpoint['max_depth'])
            
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

            if "quantize" in checkpoint.keys():
                args.quantize = checkpoint["quantize"]  
                
            if "normalize" in checkpoint.keys():
                args.normalize = checkpoint["normalize"]  
            
            if "original_scale" in checkpoint.keys():
                args.original_scale = checkpoint["original_scale"]
            else:
                args.original_scale = False   
                
            # if "to_grayscale" in checkpoint.keys():
            #     args.to_grayscale = checkpoint["to_grayscale"]            

            if "use_5_feat" in checkpoint.keys():
                args.use_5_feat = checkpoint["use_5_feat"]

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
                if 'num_epochs' in checkpoint.keys():
                    num_epochs = checkpoint['num_epochs']
                else:
                    num_epochs = 50
                model = quantization.QuantTrainModule(model, dummy_input = torch.zeros(1, 3, args.input_height, args.input_width), total_epochs = num_epochs)           
                load_xnn_weights(model, checkpoint['model'])
            else:               
                model = torch.nn.DataParallel(model)
                model.load_state_dict(checkpoint['model'])
                
            model.eval()
            model.to("cuda:0")   
                            
            logger.info("== Loaded checkpoint '{}' successfully!".format(args.weight_path))
            
            del checkpoint
            
        else:
            logger.error("== No checkpoint found at '{}'".format(args.weight_path))
            raise FileNotFoundError
    else:
        logger.error("== Checkpoint path empty!")
        raise FileNotFoundError
            
    if args.log_dir == '':
        if args.weight_path.endswith(".ckpt"):
            args.log_dir = args.weight_path.replace(".ckpt", "_ckpt")
        elif args.weight_path.endswith(".pth"):
            args.log_dir = args.weight_path.replace(".pth", "_pth")
        elif args.weight_path.endswith(".pt"):
            args.log_dir = args.weight_path.replace(".pt", "_pt")
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    output_dir = os.path.join(args.log_dir, time_stamp)        
    os.makedirs(output_dir, exist_ok=True)

    image_output_dir = os.path.join(output_dir, "image")
    os.makedirs(image_output_dir, exist_ok=True)

    eval_config_path = os.path.join(output_dir, "params.yml")
    with open(eval_config_path, 'w+') as f:
        yaml.dump(args.__dict__, f)
    
    log_path = os.path.join(output_dir, "result.txt")
        
    log_file = open(log_path, "w+")
    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    logger.info("== Total number of parameters: {}".format(num_params))
    log_file.write(f"{num_params}\n")

    try:
        macs, _ = get_model_complexity_info(model.module, (3, args.input_height, args.input_width), as_strings=True, print_per_layer_stat=False, verbose=True)
        logger.info("== Model complexity: {}".format(macs))
        log_file.write(f"{macs}\n")
    except:
        logger.warning("Error when calculating MACs!")
        macs = 0
    
    cudnn.benchmark = True

    dataloader_eval = NewDataLoader(args, "eval")
    logger.info(f"== Start evaluating on {args.dataset} dataset (to_grayscale is {args.to_grayscale}, normalization is {args.normalize})...")

    # ===== Evaluation ======
    model.eval()
    with torch.no_grad():
        eval(args, image_output_dir, log_file, model, dataloader_eval, post_process=args.use_tta)
    log_file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch model evaluation.', fromfile_prefix_chars='@')

    parser.add_argument('-c', '--config', help="configuration file *.yml", type=str, required=False, default='')


    parser.add_argument('-w', '--weight_path',           type=str,      help='Pytorch weight file path', default='')

    # Model architecture
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
    parser.add_argument('--decoder',                   type=str,   help='type of decoder', default='metricleres')
    parser.add_argument('-l', '--feature_list',                    help="Feature list (default is [128,256,512,1024])", type=lambda s: [int(item) for item in s.split(',')], default = [])
    parser.add_argument('--quantize',                              help='if set, use quantization-aware-training',                  action='store_true')
    parser.add_argument('--replace_silu', help="if set, replace every SILU with RELU", action='store_true')
    parser.add_argument('--interpolate', help="if set, replace every nn.Upsample with custom Interpolate", action='store_true')
    parser.add_argument('--use_customsilu', help="if set, replace every SILU with custom SILU (only when replace_silu = True)", action='store_true')
    parser.add_argument('--keepnum_maxpool',                       help='Keep number of maxpool layers for each SPP maxpool block (YOLO encoders)', default = [False, False, True], type=lambda s: [bool(item) for item in s.split(',')])
    parser.add_argument('--use_5_feat',                        help="if set, use 5 features",       action='store_true')
    parser.add_argument('--mid_channel',               type=int,   help='# of intermediate channels for decoder', default=0)

    # Dataset
    parser.add_argument('--use_cache',                             help="if set, use dataloader cache (train/eval)",       action='store_true')
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
    parser.add_argument('--input_height',              type=int,   help='input height', default=480)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
    parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--data_path',                 type=str,   help='path to the data', default='')
    parser.add_argument('--normalize',                             help='if set, use ImageNet normalization',                        action='store_true')
    parser.add_argument('--to_grayscale',                          help='if set, convert RGB to grayscale',                          action='store_true')
    parser.add_argument('--original_scale',                        help='if set, use the normal image input value range (0->255) instead of (0->1)',                          action='store_true')

    # Preprocessing
    # parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
    # parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
    # parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

    # Postprocessing
    parser.add_argument('-t', '--use_tta', help='if set, will perform TTA (flip image)', action='store_true')

    # Eval
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=False)
    # parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for evaluation', required=False)
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=False)
    parser.add_argument('--min_depth',                 type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')

    # Log and save
    parser.add_argument('--log_dir',                   type=str,   help='directory to save checkpoints and summaries', default='')

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

    args.dataset = args.dataset.lower()
    if args.dataset.find('kitti') != -1:
        from dataloaders.dataloader_kitti import KittiDataLoader as NewDataLoader
    elif args.dataset.find('nyu') != -1:
        # if args.dataset.find('nyud') != -1:
        #     from dataloaders.dataloader_nyud import NyuDDataLoader as NewDataLoader
        # else:        
        from dataloaders.dataloader_nyu import NyuDataLoader as NewDataLoader    
    else:
        raise NotImplementedError("Dataset not implemented yet!")
    # elif args.dataset.find('icms') != -1:
    #     from dataloaders.dataloader_icms import ICMSDataLoader as NewDataLoader

    torch.cuda.empty_cache()
    args.distributed = False
    # ngpus_per_node = torch.cuda.device_count()
    # if ngpus_per_node > 1:
    #     print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
    #     return -1
    main_worker(args)
