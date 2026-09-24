import argparse
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import yaml

from loguru import logger
from tqdm import tqdm
from PIL import Image

from networks.TFLiteDepthModel import TFLiteDepthModel
from utils import convert_arg_line_to_args, depth_value_to_depth_image, compute_errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MetricLeReS TFLite evaluation', fromfile_prefix_chars='@')
    
    parser.add_argument('-c', '--config', help="configuration file *.yml", type=str, required=False, default='')

    # Input
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for evaluation', required=False)
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for evaluation', required=False)

    # parser.add_argument('-r', '--root', help='Root directory', type=str, default='path/to/nyu')
    parser.add_argument('-d', '--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')

    parser.add_argument('-w', '--weight_path',           type=str,   help='TFLite file path', default='')
    
    # Preprocessing
    parser.add_argument('--normalize',                             help='if set, normalize the image',                              action='store_true')
    parser.add_argument('--to_grayscale',                          help='if set, convert RGB to grayscale',                          action='store_true')
    parser.add_argument('--original_scale',                        help='if set, use the normal image input value range (0->255) instead of (0->1)',                          action='store_true')
    
    # Inference setup
    parser.add_argument('-t', '--use_tta',                            help='if set, use TTA', action='store_true')
    
    # Evaluation
    parser.add_argument('--max_depth',            type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--min_depth',            type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('-e', '--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
    
    # Log
    parser.add_argument('-o', '--log_dir',           type=str,        help='output directory', default='') # default name is checkpoint file name

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
           
    try:
        model = TFLiteDepthModel(weight_path=args.weight_path, normalize=args.normalize, use_tta=args.use_tta, original_scale=args.original_scale)
        params = len(model)
        
        logger.info(f'== Loaded TFLite model {args.weight_path} successfully!. Number of params: {params}')

        if args.log_dir == '':
            args.log_dir = args.weight_path.replace(".onnx", '')
        time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        output_dir = os.path.join(args.log_dir, time_stamp)        
        os.makedirs(output_dir, exist_ok=True)
    
        eval_config_path = os.path.join(output_dir, "params.yml")
        with open(eval_config_path, 'w+') as f:
            yaml.dump(args.__dict__, f)  

        log_file = open(os.path.join(output_dir, "result.txt"), "w+")
        log_file.write(f"{params}\n")
        
    except:
        logger.error(f"== Loaded TFLite model {args.weight_path} failed! Please check it again")
        raise FileNotFoundError
    
    logger.info(f"== Start evaluating on {args.dataset} dataset (to_grayscale is {args.to_grayscale}, normalization is {args.normalize})...")
    eval_measures = np.zeros(10)
    
    samples = pd.read_csv(args.filenames_file_eval)    
    num_samples = len(samples)

    for idx in tqdm(range(num_samples)):
        
        image_path = samples.iloc[idx].image_path
        depth_path = samples.iloc[idx].depth_path
        
        if args.dataset == 'kitti':
            image_path = 'raw_dataset/' + image_path
        
        image = cv2.imread(f"{args.data_path_eval}/{image_path}")
        
        if args.to_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.repeat(image[..., np.newaxis], 3, axis=2)
        
        # basename = os.path.basename(image_path)

        # preprocess groundtruth depth
        if args.dataset == 'icms' or args.dataset == 'nyud':
            gt_depth = np.load(f"{args.data_path_eval}/{depth_path}")
        else:
            if args.dataset == 'kitti':
                depth_path = image_path.replace('raw_dataset', 'val').replace('image_02/data', 'proj_depth/groundtruth/image_02')
            gt_depth = Image.open(f"{args.data_path_eval}/{depth_path}")
            gt_depth = np.asarray(gt_depth, dtype=np.float32) 
            
            if args.dataset == 'kitti':
                gt_depth = gt_depth / 256.0
            elif args.dataset == 'nyu':
                gt_depth = gt_depth / 1000.0
        
        if args.do_kb_crop:
            # width, height = image.size#shape[0]
            height, width  = image.shape[:2]
            #width = image.shape[1]
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]# image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))  #[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            gt_depth = gt_depth[top_margin:top_margin + 352, left_margin:left_margin + 1216]

        # predicted depth
        pred_depth = model(image)
        
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

        # Save eval result
        if args.dataset == 'kitti':
            cmap = 'magma_r'
        else:
            cmap = 'magma'
        
        pred_vis = depth_value_to_depth_image(pred_depth, args.min_depth, args.max_depth, cmap)      
        gt_vis = depth_value_to_depth_image(gt_depth, args.min_depth, args.max_depth, cmap)
          
        disp_image_flat = np.hstack([image, pred_vis, gt_vis])
        tokens = image_path.split('/')
        disp_image_path = "{}/{}_{}".format(output_dir, tokens[-2], tokens[-1])
        cv2.imwrite(disp_image_path, disp_image_flat)
                
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
            
        pred_depth = pred_depth[valid_mask]
        gt_depth = gt_depth[valid_mask]
            
        if len(pred_depth) == 0 or len(gt_depth) == 0:
            logger.error(f"error on image {os.path.basename(disp_image_path)}: {len(pred_depth)}, {len(gt_depth)}")
            exit(-2)
            
        measures = compute_errors(gt_depth, pred_depth)
            
        eval_measures[:9] += measures
        eval_measures[9] += 1
        
    # eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures[9].item()
    eval_measures /= cnt
    logger.info('== Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', args.use_tta)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                    'd3'))   
    log_file.write("silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3\n")
    
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures[i]), end='')
        log_file.write('{:7.4f}, '.format(eval_measures[i]))
    print('{:7.4f}'.format(eval_measures[8]))
    log_file.write('{:7.4f}'.format(eval_measures[8]))
    log_file.close()
    
 


 
