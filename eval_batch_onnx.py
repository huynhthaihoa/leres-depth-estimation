import os
import argparse
import sys
import time
import yaml

from loguru import logger
from glob import glob

from utils import convert_arg_line_to_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX model evaluation (multiple models).', fromfile_prefix_chars='@')

    parser.add_argument('-c', '--config', help="configuration file *.yml", type=str, required=False, default='')

    # Input dir
    parser.add_argument('-id', '--input_dir',           type=str,   help='Directory contains ONNX models', required=False)

    # Input
    parser.add_argument('--filenames_file_eval', help='Text file path (list of images & ground truth depth maps)', type=str, default='/hdd/hoa/Depth_Benchmark/data_splits/nyu_test.txt')
    parser.add_argument('--data_path_eval', help='Root directory', type=str, default='/hdd/team_2/robodepth/nyu')
    parser.add_argument('-d', '--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')

    parser.add_argument('--max_depth',            type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--min_depth',            type=float, help='minimum depth for evaluation', default=1e-3)

    parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('-e', '--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')

    # Preprocessing
    parser.add_argument('--normalize',                             help='if set, normalize the image',                              action='store_true')
    parser.add_argument('--to_grayscale',                          help='if set, convert RGB to grayscale',                          action='store_true')
    parser.add_argument('--original_scale',                        help='if set, use the normal image input value range (0->255) instead of (0->1)',                          action='store_true')

    # Inference setup
    parser.add_argument('-t', '--use_tta',                            help='if set, use TTA', action='store_true')

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
        
    weight_paths = glob(f"{args.input_dir}/**/*.onnx", recursive=True) #+ glob(f"{args.input_dir}/*/*.onnx")
    
    for weight_path in weight_paths:
        # new approach: create "temporary" config file for each model
        basename = os.path.basename(weight_path)
        basename = basename[: basename.rfind('.')]
        # to prevent accidentally overwriting the config file, we add a timestamp to the filename
        time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        config_path = f"tmp_eval_{basename}_{time_stamp}.yml"
        with open(config_path, 'w') as f:
            yaml.dump({'weight_path': weight_path,
                       'filenames_file_eval': args.filenames_file_eval, 
                       'data_path_eval': args.data_path_eval,
                       'dataset': args.dataset,
                       'max_depth': args.max_depth,
                       'min_depth': args.min_depth,
                       'garg_crop': args.garg_crop,
                       'eigen_crop': args.eigen_crop,
                       'do_kb_crop': args.do_kb_crop,
                       'use_tta': args.use_tta,
                       'normalize': args.normalize, 
                       'to_grayscale': args.to_grayscale, 
                       'original_scale': args.original_scale,
                       'log_dir': args.log_dir}, f) 
                   
        command = f"python eval_onnx.py -c {config_path}"
        logger.info(f"Evaluating model {weight_path}...")
        os.system(command)
        
        #remove temp file
        os.remove(config_path)
            
       
        