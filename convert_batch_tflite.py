from glob import glob
import os
import argparse
import sys
import yaml

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Pytorch2TFLite conversion (multiple models).', fromfile_prefix_chars='@')
    
    parser.add_argument('-c', '--config', help="configuration file *.yml", type=str, required=False, default='')
   
    # Input dir
    parser.add_argument('-id', '--input_dir',           type=str,   help='Directory contains Pytorch weights', required=False)
    parser.add_argument('-od', '--output_dir',          type=str,   help='Directory contains output TFLite files', default='')

    # Output setup  
    parser.add_argument('-iw', '--input_width',                   type=int,   help='input width', default=640)
    parser.add_argument('-ih', '--input_height',                   type=int,   help='input height', default=480)
   
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
        
    paths = glob(f"{args.input_dir}/**/*.ckpt", recursive=True) # + glob(f"{args.input_dir}/*/*.ckpt")

    for path in paths:
        tflite_path = path.replace("ckpt", "tflite")
        if args.output_dir != '':
            tflite_path = tflite_path.replace(args.input_dir, args.output_dir)
        command = f"python convert_tflite.py -w {path} -o {tflite_path} -iw {args.input_width} -ih {args.input_height}"
        os.system(command)
        