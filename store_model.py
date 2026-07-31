import os
import argparse
import yaml
import sys
import torch
import onnx
import numpy as np

from collections import OrderedDict
from loguru import logger
from glob import glob
from onnx_opcounter import calculate_params

from utils import convert_arg_line_to_args
from xnn import quantization
from xnn.utils import load_weights as load_xnn_weights
from models_forge.utils import mongo_connection, store
from networks.MetricLeRes import MetricLeRes

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Pytorch2ONNX conversion (single model)', fromfile_prefix_chars='@')

    parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='')

    parser.add_argument("--project_name",                       type=str, help="project name",                  required=False)
    parser.add_argument("--model_application",                  type=str, help="model application",             required=False)
    parser.add_argument("--model_architecture",                 type=str, help="model application",             required=False)
    parser.add_argument("--model_version",                      type=str, help="model version",                 required=False)

    parser.add_argument('-d', '--max_depth',                   type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('-c', '--checkpoint_path',             type=str,   help='path to a checkpoint to load', default='')
    parser.add_argument('--quantize',                                      help='if set, use quantization-aware-training',                  action='store_true')
    parser.add_argument('-e', '--encoder',                     type=str,   help='type of encoder', default='efficientnetb4')
    parser.add_argument('--decoder',                           type=str,   help='type of decoder', default='metricleres')

    parser.add_argument('-l', '--feature_list', help="Feature list (default is [128,256,512,1024])", type=lambda s: [int(item) for item in s.split(',')], default = []) #128, 256, 512, 1024
    parser.add_argument('--replace_silu', help="if set, replace every SILU with RELU", action='store_true')
    parser.add_argument('--interpolate', help="if set, replace every nn.Upsample with custom Interpolate", action='store_true')

    # Output
    parser.add_argument('-o', '--output_path',                 type=str,   help='ONNX file output path', default='')
    parser.add_argument('-iw', '--input_width',                type=int,   help='input width', default=640)
    parser.add_argument('-ih', '--input_height',               type=int,   help='input height', default=480)
    
    # Output setup
    parser.add_argument('-v', '--opset_version',               type=int, help="Opset version (default is 11)", default = 11)

    if sys.argv.__len__() == 2:
        parser.convert_arg_line_to_args = convert_arg_line_to_args
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
        if args.config != '':
            args = parser.parse_args()
            yaml_data = yaml.load(open(args.config), Loader=yaml.FullLoader)
            args_dict = args.__dict__
            for key, value in yaml_data.items():
                if isinstance(value, list):
                    for v in value:
                        args_dict[key].append(v)
                else:
                    args_dict[key] = value 
                    
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            
            if "encoder" in checkpoint.keys():
                args.encoder = checkpoint["encoder"]
            
            if "decoder" in checkpoint.keys():
                args.decoder = checkpoint["decoder"]
                
            if "feature_list" in checkpoint.keys():
                args.feature_list = checkpoint["feature_list"]  
            
            if "replace_silu" in checkpoint.keys():
                args.replace_silu = checkpoint["replace_silu"]
        
            if "interpolate" in checkpoint.keys():
                args.interpolate = checkpoint["interpolate"]
            
            if "max_depth" in checkpoint.keys():
                args.max_depth = checkpoint['max_depth']
                
            if "quantize" in checkpoint.keys():
                args.quantize = checkpoint["quantize"]
                
            from networks.MetricLeRes import MetricLeRes
            model = MetricLeRes(encoder_type=args.encoder, decoder_type = args.decoder, max_depth=args.max_depth, pretrained=False, feature_list=args.feature_list, replace_silu=args.replace_silu, interpolate=args.interpolate)#, frozen_stages=args.frozen_stage)    

            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k[7:]
                new_state_dict[name] = v
            if args.quantize:
                model = quantization.QuantTrainModule(model, dummy_input = torch.zeros(1, 3, args.input_height, args.input_width), total_epochs = 50)           
                load_xnn_weights(model, new_state_dict)
            else:
                # new_state_dict = OrderedDict()
                # for k, v in checkpoint['model'].items():
                #     name = k[7:]
                #     new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            model.eval()
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            
            del checkpoint
            
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            try:
                train_config_path = [glob(f"{checkpoint_dir}/*.yml") + glob(f"{checkpoint_dir}/*.yaml")]
                train_config_path = train_config_path[0]
                config = yaml.load(open(train_config_path[0]), Loader=yaml.FullLoader)
            except:
                config = dict() # metadata
            
            config["created_by"] = config["trained_by"] = config["author"] = "hoa"
            config["model_type"] = "depth estimate" 

        else:
            print("== No checkpoint found at '{}'".format(args.checkpoint_path))
            raise FileNotFoundError 
    else:
        print("== No checkpoint found")
        raise FileNotFoundError
    
    image_torch = torch.rand((1, 3, args.input_height, args.input_width))
    
    output_path = args.output_path
    
    if output_path == '':
        if args.checkpoint_path.endswith("ckpt"):
            output_path = args.checkpoint_path.replace("ckpt", "onnx")
        elif args.checkpoint_path.endswith("pth"):
            output_path = args.checkpoint_path.replace("pth", "onnx")
    
    # Export the model
    if args.quantize:
        torch.onnx.export(model.module,                                    # model being run
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
        
    simplify_output_path = os.path.basename(output_path)
    simplify_output_path = output_path[: output_path.find('.')] + '_sim.onnx'

    try:
        os.system(f'onnxsim {output_path} {simplify_output_path}')
    except:
        pass
    
    # precaution: these commands will replace the original ONNX model with simplified ONNX model
    if os.path.exists(simplify_output_path):
        os.remove(output_path)
        os.rename(simplify_output_path, output_path)
        
    config["project_name"] = args.project_name
    config["model_application"] = args.model_application
    config["model_architecture"] = args.model_architecture
    config["model_version"] = args.model_version
    model_name = config["project_name"] + "_"+ config["model_application"] + "_" + config["encoder"] + config["decoder"] # + "_" + "V"+ str(config["model_version"])
    if args.quantize:
        model_name += "_qat"
    model_name += "_" + "V"+ str(config["model_version"])
    
    config["model_name"] = model_name

    config["train_path"] = os.path.basename(config["filenames_file"])
    config["valid_path"] = os.path.basename(config["filenames_file_eval"])
    # config["test_path"] = args.test_path    
    config["pre_processing"] = args.pre_processing
    config["post_processing"] = args.post_processing
    config["source_code"] = args.source_code
    config["in_channels"] = 3
    
    DB_USERNAME = os.environ.get("db_username")
    DB_PASSWORD =  os.environ.get("db_password")
    DB_URL = os.environ.get("db_url_oms")
    # DB_PORT = int(os.environ.get("db_port"))

    logger.info("Storing Pytorch model...")  
    # # Specify the path where you saved the model during the training
    config["model_path"] = args.checkpoint_path
    config["model_format"] = "pytorch"
    file_stats = os.stat(args.checkpoint_path)
    config["model_size"] = file_stats.st_size / (1024 * 1024)
    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    config["numbers_parameters"] = int(num_params)

    logger.info(f"connecting to DB: {DB_URL}. Username: {DB_USERNAME}. Password: {DB_PASSWORD}. authSource: {config['model_application']}...")
    client = mongo_connection(host=DB_URL, username=DB_USERNAME, password=DB_PASSWORD, authSource=config["model_application"])
    logger.info("uploading model to DB...")
    
    model_stored = store(client=client, metadata=config)
    if model_stored:
        logger.info("successfully stored Pytorch model!")
    else:
        logger.warning("Pytorch model could not be stored")

    logger.info("Storing ONNX model...")  
    # # Specify the path where you saved the model during the training
    config["model_path"] = output_path
    config["model_format"] = "onnx"
    file_stats = os.stat(output_path)
    config["model_size"] = file_stats.st_size / (1024 * 1024)
    
    params = calculate_params(onnx.load_model(output_path))
    config["numbers_parameters"] = int(params)
    logger.info(f"connecting to DB: {DB_URL}. Username: {DB_USERNAME}. Password: {DB_PASSWORD}. authSource: {config['model_application']}...")
    client = mongo_connection(host=DB_URL, username=DB_USERNAME, password=DB_PASSWORD, authSource=config["model_application"])
    logger.info("uploading model to DB...")
    
    model_stored = store(client=client, metadata=config)
    if model_stored:
        logger.info("successfully stored ONNX model!")
    else:
        logger.warning("ONNX model could not be stored!")

  