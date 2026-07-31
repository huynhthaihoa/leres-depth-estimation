import os
import argparse
import yaml
import sys

from loguru import logger

from models_forge.utils import mongo_connection, store
from utils import convert_arg_line_to_args

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Store ONNX model to DB", fromfile_prefix_chars="@")
    parser.add_argument('-c', '--config',                     type=str,                                            help="configuration file *.yml",                                  default='')
    
    parser.add_argument("--project_name",               type=str, help="project name",                  required=False)
    parser.add_argument("--model_application",          type=str, help="model application",             required=False)

    # Model setup
    parser.add_argument("--model_architecture",         type=str, help="model application",             required=False)
    parser.add_argument("--model_version",              type=str, help="model version",                 required=False)

    # parser.add_argument("-r", "--root",                 type=str, help="root directory",                required=False)
    # parser.add_argument("--dataset",                    type=str, help="dataset",                       default="kakaov2.0", required=False)
    # parser.add_argument("--test_data",                  type=str, help="test data directory",            required=False)
    
    # parser.add_argument("--train_path",                 type=str, help="train path",                    required=False)
    # parser.add_argument("--valid_path",                 type=str, help="valid path",                    required=False)
    # parser.add_argument("--test_path",                  type=str, help="test path",                     required=False)
    parser.add_argument("--source_code",                type=str, help="source code",                   required=False)
    parser.add_argument("--pre_processing",             type=str, help="preprocessing",                 required=False)
    parser.add_argument("--post_processing",            type=str, help="post processing",               required=False)
    
    # Input
    parser.add_argument('-w', "--weight_path",      type=str, help="ONNX file path",  default="")
    parser.add_argument("--train_config_path",    type=str, help="training config file path",  default="")

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
        config = yaml.load(open(args.train_config_path), Loader=yaml.FullLoader)
    except:
        config = dict() # metadata
    
    config["created_by"] = config["trained_by"] = config["author"] = "hoa"
    config["model_type"] = "Depth estimation"
    config["project_name"] = args.project_name
    config["model_application"] = args.model_application
    config["model_architecture"] = args.model_architecture
    config["model_version"] = args.model_version
    model_name = config["project_name"] + "_"+ config["model_application"] + "_" + config["encoder"] + config["decoder"] 
    if config["quantize"] is True:
        model_name += "_qat"
    model_name += "_" + "V"+ str(config["model_version"])
    config["model_name"] = model_name

    config["train_path"] = os.path.basename(config["filenames_file"])
    config["valid_path"] = os.path.basename(config["filenames_file_eval"])
    config["pre_processing"] = args.pre_processing
    config["post_processing"] = args.post_processing
    config["source_code"] = args.source_code
    config["in_channels"] = 3
    
    DB_USERNAME = os.environ.get("db_username")
    DB_PASSWORD =  os.environ.get("db_password")
    DB_URL = os.environ.get("db_url_oms")

    logger.info("Storing ONNX model...")  
    # # Specify the path where you saved the model during the training
    config["model_path"] = args.weight_path
    config["model_format"] = "onnx"
    file_stats = os.stat(args.weight_path)
    config["model_size"] = file_stats.st_size / (1024 * 1024)
    logger.info(f"connecting to DB: {DB_URL}. Username: {DB_USERNAME}. Password: {DB_PASSWORD}. authSource: {config['model_application']}...")
    client = mongo_connection(host=DB_URL, username=DB_USERNAME, password=DB_PASSWORD, authSource=config["model_application"])
    logger.info("uploading model to DB...")
    
    model_stored = store(client=client, metadata=config)
    if model_stored:
        logger.info("successfully stored ONNX model!")
    else:
        logger.warning("ONNX model could not be stored!")


