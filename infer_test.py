import os
from time import time
import cv2
import argparse
import sys
import yaml
import numpy as np

from loguru import logger

from utils import convert_arg_line_to_args, depth_value_to_depth_image
from networks.SmoothWrapper import SmoothWrapper
from networks.ONNXDepthModel import ONNXDepthModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MetricLeReS inference (video)', fromfile_prefix_chars='@')
    parser.add_argument('-c', '--config', help="configuration file *.yml", type=str, required=False, default='')

    # Input
    parser.add_argument('--input', help='Input video path', type=str, default='0')
    parser.add_argument('-w', '--weight_path',           type=str,   help='path to a checkpoint to load', default='')

    # Preprocessing
    parser.add_argument('--normalize',                             help='if set, use ImageNet normalization',                        action='store_true')

    #Postprocessing
    # parser.add_argument('--smooth_type', help='Smooth type (0: cumulative average, 1: weighted moving average, 2: exponential moving average). Defaults to 2', default=2, type=int)
    parser.add_argument('--smooth_factor', help='Smooth factor (for smooth type 2). Default is 0.7', type=float, default=0.7)
    parser.add_argument('--buffer_capacity', help='Smooth buffer capacity. Default is 15', type=int, default=15)
    
    # Log
    parser.add_argument('-o', '--output',           type=str,        help='output video path', default='') # default name is checkpoint file name

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

    # Load checkpoint
    logger.info("== Loading '{}'".format(args.weight_path))
    if args.weight_path != '':
        if os.path.isfile(args.weight_path):    
            try:
                model = ONNXDepthModel(weight_path=args.weight_path, use_cuda = True, normalize=args.normalize, use_tta=True, recover_original=True)#True)
                logger.info("== Loaded ONNX model '{}'".format(args.weight_path))
            except:
                logger.error("== No usable ONNX model found at '{}'".format(args.weight_path))
                raise FileNotFoundError   
        else:
            logger.error("== '{}' is an invalid path!".format(args.weight_path))
            raise FileNotFoundError   
    else:
        logger.error("== ONNX model path is empty!")# at '{}'".format(args.weight_path))
        raise FileNotFoundError   
        
    # Initialize smoothwrapper
    cumulative_smoother = SmoothWrapper(smooth_type=0, smooth_factor=args.smooth_factor, buffer_capacity=args.buffer_capacity)
    ema_smoother = SmoothWrapper(smooth_type=2, smooth_factor=args.smooth_factor, buffer_capacity=args.buffer_capacity)
    wma_smoother = SmoothWrapper(smooth_type=1, smooth_factor=args.smooth_factor, buffer_capacity=args.buffer_capacity)
    
    #Points
    width = 3840
    height=2160
    points = list()#[(258, 773), (258, 1200), (3520, 773), (3520, 1200)]
    offset_x = width // 3
    offset_y = 500
    
    points.append((offset_x, offset_y))
    points.append((width - 1 - offset_x, height - 1 - offset_y))
    
    points.append((offset_x, height - 1 - offset_y))
    points.append((width - 1 - offset_x, offset_y))
    
    points.append((offset_x // 2, height // 2))
    points.append((width - 1 - offset_x // 2, height // 2))    
    
    points.append((width // 2, height // 2))
    
    point_color = (0, 255, 0)#, 255)
    thickness = 4
    
    frameNum = 0
    accumFPS = 0
    pTime = 0
    writer = None
    #Read input
    video = cv2.VideoCapture(args.input)
    while True:
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        
        
        ret, frame = video.read()        
        if not ret:
            break

        start = time()
        
        depth = model(frame)
        
        cTime = time()
        fps = (1.0/(cTime - pTime))
        pTime = cTime
        accumFPS += fps
        frameNum += 1           

        cumulative_depth = cumulative_smoother(depth)
        ema_depth = ema_smoother(depth)
        wma_depth = wma_smoother(depth)
                     
        depth_vis = frame.copy()#depth_value_to_depth_image(depth) 
        
        for point in points:
            cv2.circle(depth_vis, point, 6, (255, 255, 255), 6)
            cv2.putText(depth_vis, "original: %.2fm" % depth[point[1], point[0]], (point[0] - 160, point[1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 2, point_color, thickness)
            cv2.putText(depth_vis, "ca: %.2fm" % cumulative_depth[point[1], point[0]], (point[0] - 160, point[1] + 160), cv2.FONT_HERSHEY_SIMPLEX, 2, point_color, thickness)
            cv2.putText(depth_vis, "ema: %.2fm" % ema_depth[point[1], point[0]], (point[0] - 160, point[1] + 240), cv2.FONT_HERSHEY_SIMPLEX, 2, point_color, thickness)
            cv2.putText(depth_vis, "wma: %.2fm" % wma_depth[point[1], point[0]], (point[0] - 160, point[1] + 320), cv2.FONT_HERSHEY_SIMPLEX, 2, point_color, thickness)

        # cv2.rectangle(depth_vis, (0, 0), (255, 255), (255, 255, 255))
        
        # cv2.putText(depth_vis, "FPS: %.2f" % fps, (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        # cv2.putText(depth_vis, "FPS", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        cv2.imshow("preview", depth_vis)

        if writer is None:
            writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 5, (depth_vis.shape[1], depth_vis.shape[0]))
        writer.write(depth_vis.copy())   

    logger.info(f"Running model {args.weight_path} on {args.input} successfully, average FPS: {accumFPS / frameNum}") 
    
    if writer is not None:
        writer.release()        