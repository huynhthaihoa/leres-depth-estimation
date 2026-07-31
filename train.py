import cv2
import torch
import os
import sys
import time
import argparse
import yaml
import numpy as np

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from collections import OrderedDict
from tqdm import tqdm
from loguru import logger
from tensorboardX import SummaryWriter
from dvclive import Live

from networks.MetricLeRes import MetricLeRes
from xnn import quantization
from xnn.utils import load_weights as load_xnn_weights
from xnn.utils import AverageMeter as XNNAverageMeter
from xnn.optim.lr_scheduler import SchedulerWrapper as XNNScheduler
from early_stopping import EarlyStopping, AverageMeter
import utils
# from utils import EMA, depth_value_to_depth_image, compute_errors, eval_metrics, normalize_result,\
#                   block_print, enable_print, convert_arg_line_to_args, set_seeds
          
def eval(model, eval_dataloader, gpu, ngpus, eval_criterion, eval_loss_meter, eval_logger, epoch, global_step, args):
    """Evaluation step

    Args:
        model: model for evaluation
        eval_dataloader: dataloader for evaluation
        gpu: GPU id
        ngpus: # of GPUS
        eval_criterion: eval criterion
        eval_loss_meter: eval loss meter
        eval_logger: DVC eval writer to write eval results
        epoch: current epoch
        global_step: current global step
        args: argument list

    Returns:
        eval_measures: evaluation result on 9 metrics (silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3) or None if error occurs
    """    
    eval_measures = torch.zeros(10).cuda(device=gpu)
    
    use_dvc = isinstance(eval_logger, Live)

    for _, eval_sample_batched in enumerate(tqdm(eval_dataloader.data_loader)):
        
        image = torch.autograd.Variable(eval_sample_batched["image"].cuda(gpu, non_blocking=True))
        gt_depth = eval_sample_batched["depth"].cuda(gpu, non_blocking=True) # (1, H, W)

        has_valid_depth = eval_sample_batched['has_valid_depth']
        if not has_valid_depth:
            logger.warning(f'Invalid depth on image {eval_sample_batched["name"][0]}. continue.')
            continue

        with torch.no_grad():
            
            pred_depth = model(image)
            
        pred_depth = pred_depth.cpu().numpy().squeeze()
        gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        pred_depth[np.isinf(pred_depth)] = args.max_depth
        pred_depth[np.isnan(pred_depth)] = args.min_depth
        
        image_vis = cv2.imread(eval_sample_batched["name"][0])
        if args.to_grayscale:
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2GRAY)
            image_vis = np.repeat(image_vis[..., np.newaxis], 3, axis=2)
        else:
            image_vis = cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB)

        if args.dataset.find('kitti') != -1:
            cmap = 'magma_r'
        else:
            cmap = 'magma'
                        
        pred_vis = utils.depth_value_to_depth_image(pred_depth, args.min_depth, args.max_depth, cmap)
        gt_vis = utils.depth_value_to_depth_image(gt_depth, args.min_depth, args.max_depth, cmap)
        
        if args.do_kb_crop:
            image_vis = image_vis[top_margin:top_margin + 352, left_margin:left_margin + 1216]
        
        if args.dataset == 'nyu_crop_resize':
            image_vis = cv2.resize(image_vis, (args.input_width, args.input_height))

        disp_image_flat = np.hstack([image_vis, pred_vis, gt_vis]) # (H, 3W, C)        
        tokens = eval_sample_batched["name"][0].split('/')

        if use_dvc:
            eval_logger.log_image(f"{epoch}_{global_step}/{tokens[-2]}_{tokens[-1]}.jpg", disp_image_flat)
        else:
            # to track the change over time, each image is saved with the same tag
            eval_logger.add_image(f"image/{tokens[-2]}_{tokens[-1]}", disp_image_flat[:,:,::-1], global_step, dataformats='HWC')

        if args.dataset == 'void' or args.dataset == 'nyu_short' or args.dataset.find('d435f') != -1:
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
                #elif args.dataset == 'nyu' or args.dataset == 'nyu_crop':
                elif args.dataset.find('nyu') != -1:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)
                
        pred_depth = pred_depth[valid_mask]
        gt_depth = gt_depth[valid_mask]
        
        if len(pred_depth) == 0 or len(gt_depth) == 0:
            logger.error("error!")
            continue
        
        eval_loss = eval_criterion.forward(torch.tensor(pred_depth), torch.tensor(gt_depth))
                
        eval_loss_meter.update(eval_loss.item())

        measures = utils.compute_errors(gt_depth, pred_depth)
        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt

        if use_dvc:
            # eval_logger.log_metric(f"{epoch}_{global_step}/loss/eval_loss", eval_loss.item())#, global_step)
            eval_logger.log_metric("avg_loss", eval_loss_meter.avg, True)
            for i in range(9):
                eval_logger.log_metric(utils.eval_metrics[i], eval_measures_cpu[i].item())
            eval_logger.next_step()
        else:
            # eval_logger.add_scalar("loss/eval_loss", eval_loss.item(), global_step)
            eval_logger.add_scalar("avg_loss", eval_loss_meter.avg, global_step)
            for i in range(9):
                eval_logger.add_scalar(f"metric/{utils.eval_metrics[i]}", eval_measures_cpu[i].item(), global_step)
            eval_logger.flush()
        logger.info('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ')#, post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None

def train(gpu, ngpus_per_node, args):
    """Main training function
    """    

    args.gpu = gpu

    if args.gpu is not None:
        logger.info("== Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    if args.use_early_stopping:
        early_stopping = EarlyStopping(patience=args.lr_patience, verbose=True)
        
    torch.set_num_threads(args.num_threads)

    if type(args.keepnum_maxpool) is str:
        keepnum_maxpool = args.keepnum_maxpool.split()  
        args.keepnum_maxpool = list()
        for elem in keepnum_maxpool:
            if elem == "True":
                args.keepnum_maxpool.append(True)
            elif elem == "False":
                args.keepnum_maxpool.append(False)

    if args.data_path_eval == '':
        args.data_path_eval = args.data_path

    if args.original_scale:
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        # logger.info("Keep original scale")
        # from utils import inv_standard_image as inv_image
    else:
        if args.normalize:
            # from utils import inv_normalize_image as inv_image
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
        else:
            # from utils import inv_unnormalize_image as inv_image 
            mean = [0., 0., 0.]
            std = [1., 1., 1.]
            
    # Dataloader
    args.dataset = args.dataset.lower()
    if args.dataset.find('kitti') != -1:
        from dataloaders.dataloader_kitti import KittiDataLoader as DataLoader
    elif args.dataset.find('nyu') != -1:
        if args.dataset.find('shift2') != -1:
            from dataloaders.dataloader_nyushift2 import NyuShift2DataLoader as DataLoader
        elif args.dataset.find('shift') != -1:
            from dataloaders.dataloader_nyushift import NyuShiftDataLoader as DataLoader
        else:
            from dataloaders.dataloader_nyu import NyuDataLoader as DataLoader
    #     elif args.dataset.find('nyud') != -1:
    #         from dataloaders.dataloader_nyud import NyuDDataLoader as DataLoader
    #     else:
    #         from dataloaders.dataloader_nyu import NyuDataLoader as DataLoader
    # elif args.dataset.find('icms') != -1:
    #     from dataloaders.dataloader_icms import ICMSDataLoader as DataLoader
    else:
        raise Exception("Dataloader not defined!")    
    train_dataloader = DataLoader(args, "train")
    eval_dataloader = DataLoader(args, "eval")

    # Model
    model = MetricLeRes(encoder_type=args.encoder, decoder_type=args.decoder, max_depth=args.max_depth, \
        pretrained=args.pretrained, frozen_stages=args.freeze_encoder, \
        feature_list=args.feature_list, replace_silu=args.replace_silu, use_customsilu=args.use_customsilu, \
        interpolate=args.interpolate, init_type=args.init_type, keepnum_maxpool=args.keepnum_maxpool, \
        use_5_feat=args.use_5_feat, mid_channel=args.mid_channel,
        input_width=args.input_width, input_height=args.input_height)

    if args.quantize:
        model = quantization.QuantTrainModule(model, dummy_input = torch.zeros(1, 3, args.input_height, args.input_width), total_epochs = args.num_epochs)
        
    # load weight & optimizer from checkpoint
    optimizer_state_dict = None
    if args.weight_path != '':
        if os.path.isfile(args.weight_path):
            logger.info("== Loading checkpoint '{}'".format(args.weight_path))
            
            if args.gpu is None:
                checkpoint = torch.load(args.weight_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.weight_path, map_location=loc)
            
            if args.quantize:
                load_xnn_weights(model, checkpoint['model'])
            else:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['model'].items():
                    name = k[7:]
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
                            
            if not args.retrain:
                
                if 'optimizer' in list(checkpoint.keys()):
                    optimizer_state_dict = checkpoint['optimizer']
                
                if 'scheduler' in list(checkpoint.keys()):
                    scheduler.load_state_dict(checkpoint['scheduler'])
                
                if 'global_step' in list(checkpoint.keys()):
                    global_step = checkpoint['global_step']
                    logger.info("== Loaded checkpoint '{}' (global_step {})".format(args.weight_path, checkpoint['global_step']))
                else:
                    global_step = 0 # hard code
                    logger.info("== Loaded checkpoint '{}'".format(args.weight_path))
            else:
                global_step = 0
                logger.info("== Loaded checkpoint '{}'".format(args.weight_path))

        else:
            logger.error("== No checkpoint found at '{}'".format(args.weight_path))
        model_just_loaded = True
        del checkpoint
    else:
        global_step = 0
                    
    model.train()

    # Distributed setup
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            if args.quantize:
                model = torch.nn.parallel.DistributedDataParallel(model.module, device_ids=[args.gpu], find_unused_parameters=True)
            else:   
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            if args.quantize:
                model = torch.nn.parallel.DistributedDataParallel(model.module, find_unused_parameters=True)
            else:   
                model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        if args.quantize:
            model = torch.nn.DataParallel(model.module)
        else:
            model = torch.nn.DataParallel(model)
        model.cuda()
        
    # Define optimizer
    args.optimizer = args.optimizer.lower()
    opt_params = [
                        {'params': model.module.decoder.parameters()},
                        {'params': model.module.backbone.parameters(), 'lr': args.learning_rate * args.learning_rate_ratio}
                        ]
    if args.optimizer == 'adam':
        if args.default_decay:
            optimizer = torch.optim.Adam(opt_params,
                        lr=args.learning_rate)
        else:           
            optimizer = torch.optim.Adam(opt_params,
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        eps=args.adam_eps)
    elif args.optimizer == 'adagrad':
        if args.default_decay:
            optimizer = torch.optim.Adagrad(opt_params,
                        lr=args.learning_rate)
        else:      
            optimizer = torch.optim.Adagrad(opt_params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=args.adam_eps          
            )    
    elif args.optimizer == 'adamw':
        if args.default_decay:
            optimizer = torch.optim.AdamW(opt_params,
                        lr=args.learning_rate)
        else:                  
            optimizer = torch.optim.AdamW(
                opt_params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=args.adam_eps
            )
    elif args.optimizer == 'amsgrad':
        if args.default_decay:
            optimizer = torch.optim.Adam(opt_params,
                    lr=args.learning_rate,
                    amsgrad=True)
        else:
            optimizer = torch.optim.Adam(opt_params,
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    eps=args.adam_eps, 
                    amsgrad=True)
    elif args.optimizer == 'rmsprop': 
        if args.default_decay:
            optimizer = torch.optim.RMSprop(opt_params,
            lr=args.learning_rate)
        else:      
            optimizer = torch.optim.RMSprop(opt_params,
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay,
                        eps=args.adam_eps)
    elif args.optimizer == 'sgd':
        if args.default_decay:
            optimizer = torch.optim.SGD(opt_params,
            lr=args.learning_rate)
        else:
            optimizer = torch.optim.SGD(opt_params,
                        lr=args.learning_rate,
                        weight_decay=args.weight_decay)
    elif args.optimizer == 'adashift':
        from optimizer.adashift import AdaShift
        if args.default_decay:
            optimizer = AdaShift(opt_params,
            lr=args.learning_rate)
        else:
            optimizer = AdaShift(opt_params,
                lr=args.learning_rate,
                eps=args.adam_eps            
            )
    elif args.optimizer.startswith('adopt') is True:
        from optimizer.adopt import ADOPT
        if args.default_decay:
            optimizer = ADOPT(opt_params,
            lr=args.learning_rate,
            decoupled=(args.optimizer == 'adoptdecoupled')
        )
        else:
            optimizer = ADOPT(opt_params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=args.adam_eps,
                decoupled=(args.optimizer == 'adoptdecoupled')
            )
    elif args.optimizer == 'soap':
        from optimizer.soap import SOAP
        if args.default_decay:
            optimizer = SOAP(opt_params,
                lr=args.learning_rate)  
        else:          
            optimizer = SOAP(opt_params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                eps=args.adam_eps, 
                precondition_frequency=10)
    else:
        raise Exception("optimizer not defined!")
            
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    
    num_params = sum([np.prod(p.size()) for p in model.module.parameters()])
    logger.info("== Total number of parameters: {}".format(num_params))
    num_params_update = sum([np.prod(p.shape) for p in model.module.parameters() if p.requires_grad])
    logger.info("== Total number of learning parameters: {}".format(num_params_update))

    # Log # of parameters
    args.num_params = int(num_params)
    args.num_params_update = int(num_params_update)

    # Define scheduler
    if args.quantize:
        scheduler = XNNScheduler(scheduler_type='step', optimizer=optimizer, epochs=args.num_epochs, warmup_factor=1e-3, max_iter=args.num_epochs * len(train_dataloader.data_loader))
    else:
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.lr_patience,threshold_mode='abs', min_lr=1e-8, verbose=True)
    
    # Use EMA
    ema = None
    if args.use_ema:
        ema = utils.EMA(model,             
            decay=args.model_ema_decay)#,
            # use_warmup=args.model_ema_warmup,
            # device='cpu' if args.model_ema_force_cpu else None,
        # )
        # ema = utils.

    # Distributed training
    if args.distributed:
        logger.info("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        logger.info("== Model Initialized")
        
    # Auto mixed precision training
    scaler = None
    if args.use_amp:
        scaler = torch.amp.GradScaler()

    best_eval_measures_lower_better = torch.ones(6) * torch.inf#.cpu() #+ 1e3
    best_eval_measures_higher_better = torch.zeros(3)#.cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)
    best_eval_epochs = np.zeros(9, dtype=np.int32)
    model_just_loaded = False
    cudnn.benchmark = True

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        
        train_summary_dir = os.path.join(args.log_dir, 'train')
        eval_summary_dir = os.path.join(args.log_dir, 'eval')

        if args.use_dvc:
            train_logger = Live(dir=train_summary_dir, 
                            #    report='html',
                               monitor_system=True,
                               dvcyaml="dvc.yaml") 
            # save training hyperparams
            train_logger.log_params(args.__dict__)

            # evaluation logger
            eval_logger = Live(dir=eval_summary_dir,
                                #    report='html',
                                dvcyaml="dvc.yaml")#,   
        else:
            if os.path.exists(train_summary_dir) is False:
                os.makedirs(train_summary_dir)
            train_logger = SummaryWriter(train_summary_dir, flush_secs=30)
            
            train_config_path = os.path.join(train_summary_dir, 'params.yml')
            with open(train_config_path, 'w+') as f:
                yaml.dump(args.__dict__, f)  
            
            if os.path.exists(eval_summary_dir) is False:
                os.makedirs(eval_summary_dir)
            eval_logger = SummaryWriter(eval_summary_dir, flush_secs=30)
            
    # Loss function(s)
    args.loss_func = args.loss_func.lower()
    if args.loss_func == 'zoe':
        from utils import ZoeSilogLoss
        silog_criterion = ZoeSilogLoss()
        eval_criterion = ZoeSilogLoss()
    elif args.loss_func == 'newcrfs':
        from utils import NewcrfsSilogLoss
        silog_criterion = NewcrfsSilogLoss(variance_focus=args.variance_focus)
        eval_criterion = NewcrfsSilogLoss(variance_focus=args.variance_focus)
    elif args.loss_func == 'vae':
        from utils import VaeSilogLoss
        silog_criterion = VaeSilogLoss(max_depth=args.max_depth)
        eval_criterion = VaeSilogLoss(max_depth=args.max_depth)
    else:
        raise Exception("Loss function not defined!")
    
    msg_criterion = utils.MSGradientLoss()
    if args.use_msgloss:# is True:
        msg_weight = 0.5
    else:
        msg_weight = 0
    
    oe_criterion = utils.OrdinalEntropyLoss()
    if args.use_ordinalentropy:# is True:
        oe_weight = 1
    else:
        oe_weight = 0
    
    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    logger.info(f"batch size: {num_log_images}")
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    logger.info("== Initial variables' sum: {:.3f}, avg: {:.3f}".format(var_sum, var_sum/var_cnt))

    steps_per_epoch = len(train_dataloader.data_loader)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
    
    tracing_model_save_name = None

    if args.quantize:
        eval_loss_meter = XNNAverageMeter()
    else:
        eval_loss_meter = AverageMeter()
    
    if args.eval_freq == 0:
        args.eval_freq = steps_per_epoch - 1
    logger.info(f"eval frequency: {args.eval_freq}")
    
    # Training loop
    while epoch < args.num_epochs:

        if args.use_early_stopping and early_stopping.early_stop:
            logger.warning("Early stopping triggered. Training stopped!")
            break 
                
        if args.distributed:
            train_dataloader.data_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(tqdm(train_dataloader.data_loader)):
            
            # Zero your gradients for every batch
            optimizer.zero_grad()   
                     
            # if args.use_ema:
            #     ema.update(model)
                
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched["image"].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched["depth"].cuda(args.gpu, non_blocking=True))

            # Runs the forward pass with autocasting
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                
                # Make predictions
                depth_est, feat = model(image, True)

                # Set mask for loss calculation if needed
                if args.dataset.find('nyu') != -1:
                    min_depth = 0.1
                elif args.dataset == 'kitti' or args.dataset == 'kittipred':
                    min_depth = 1.0
                    
                mask = depth_gt > min_depth
                # if args.loss_func == 'vae':
                #     if args.dataset.find('nyu') != -1:
                #         min_depth = 0.1
                #     elif args.dataset == 'kitti' or args.dataset == 'kittipred':
                #         min_depth = 1.0
                #     mask = depth_gt > min_depth

                # Compute the loss. Seperate for logging purpose
                silog_loss = silog_criterion.forward(depth_est, depth_gt, mask)
                msg_loss = msg_criterion.forward(depth_est, depth_gt)
                oe_loss = oe_criterion.forward(feat, depth_gt, min_depth)
                loss = silog_loss + msg_weight * msg_loss + oe_weight * oe_loss
                           
            # Run the backpropagation
            if args.use_amp:
                
                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward()
            
                for param_group in optimizer.param_groups:
                    current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                    param_group['lr'] = current_lr

                # scaler.step() first unscales the gradients of the optimizer's assigned params.
                # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                
                #update the scale for next iteration
                scaler.update()     
            else:
                
                # Calculates the backward gradients over the learning weights
                loss.backward()
            
                for param_group in optimizer.param_groups:
                    current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                    param_group['lr'] = current_lr

                # Adjust learning weights
                optimizer.step()                                        

            if args.use_ema:
                ema.update(model)#, step=global_step)
            
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                logger.info('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    logger.error('NaN in loss occurred. Aborting training.')
                    return -1
            
            duration += time.time() - before_op_time
            if global_step and (global_step % args.log_freq == 0 or step == steps_per_epoch) and not model_just_loaded:
                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    logger.info("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                logger.info(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    
                    # Log training progress
                    if args.use_dvc:
                        train_logger.log_metric('loss/train_loss', loss.item())#, global_step)
                        train_logger.log_metric('loss/silog_loss', silog_loss.item())
                        if args.use_msgloss:
                            train_logger.log_metric('loss/msg_loss', msg_loss.item())
                        if args.use_ordinalentropy:
                            train_logger.log_metric('loss/oe_loss', oe_loss.item())
                        train_logger.log_metric('params/learning_rate', current_lr)
                        train_logger.log_metric('params/var_average', var_sum.item()/var_cnt)
                    else:
                        train_logger.add_scalar('loss/total_loss', loss.item(), global_step)
                        train_logger.add_scalar('loss/silog_loss', silog_loss.item(), global_step)
                        if args.use_msgloss:
                            train_logger.add_scalar('loss/msg_loss', msg_loss.item(), global_step)
                        if args.use_ordinalentropy:
                            train_logger.add_scalar('loss/oe_loss', oe_loss.item(), global_step)
                        train_logger.add_scalar('params/learning_rate', current_lr, global_step)
                        train_logger.add_scalar('params/var_average', var_sum.item()/var_cnt, global_step)
                    
                    depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                    log_image = None
                    # to prevent case actual batch size is smaller than desired batch size
                    for i in range(image.shape[0]):
                        # input_image = inv_image(image[i, :, :, :].data, not args.use_dvc)
                        input_image = utils.inv_image(image[i, :, :, :].data, mean=mean, std=std, scale=(not args.original_scale), to_tensorboard=(not args.use_dvc))
                        pred_image = utils.normalize_result(depth_est[i, :, :, :].data, not args.use_dvc, args.min_depth, args.max_depth)
                        gt_image = utils.normalize_result(depth_gt[i, :, :, :].data, not args.use_dvc, args.min_depth, args.max_depth)

                        if args.use_dvc:
                            batch_image = np.concatenate((input_image, pred_image, gt_image), axis=1)
                            if log_image is None:
                                log_image = np.zeros((0, batch_image.shape[1], 3), dtype=np.uint8)
                            log_image = np.concatenate((log_image, batch_image), axis=0)
                        else:
                            batch_image = torch.concat((input_image.cpu(), pred_image.cpu(), gt_image.cpu()), axis=2) # (C, H, 3W)
                            if log_image is None: # (N, C, H, 3W)
                                log_image = torch.zeros((0, batch_image.shape[0], batch_image.shape[1], batch_image.shape[2]), dtype=torch.uint8)
                            log_image  = torch.cat((log_image, batch_image.unsqueeze(0)), dim=0) # (N+1, C, H, 3W)
                    
                    if args.use_dvc: 
                        train_logger.log_image('{}_{}.jpg'.format(epoch, global_step), log_image)
                    else:
                        train_logger.add_images(f"image", log_image, global_step)
                        
                    if args.use_dvc:
                        train_logger.next_step()
                    else:
                        train_logger.flush()

            if global_step and global_step % args.eval_freq == 0 and not model_just_loaded: #args.do_online_eval and 
                
                time.sleep(0.1)
                
                model.eval()

                if args.use_ema:
                    with torch.no_grad():
                        if args.use_dvc:
                            eval_measures = eval(ema.module, eval_dataloader, gpu, ngpus_per_node, eval_criterion, eval_loss_meter, eval_logger, epoch, global_step, args)#, msg_criterion)
                        else:
                            eval_measures = eval(ema.module, eval_dataloader, gpu, ngpus_per_node, eval_criterion, eval_loss_meter, eval_logger, epoch, global_step, args)
                else:
                    with torch.no_grad():
                        if args.use_dvc:
                            eval_measures = eval(model, eval_dataloader, gpu, ngpus_per_node, eval_criterion, eval_loss_meter, eval_logger, epoch, global_step, args)#, msg_criterion)
                        else:
                            eval_measures = eval(model, eval_dataloader, gpu, ngpus_per_node, eval_criterion, eval_loss_meter, eval_logger, epoch, global_step, args)
                
                if eval_measures is not None:
                    
                    if args.use_ema:
                        model_state_dict = ema.module.state_dict()
                    else:
                        model_state_dict = model.state_dict()

                    if args.quantize:
                        scheduler.step()
                    else:
                        scheduler.step(eval_loss_meter.avg)
                                                
                    if args.use_early_stopping:
                        if early_stopping(eval_loss_meter.avg):
                            logger.info("Early stopping triggered. Training stopped!")
                            break
                        else:
                            logger.info(f"Early stopping counter percentage: {early_stopping.percentage()} %")
                        
                    for i in range(9):
                        
                        # if args.use_dvc:
                        #     eval_logger.log_metric(utils.eval_metrics[i], eval_measures[i].cpu().item())
                        # else:
                        #     eval_logger.add_scalar(f"metric/{utils.eval_metrics[i]}", eval_measures[i].cpu(), global_step)
                        
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_name = 'best_{}_{:.5f}_{}_{}.ckpt'.format(utils.eval_metrics[i], old_best, best_eval_epochs[i], best_eval_steps[i])
                            model_path = args.log_dir + '/' + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            best_eval_epochs[i] = epoch
                            model_save_name = 'best_{}_{:.5f}_{}_{}.ckpt'.format(utils.eval_metrics[i], measure, best_eval_epochs[i], best_eval_steps[i])
                            logger.info('New best for {}. Saving model: {}'.format(utils.eval_metrics[i], model_save_name))

                            checkpoint = {'model': model_state_dict, 'max_depth': args.max_depth, \
                                'feature_list': args.feature_list, 'encoder': args.encoder, 'interpolate': args.interpolate, \
                                'replace_silu': args.replace_silu, 'use_customsilu': args.use_customsilu, \
                                'keepnum_maxpool': args.keepnum_maxpool, 'decoder': args.decoder, 'mid_channel': args.mid_channel, \
                                'use_5_feat': args.use_5_feat, 'quantize': args.quantize, \
                                'optimizer': optimizer.state_dict(), 'global_step': global_step, \
                                'ngpus_per_node': ngpus_per_node, 'scheduler': scheduler.state_dict(), \
                                'normalize': args.normalize, 'to_grayscale': args.to_grayscale, 'original_scale': args.original_scale}# 'optimizer': optimizer.state_dict(), 'global_step': global_step, 'ngpus_per_node': ngpus_per_node }
                            if args.quantize:
                                checkpoint['num_epochs'] = args.num_epochs
                            checkpoint['input_height'] = args.input_height
                            checkpoint['input_width'] = args.input_width
                            torch.save(checkpoint, args.log_dir + '/' + model_save_name)  
                    
                    # Save latest checkpoint for tracing
                    if tracing_model_save_name is not None:
                        command = 'rm {}'.format(args.log_dir + '/' + tracing_model_save_name)
                        os.system(command)
                    tracing_model_save_name = 'last_{}_{}.ckpt'.format(epoch, global_step)
                    logger.info(f'Saving model {tracing_model_save_name} for tracing')
                    checkpoint = {'model': model_state_dict, 'optimizer': optimizer.state_dict(), 'global_step': global_step, 'ngpus_per_node': ngpus_per_node }
                    tracing_model_save_path = args.log_dir + '/' + tracing_model_save_name
                    torch.save(checkpoint, tracing_model_save_path)

                    # if args.use_dvc:
                    #     eval_logger.log_metric(f"{epoch}_{global_step}/loss/avg_loss", eval_loss_meter.avg)
                    #     eval_logger.next_step()   
                    # else:
                    #     eval_logger.add_scalar('loss/avg_loss', eval_loss_meter.avg, global_step)
                    #     eval_logger.flush()       
                    
                model.train()
                
                utils.block_print()
                utils.enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1
       
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        
        # Create report only at the end instead of at each iteration
        if args.use_dvc:
            train_logger.make_summary()   
            train_logger.end()
            
            eval_logger.make_summary()
            eval_logger.end()
        else:
            train_logger.close()
            eval_logger.close()
        
        logger.info("== Training Finished!")
                  
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MetricLeReS PyTorch training.', fromfile_prefix_chars='@')

    parser.add_argument('-c', '--config', help="configuration file *.yml", type=str, required=False, default='')

    parser.add_argument('--comment', help='teamCode-projectName-taskName-modelVersion', default=None, required=False, type=str)
    # parser.add_argument('--mode',                      type=str,   help='train or test', default="train")

    # Seed value
    parser.add_argument('--seed', type=int, help='Seed value', default=42)

    # Model architecture setup
    parser.add_argument('--model_name',                type=str,   help='model name', default='pixelformer')
    parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
    parser.add_argument('--decoder',                   type=str,   help='type of decoder', default='metricleres')
    parser.add_argument('-l', '--feature_list',                    help='Feature list (default is [])',                              default = [], type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument('--keepnum_maxpool',                       help='Keep number of maxpool layers for each SPP maxpool block (YOLO encoders)', default = [False, False, True], type=lambda s: [bool(item) for item in s.split(',')])
    parser.add_argument('--replace_silu'                       ,   help='if set, replace every SILU with RELU',                      action='store_true')
    parser.add_argument('--use_customsilu',                        help="if set, use custom SILU (when replace_silu is True)",       action='store_true')
    parser.add_argument('--interpolate',                           help='if set, replace every nn.Upsample with custom Interpolate', action='store_true')
    parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth',                 type=float, help='minimum depth in estimation & evaluation', default=1e-3)
    parser.add_argument('--use_5_feat',                            help="if set, use 5 features",       action='store_true')
    parser.add_argument('--mid_channel',               type=int,   help='# of intermediate channels for decoder', default=0)

    # Load checkpoint/pretrained backbone
    parser.add_argument('--weight_path',           type=str,   help='path to a checkpoint to load', default='')
    parser.add_argument('--retrain',                               help='if used with weight_path, will restart training from step zero', action='store_true')
    parser.add_argument('--pretrain_path',             type=str,   help='path of pretrained encoder',                                default=None)
    parser.add_argument('--freeze_encoder',                        help='if set, freeze the encoder',                                action='store_true')
    parser.add_argument('--init_type',                 type=str,   help='model initialization type (default is normal)',             default='normal')
    parser.add_argument('--pretrained',                            help='if set, use the pretrained encoder',                        action='store_true')

    # Dataset
    parser.add_argument('--use_cache',                             help="if set, use dataloader cache (train/eval)",       action='store_true')
    parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
    parser.add_argument('--normalize',                             help='if set, use ImageNet normalization',                        action='store_true')
    parser.add_argument('--to_grayscale',                          help='if set, convert RGB to grayscale',                          action='store_true')
    parser.add_argument('--original_scale',                        help='if set, use the normal image input value range (0->255) instead of (0->1)',                          action='store_true')

    # Load training data
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=False)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=False)

    # Log and save
    parser.add_argument('--use_dvc',                               help='use DVC instead of TensorboardX', action='store_true')
    parser.add_argument('--log_dir',                   type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
    parser.add_argument('--save_files',                        help='if set, save the current code into checkpoint directory', action='store_true')

    # Training hyperparams
    parser.add_argument('--input_height',              type=int,   help='input height', default=480)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
    parser.add_argument('--use_amp',                               help='if set, use automatic mixed precision training',           action='store_true')
    parser.add_argument('--quantize',                              help='if set, use quantization-aware-training',                  action='store_true')
    parser.add_argument('--loss_func',                 type=str,   help='Loss function options (newcrfs, zoe, vae)',                default='zoe')
    parser.add_argument('--use_ema',                               help='if set, use EMA',                                          action='store_true')
    parser.add_argument('--model_ema_decay',           type=float, help='EMA model weight decay', default=0.9999)
    parser.add_argument('--model_ema_warmup',          help='if set, use warm up for EMA',                                          action='store_true')

    parser.add_argument('--optimizer',                 type=str,   help='Optimizer options (adam, adamw, adagrad, rmsprop, sgd, adopt)',                   default='adam')
    parser.add_argument('--default_decay',                         help='if set, use default decay values (weight_decay & adam_eps)',                  action='store_true')
    parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
    parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
    parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
    parser.add_argument('--learning_rate_ratio',       type=float, help='learning rate ratio between backbone and others', default=0.4)
    parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
    parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
    parser.add_argument('--use_msgloss',                           help='if set, use multi-scale gradient loss ',                                          action='store_true')
    parser.add_argument('--use_ordinalentropy',                    help='if set, use Ordinal Entropy loss ',                                          action='store_true')

    # Augmentation hyperparams (also for training)
    parser.add_argument('--flip_prob_thres',                 type=float, help='horizontal flip augmentation prob', default=0.5)
    
    parser.add_argument('--cut_flip_horizontal_prob_thres',  type=float, help='horizontal cut flip augmentation prob', default=1)
    parser.add_argument('--cut_flip_vertical_prob_thres',    type=float, help='vertical cut flip augmentation prob', default=1)
    
    parser.add_argument('--image_transform_prob_thres',      type=float, help='image transform augmentation prob', default=0.5)
    
    parser.add_argument('--cut_depth_prob_thres',      type=float, help='cut-depth augmentation prob', default=0.5)
    parser.add_argument('--cut_depth_thres',                 type=float, help='cut-depth augmentation size thres', default=0.75)
    
    parser.add_argument('--cut_mix_prob_thres',      type=float, help='cut-mix augmentation prob', default=0.5)
    parser.add_argument('--cut_mix_thres',              type=float, help='cut-mix augmentation size thres', default=0.75)
    
    parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
    parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')
    parser.add_argument('--use_early_stopping',                    help='if set, use early stopping', action='store_true')
    parser.add_argument('--lr_patience',               type=int,   help='learning rate patience threshold', default=5)
    parser.add_argument('--use_albumentations',                    help='if set, will use albumentations for augmentation', action='store_true')
    # parser.add_argument('--use_cutdepth',                            help='if set, will use cutdepth for augmentation', action='store_true')
    # parser.add_argument('--use_cutmix',                            help='if set, will use cutmix for augmentation', action='store_true')
    parser.add_argument('--shuffle',                               help='if set, shuffle the training samples', action='store_true')
    # augmentation range
    parser.add_argument('--min_gamma',                    type=float, help='minimum gamma range for gamma augmentation', default=0.9)
    parser.add_argument('--max_gamma',                    type=float, help='maximum gamma range for gamma augmentation', default=1.1)
    parser.add_argument('--min_brightness',               type=float, help='minimum brightness range for brightness augmentation', default=0.9)#for nyu recommend to use 0.75
    parser.add_argument('--max_brightness',               type=float, help='maximum brightness range for brightness augmentation', default=1.1)#for nyu recommend to use 1.25
    parser.add_argument('--min_color',                    type=float, help='minimum color range for color augmentation', default=0.9)
    parser.add_argument('--max_color',                    type=float, help='maximum color range for color augmentation', default=1.1)

    # Evaluation
    parser.add_argument('--data_path_eval',            type=str,   help='path to the evaluation data', default='')
    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)
    parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=0)

    # Multi-gpu training
    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
    
    # The total number of processes, so that the master knows how many workers to wait for.
    parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
    
    #  Rank of each process, so they will know whether it is the master or a worker.
    parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
    parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
    parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
    parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
    parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                        'N processes per node, which has N GPUs. This is the '
                                                                        'fastest way to use PyTorch for either single node or '
                                                                        'multi node data parallel training', action='store_true',)
   
    if sys.argv.__len__() == 2:
        parser.convert_arg_line_to_args = utils.convert_arg_line_to_args
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
    # Set seeds
    utils.set_seeds(args.seed)

    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    args.log_dir = os.path.join(args.log_dir, args.model_name, time_stamp)
    logger.info(f"Training Start! Saving logs into: {args.log_dir}")
    os.makedirs(args.log_dir, exist_ok=True)

    if args.save_files:
        # aux_out_path = os.path.join(args.log_dir, args.model_name)
        networks_savepath = os.path.join(args.log_dir, 'networks')
        dataloaders_savepath = os.path.join(args.log_dir, 'dataloaders')
        command = 'cp train.py ' + args.log_dir
        os.system(command)
        command = 'mkdir -p ' + networks_savepath + ' && cp networks/*.py ' + networks_savepath
        os.system(command)
        command = 'mkdir -p ' + dataloaders_savepath + ' && cp dataloaders/*.py ' + dataloaders_savepath
        os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    logger.info(f"ngpus_per_node: {ngpus_per_node}")
    # args.batch_size = ngpus_per_node * args.batch_size# 4
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        logger.error("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        exit(-1)
        # return -1

    logger.info("Model will be evaluated every eval_freq {} steps and save best models for individual eval metrics."
            .format(args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        train(args.gpu, ngpus_per_node, args)




# def main(args):
#     # if args.mode != "train":
#     #     logger.error('train.py is only for training.')
#     #     return -1
#     time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
#     args.log_dir = os.path.join(args.log_dir, args.model_name, time_stamp)
#     logger.info(f"Training Start! Saving logs into: {args.log_dir}")
#     os.makedirs(args.log_dir, exist_ok=True)

#     if args.save_files:
#         # aux_out_path = os.path.join(args.log_dir, args.model_name)
#         networks_savepath = os.path.join(args.log_dir, 'networks')
#         dataloaders_savepath = os.path.join(args.log_dir, 'dataloaders')
#         command = 'cp train.py ' + args.log_dir
#         os.system(command)
#         command = 'mkdir -p ' + networks_savepath + ' && cp networks/*.py ' + networks_savepath
#         os.system(command)
#         command = 'mkdir -p ' + dataloaders_savepath + ' && cp dataloaders/*.py ' + dataloaders_savepath
#         os.system(command)

#     torch.cuda.empty_cache()
#     args.distributed = args.world_size > 1 or args.multiprocessing_distributed

#     ngpus_per_node = torch.cuda.device_count()
#     logger.info(f"ngpus_per_node: {ngpus_per_node}")
#     # args.batch_size = ngpus_per_node * args.batch_size# 4
#     if ngpus_per_node > 1 and not args.multiprocessing_distributed:
#         logger.error("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
#         return -1

#     logger.info("Model will be evaluated every eval_freq {} steps and save best models for individual eval metrics."
#             .format(args.eval_freq))

#     if args.multiprocessing_distributed:
#         args.world_size = ngpus_per_node * args.world_size
#         mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
#     else:
#         train(args.gpu, ngpus_per_node, args)
        
    # main(args)


    # val_dir_path = "{}{}/{}_{}".format(args.log_dir, args.model_name, epoch, global_step)
    # try:
    #     os.mkdir(val_dir_path)
    # except OSError:
    #     pass   

    # time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    # train_config_path = os.path.join(args.log_dir, args.model_name, f"train_{time_stamp}.yml")
    # with open(train_config_path, 'w+') as f:
    #     yaml.dump(args.__dict__, f)

            # if scaler is not None:
            #     with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    
            #         # if args.loss_func == 'vae':
            #         # if args.dataset.find('nyu') != -1:
            #         #     min_depth = 0.1
            #         # elif args.dataset == 'kitti' or args.dataset == 'kittipred':
            #         #     min_depth = 1.0
            #         # mask = depth_gt > min_depth
                    
            #         loss = silog_criterion.forward(depth_est, depth_gt, mask)
            #         if args.use_msgloss:
            #             loss += 0.5 * msg_criterion.forward(depth_est, depth_gt)
            #         if args.use_ordinalentropy:
            #             loss += oe_criterion.forward(feat, depth_gt, min_depth)
                            
            #         # else:
            #         #     if args.dataset.find('nyu') != -1:                
            #         #         if args.dataset == 'nyu_crop_1000':
            #         #             depth_est = depth_est.int()
            #         #         mask = depth_gt > 0.1
            #         #     elif args.dataset == 'kitti' or args.dataset == 'kittipred':
            #         #         mask = depth_gt > 1.0
            #         #     elif args.dataset == 'void' or args.dataset == 'nyu_short' or args.dataset.find('d435f') != -1:
            #         #         mask = torch.logical_and(depth_gt >= args.min_depth, depth_gt <= args.max_depth) 
            #         #     else:
            #         #         mask = torch.logical_and(depth_gt > args.min_depth, depth_gt < args.max_depth) 
                        
            #         #     loss = silog_criterion.forward(depth_est, depth_gt, mask)
            #         #     if args.use_msgloss:
            #         #         loss += 0.5 * msg_criterion.forward(depth_est, depth_gt)
            #         #     if args.use_ordinalentropy:
            #         #         loss += oe_criterion.forward(feat, depth_gt, min_depth)
                    
            #         # Backpropagation
                    
            #         # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            #         # Backward passes under autocast are not recommended.
            #         # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            #         scaler.scale(loss).backward()
                
            #         for param_group in optimizer.param_groups:
            #             current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
            #             param_group['lr'] = current_lr

            #         # scaler.step() first unscales the gradients of the optimizer's assigned params.
            #         # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            #         # otherwise, optimizer.step() is skipped.
            #         scaler.step(optimizer)
                    
            #         #update the scale for next iteration
            #         scaler.update()
            # else:
            #     with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            #         # if args.loss_func == 'vae':
            #         #     if args.dataset.find('nyu') != -1:
            #         #         min_depth = 0.1
            #         #     elif args.dataset == 'kitti' or args.dataset == 'kittipred':
            #         #         min_depth = 1.0
            #         #     mask = depth_gt > min_depth

            #         loss = silog_criterion.forward(depth_est, depth_gt, mask)
            #         if args.use_msgloss:
            #             loss += 0.5 * msg_criterion.forward(depth_est, depth_gt)
            #         if args.use_ordinalentropy:
            #             loss += oe_criterion.forward(feat, depth_gt, min_depth)

            #         # else:
            #         #     if args.dataset.find('nyu') != -1:                
            #         #         if args.dataset == 'nyu_crop_1000':
            #         #             depth_est = depth_est.int()
            #         #         mask = depth_gt > 0.1
            #         #     elif args.dataset == 'kitti' or args.dataset == 'kittipred':
            #         #         mask = depth_gt > 1.0
            #         #     elif args.dataset == 'void' or args.dataset == 'nyu_short' or args.dataset.find('d435f') != -1:
            #         #         mask = torch.logical_and(depth_gt >= args.min_depth, depth_gt <= args.max_depth) 
            #         #     else:
            #         #         mask = torch.logical_and(depth_gt > args.min_depth, depth_gt < args.max_depth) 
            
            #         #     loss = silog_criterion.forward(depth_est, depth_gt, mask)
            #         #     if args.use_msgloss:
            #         #         loss += 0.5 * msg_criterion.forward(depth_est, depth_gt)
            #         #     if args.use_ordinalentropy:
            #         #         loss += oe_criterion.forward(feat, depth_gt, min_depth)

            #         loss.backward()
                
            #         for param_group in optimizer.param_groups:
            #             current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
            #             param_group['lr'] = current_lr

            #         optimizer.step()     
            
                # else:
                #     with torch.no_grad():
                #         eval_measures = eval(model, eval_dataloader, gpu, ngpus_per_node, eval_criterion, eval_loss_meter, eval_logger, epoch, global_step)#, msg_criterion)
                #     if eval_measures is not None:
                        
                #         if args.quantize:
                #             scheduler.step()
                #         else:
                #             scheduler.step(eval_loss_meter.avg)
                        
                #         if args.use_early_stopping:
                #             early_stopping(eval_loss_meter.avg)
                #             if early_stopping.early_stop:
                #                 logger.info("Early stopping!")
                #                 break 
                #             else:
                #                 logger.info(f"Early stopping counter percentage: {early_stopping.percentage()} %")

                            
                #         for i in range(9):
                #             # eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                #             eval_logger.log_metric(eval_metrics[i], eval_measures[i].cpu().item())
                #             measure = eval_measures[i]
                #             is_best = False
                #             if i < 6 and measure < best_eval_measures_lower_better[i]:
                #                 old_best = best_eval_measures_lower_better[i].item()
                #                 best_eval_measures_lower_better[i] = measure.item()
                #                 is_best = True
                #             elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                #                 old_best = best_eval_measures_higher_better[i-6].item()
                #                 best_eval_measures_higher_better[i-6] = measure.item()
                #                 is_best = True
                #             if is_best:
                #                 old_best_name = 'best_{}_{:.5f}_{}_{}.ckpt'.format(eval_metrics[i], old_best, best_eval_epochs[i], best_eval_steps[i])
                #                 model_path = args.log_dir + '/' + old_best_name
                #                 if os.path.exists(model_path):
                #                     command = 'rm {}'.format(model_path)
                #                     os.system(command)
                #                 best_eval_steps[i] = global_step
                #                 model_save_name = 'best_{}_{:.5f}_{}_{}.ckpt'.format(eval_metrics[i], measure, best_eval_epochs[i], best_eval_steps[i])
                #                 logger.info('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                #                 checkpoint = {'model': model.state_dict(), 'max_depth': args.max_depth, 'feature_list': args.feature_list, \
                #                 'encoder': args.encoder, 'interpolate': args.interpolate, 'replace_silu': args.replace_silu, \
                #                 'use_customsilu': args.use_customsilu, 'keepnum_maxpool': args.keepnum_maxpool, 'mid_channel': args.mid_channel,\
                #                 'decoder': args.decoder, 'use_5_feat': args.use_5_feat, 'quantize': args.quantize, \
                #                 'optimizer': optimizer.state_dict(), 'global_step': global_step, 'ngpus_per_node': ngpus_per_node, \
                #                 'scheduler': scheduler.state_dict(), 'normalize': args.normalize, 'to_grayscale': args.to_grayscale, 'original_scale': args.original_scale}# 'optimizer': optimizer.state_dict(), 'global_step': global_step, 'ngpus_per_node': ngpus_per_node }
                #                 if args.quantize:
                #                     checkpoint['num_epochs'] = args.num_epochs
                #                 checkpoint['input_height'] = args.input_height
                #                 checkpoint['input_width'] = args.input_width
                #                 torch.save(checkpoint, args.log_dir + '/' + model_save_name)
                        
                #         # Save checkpoint for tracing
                #         if tracing_model_save_name is not None:
                #             command = 'rm {}'.format(args.log_dir + '/' + tracing_model_save_name)
                #             os.system(command)
                #         tracing_model_save_name = 'last_{}_{}.ckpt'.format(epoch, global_step)
                #         logger.info(f'Saving model {tracing_model_save_name} for tracing')
                #         checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'global_step': global_step, 'ngpus_per_node': ngpus_per_node }
                #         torch.save(checkpoint, args.log_dir + '/' + tracing_model_save_name)
                        
                        # eval_summary_writer.flush()
                        # eval_logger.next_step()
