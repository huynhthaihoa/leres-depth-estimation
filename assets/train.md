# TRAINING GUIDELINE

## Steps
1. Select the config file:
- **NYUv2 dataset**: use [train_nyu.yml](../configs/train_nyu.yml) 
<!-- #to train on a single GPU, or [train_nyu_multigpus.txt](configs/train_nyu_multigpus.txt) to train on multiple GPUs -->
- **KITTI dataset**: use [train_kitti.yml](../configs/train_kitti.yml) 
<!-- to train on a single GPU, or [train_kitti_multigpus.txt](configs/train_kitti_multigpus.txt) to train on multiple GPUs -->
2. (Optional) Update some hyperparameters in the config file:
    - **model_name**: name of the model
    - **encoder**: encoder type (please choose from the [suppoted encoders](encoders_decoders.md#1-supported-encoders))
    - **decoder**: decoder type (please choose from the [suppoted decoders](encoders_decoders.md#2-supported-decoders))
    - **batch_size**: batch size (on each GPU)
    - **data_path**: the root directory of the dataset (both KITTI and NYUv2 datasets can be found on Jupiter server)
    - **num_epochs**: number of epochs
    - **learning_rate**: initial learning rate (Adam optimizer, one tip is to use smaller learning rate for bigger encoder)
    - **init_type**: model parameter initialization type, choose between `'normal'`, `'xavier'`, `'kaiming'`, `'lecun'`, `'uniform'`, `'orthogonal'`, `'constant'`, `'he'`, `'kaiming_uniform'`, `'sparse'`, and `'identity'`
    - **log_dir**: a directory named `[log_dir]`/`[model_name]`/`[time_stamp]` will be created automatically (`[time_stamp]` is the time one executes the training script) to save all the training artifacts (logs, checkpoints, etc.)
    - **normalize**: if one wants to apply [ImageNet normalization](https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2), set it as `True`, otherwise set as `False`
    - **pretrained**: if one wants to use the pretrained encoder, set it as `True`, otherwise set as `False`

> Note: besides the parameters above, please refer to `train.py` to see the full list of supported parameters

3. Run the following cli to start training:

    ```
        CUDA_VISIBLE_DEVICES=[] python train.py -c configs/[config_file_name] --comment [comment]
    ```

    example 1 (training on NYUv2 dataset):
    
    ```
    CUDA_VISIBLE_DEVICES=0 cpulimit  -c 16 -l 1600 python train.py -c configs/train_nyu.yml --comment traindepthmodel
    ```

    example 2 (training on KITTI dataset):

    ```
    CUDA_VISIBLE_DEVICES=0 cpulimit  -c 16 -l 1600 python train.py -c configs/train_kitti.yml --comment traindepthmodel
    ```
## Training monitoring

As default, we support Tensorboard to log the training and evaluation progress:
    
- To track the training progress, run the above cli:

    ```
    tensorboard --logdir [log_dir]/[model_name]/[timestamp]/train --port [port number]
    ```

    example:

    ```
    tensorboard --logdir /hdd/hoa/models/pytorch/depth/nyu_yolov8s_metricleres/2025_05_14_15_57_20/train --port 3003
    ```

- To track the evaluation progress, run the above cli:

    ```
    tensorboard --logdir [log_dir]/[model_name]/[timestamp]/eval --port [port number]
    ```

    example:

    ```
    tensorboard --logdir /hdd/hoa/models/pytorch/depth/nyu_yolov8s_metricleres/2025_05_14_15_57_20/eval --port 3003
    ```

## Useful references
- [UvA DL Notebooks Tutorial 4: Optimization and Initialization](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial4/Optimization_and_Initialization.html)
- [Google Research Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
