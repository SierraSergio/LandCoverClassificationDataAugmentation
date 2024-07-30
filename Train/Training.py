import torch
print(torch.__version__, torch.cuda.is_available())
import mmseg
print(mmseg.__version__)
from mmengine import Config

data_root = ''
img_dir = 'imgs'
ann_dir = 'masks'

cfg = Config.fromfile('./checkpoints/deeplabv3plus_r101-d8_4xb2-80k_cityscapes-512x1024.py')
print(f'Config:\n{cfg.pretty_text}')

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg

# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 5
cfg.model.auxiliary_head.num_classes = 5

# Modify dataset type and path
cfg.dataset_type = ''
cfg.data_root = data_root

cfg.train_dataloader.batch_size = 8

cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(320, 240), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(320, 240), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
cfg.train_dataloader.dataset.ann_file = 'splits/train.txt'

cfg.val_dataloader.dataset.type = cfg.dataset_type
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix = dict(img_path=img_dir, seg_map_path=ann_dir)
cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
cfg.val_dataloader.dataset.ann_file = 'splits/val.txt'

cfg.test_dataloader = cfg.val_dataloader

# Load the pretrained weights
cfg.load_from = './checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/deeplabv3+'

cfg.train_cfg.max_iters = 5000
cfg.train_cfg.val_interval = 1000
cfg.default_hooks.logger.interval = 100
cfg.default_hooks.checkpoint.interval = 1000

# Set seed to facilitate reproducing the result
cfg['randomness'] = dict(seed=0)

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

from mmengine.runner import Runner
# Add this import at the top of your script
import multiprocessing

if __name__ == '__main__':
    # Add this line to ensure compatibility with multiprocessing on Windows
    multiprocessing.freeze_support()
    runner = Runner.from_cfg(cfg)
    # start training
    runner.train()
