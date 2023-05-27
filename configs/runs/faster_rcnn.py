_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/dac_contest_base.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


load_from = '/home/linyan/code/gpu_starter_2023/checkpoints/epoch_12.pth'
gpu_ids = range(1)
seed = 0
device = 'cuda'
# Set up working dir to save files and logs.
work_dir = './tutorial_exps'