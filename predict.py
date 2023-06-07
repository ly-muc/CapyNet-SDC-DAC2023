import mmcv
import torch
from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, show_result_pyplot

from mmdet.apis import train_detector
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

if __name__ == '__main__':

    # change config path
    config = '/home/linyan/code/gpu_starter_2023/configs/runs/faster_rcnn.py'
    config = mmcv.Config.fromfile(config)

    # modify num classes of the model in box head
    config.model.roi_head.bbox_head.num_classes = 7

    # Set seed thus the results are more reproducible
    seed = 0
    device = 'cuda'
    gpu_ids = range(1)

    # takes a while
    datasets = [build_dataset(config.data.train)]

    # build model
    model = build_detector(config.model)

    # load checkpoint
    checkpoint = load_checkpoint(model, config.load_from, map_location=device)

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    model.cfg = config

    idx = 43
    # Use the detector to do inference
    img = f'/home/linyan/data/dac/train/JPEGImages/{idx:05}.jpg'
    result = inference_detector(model, img)
    # Let's plot the result
    show_result_pyplot(model, img, result, score_thr=0.7)