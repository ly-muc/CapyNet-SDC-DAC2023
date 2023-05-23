# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from mmdet.core import eval_map, eval_recalls
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

from .builder import DATASETS
from .xml_style import XMLDataset


def get_meta(index: int,
             label_formater: Optional[Callable]=None)-> Tuple[List[str], List[int]]:
    """Return value of labels depends on formatter"""
    # read annotation file
    soln_root = f'/home/linyan/data/dac/train/Annotations/bbox/{index:05}.xml'
    soln_root = ET.parse(soln_root).getroot()

    objects = []
    labels = []
    
    # find all objects in frame
    for obj in soln_root.findall('object'):
    
        obj_class = obj.find('name').text
        x_min = int(obj.find('bndbox/xmin').text)
        y_min = int(obj.find('bndbox/ymin').text)
        x_max = int(obj.find('bndbox/xmax').text)
        y_max = int(obj.find('bndbox/ymax').text)
        
        if label_formater:
            labels.append(label_formater(obj_class))
        else:
            labels.append(obj_class)
            
        objects.append(list((x_min, y_min, x_max, y_max)))
    
    objects = torch.tensor(objects, dtype=torch.int)

    return labels, objects

@DATASETS.register_module()
class DACDataset(CustomDataset):

    CLASSES = ('Motor Vehicle', 'Non-motorized Vehicle', 'Pedestrian', 
               'Traffic Light-Red Light', 'Traffic Light-Yellow Light',
               'Traffic Light-Green Light', 'Traffic Light-Off')

    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file
        image_list = mmcv.list_from_file(self.ann_file)
    
        data_infos = []
        # convert annotations to middle format
        for image_id in image_list:
            filename = f'{self.img_prefix}/{image_id}.jpg'
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=f'{image_id}.jpg', width=width, height=height)
    
            # load annotations    
            labels, bboxes = get_meta(int(image_id), cat2label.get)
            bboxes = np.array(bboxes, float)
            labels = np.array(labels, np.long)

            data_anno = dict(
                bboxes=bboxes,
                labels=labels)

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos