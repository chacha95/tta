# -*- coding: utf-8 -*-
from detectron2.data.datasets import register_coco_instances


def regist_custom_dataset(dataset_dir: str):
    json_file = f"{dataset_dir}/annotations/instances_val2017.json"
    image_root = f"{dataset_dir}/val2017"
    # BoxMode.XYXY_ABS -> 0
    # BoxMode.XYWH_ABS -> 1
    register_coco_instances("coco2017val", {"bbox_mode": 1}, json_file, image_root)
