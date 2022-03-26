from typing import List
import copy
import numpy as np
from contextlib import contextmanager
from itertools import count
import torch
from fvcore.transforms import HFlipTransform, NoOpTransform
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from detectron2.config import configurable
from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import (
    RandomFlip,
    ResizeShortestEdge,
    ResizeTransform,
    RandomContrast,
    apply_augmentations,
)
from detectron2.structures import Boxes, Instances


__all__ = ["DatasetMapperTTA", "GeneralizedRCNNWithTTA"]


# ----------------------------------------Changed code------------------------------------------
class TTA(object):
    """
    Set TTA options.
    _flip: do horizontal flip?
    _multi_scale: Takes target size as input and randomly scales the given target size(Scale the shorter edge).
    _contrast: Contrast intensity is uniformly sampled in (intensity_min, intensity_max).
               intensity < 1 will reduce contrast
               intensity = 1 will preserve the input image
               intensity > 1 will increase contrast

    Examples:
        flip only:
            _flip: bool = True
            _multi_scale: List[int] = []
            _contrast: List[float] = []
        multi scale only:
            _flip: bool = False
            _multi_scale: List[int] = [400, 600, 800, 1000]
            _contrast: List[float] = []
        contrast only:
            _flip: bool = False
            _multi_scale: List[int] = []
            _contrast: List[float] = [0.95, 1.05]
    """
    _flip: bool = True
    _multi_scale: List[int] = [400, 600, 800, 1000]
    _contrast: List[float] = []

    @classmethod
    def get_multi_scale(cls):
        return cls._multi_scale

    @classmethod
    def get_flip(cls):
        return cls._flip

    @classmethod
    def get_contrast(cls):
        return cls._contrast
# ----------------------------------------Changed code------------------------------------------


class DatasetMapperTTA:
    @configurable
    def __init__(self, **kwargs):
        self.min_sizes = TTA.get_multi_scale()
        self.contrast = TTA.get_contrast()
        self.flip = TTA.get_flip()
        self.max_size = 4000

    @classmethod
    def from_config(cls, cfg):
        return {
            "min_sizes": cfg.TEST.AUG.MIN_SIZES,
            "max_size": cfg.TEST.AUG.MAX_SIZE,
            "flip": cfg.TEST.AUG.FLIP,
        }

    def __call__(self, dataset_dict):
        """
        Args:
            dict: a dict in standard model input format. See tutorials for details.
        Returns:
            list[dict]:
                a list of dicts, which contain augmented version of the input image.
                The total number of dicts is ``len(min_sizes) * (2 if flip else 1)``.
                Each dict has field "transforms" which is a TransformList,
                containing the transforms that are used to generate this image.
        """
        numpy_image = dataset_dict["image"].permute(1, 2, 0).numpy()
        shape = numpy_image.shape
        orig_shape = (dataset_dict["height"], dataset_dict["width"])
        if shape[:2] != orig_shape:
            # It transforms the "original" image in the dataset to the input image
            pre_tfm = ResizeTransform(orig_shape[0], orig_shape[1], shape[0], shape[1])
        else:
            pre_tfm = NoOpTransform()
# ----------------------------------------Changed code------------------------------------------
        aug_candidates = []
        if self.flip:
            flip = RandomFlip(prob=1.0, horizontal=True, vertical=False)
        if self.contrast:
            contrast = RandomContrast(self.contrast[0], self.contrast[1])

        # use multi scale
        if self.min_sizes:
            for min_size in self.min_sizes:
                resize = ResizeShortestEdge(min_size, self.max_size)
                aug_candidates.append([resize])
                # use multi scale + flip
                if self.flip:
                    aug_candidates.append([resize, flip])
                # use multi scale + contrast
                if self.contrast:
                    aug_candidates.append([resize, contrast])
                # use multi scale + flip + contrast
                if self.flip and self.contrast:
                    aug_candidates.append([resize, flip, contrast])

        if self.flip:
            aug_candidates.append([flip])

        if self.contrast:
            aug_candidates.append([contrast])
# ----------------------------------------Changed code------------------------------------------
        # Apply all the augmentations
        ret = []
        for aug in aug_candidates:
            new_image, tfms = apply_augmentations(aug, np.copy(numpy_image))
            torch_image = torch.from_numpy(np.ascontiguousarray(new_image.transpose(2, 0, 1)))
            dic = copy.deepcopy(dataset_dict)
            dic["transforms"] = pre_tfm + tfms
            dic["image"] = torch_image
            ret.append(dic)
        return ret


class GeneralizedRCNNWithTTA(nn.Module):
    """
    A GeneralizedRCNN with test-time augmentation enabled.
    Its :meth:`__call__` method has the same interface as :meth:`GeneralizedRCNN.forward`.
    """

    def __init__(self, cfg, model, tta_mapper=None, batch_size=3):
        """
        Args:
            cfg (CfgNode):
            model (GeneralizedRCNN): a GeneralizedRCNN to apply TTA on.
            tta_mapper (callable): takes a dataset dict and returns a list of
                augmented versions of the dataset dict. Defaults to
                `DatasetMapperTTA(cfg)`.
            batch_size (int): batch the augmented images into this batch size for inference.
        """
        super().__init__()
        if isinstance(model, DistributedDataParallel):
            model = model.module
        self.cfg = cfg.clone()
        self.model = model
        self.tta_mapper = DatasetMapperTTA(cfg)
        self.batch_size = batch_size

    def _batch_inference(self, batched_inputs, detected_instances=None):
        """
        Execute inference on a list of inputs,
        using batch size = self.batch_size, instead of the length of the list.

        Inputs & outputs have the same format as :meth:`GeneralizedRCNN.inference`
        """
        if detected_instances is None:
            detected_instances = [None] * len(batched_inputs)

        outputs = []
        inputs, instances = [], []
        for idx, input, instance in zip(count(), batched_inputs, detected_instances):
            inputs.append(input)
            instances.append(instance)
            if len(inputs) == self.batch_size or idx == len(batched_inputs) - 1:
                outputs.extend(
                    self.model.inference(
                        inputs,
                        instances if instances[0] is not None else None,
                        do_postprocess=False,
                    )
                )
                inputs, instances = [], []
        return outputs

    def __call__(self, batched_inputs):
        """
        Same input/output format as :meth:`GeneralizedRCNN.forward`
        """

        def _maybe_read_image(dataset_dict):
            ret = copy.copy(dataset_dict)
            if "image" not in ret:
                image = read_image(ret.pop("file_name"), self.model.input_format)
                image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))  # CHW
                ret["image"] = image
            if "height" not in ret and "width" not in ret:
                ret["height"] = image.shape[1]
                ret["width"] = image.shape[2]
            return ret

        return [self._inference_one_image(_maybe_read_image(x)) for x in batched_inputs]

    def _inference_one_image(self, input):
        """
        Args:
            input (dict): one dataset dict with "image" field being a CHW tensor

        Returns:
            dict: one output dict
        """
        orig_shape = (input["height"], input["width"])
        augmented_inputs, tfms = self._get_augmented_inputs(input)
        # Detect boxes from all augmented versions
        all_boxes, all_scores, all_classes = self._get_augmented_boxes(augmented_inputs, tfms)
        # merge all detected boxes to obtain final predictions for boxes
        merged_instances = self._merge_detections(all_boxes, all_scores, all_classes, orig_shape)
        return {"instances": merged_instances}

    def _get_augmented_inputs(self, input):
        augmented_inputs = self.tta_mapper(input)
        tfms = [x.pop("transforms") for x in augmented_inputs]
        return augmented_inputs, tfms

    def _get_augmented_boxes(self, augmented_inputs, tfms):
        """
        This function inverse the transforms on bbox.
        So you can get bbox from augmentation images and inverse to original image's bbox
        """
        # 1: forward with all augmented images
        outputs = self._batch_inference(augmented_inputs)
        # 2: union the results
        all_boxes = []
        all_scores = []
        all_classes = []
        for output, tfm in zip(outputs, tfms):
            pred_boxes = output.pred_boxes.tensor
            original_pred_boxes = tfm.inverse().apply_box(pred_boxes.cpu().numpy())
            all_boxes.append(torch.from_numpy(original_pred_boxes).to(pred_boxes.device))

            all_scores.extend(output.scores)
            all_classes.extend(output.pred_classes)
        all_boxes = torch.cat(all_boxes, dim=0)
        return all_boxes, all_scores, all_classes

    def _merge_detections(self, all_boxes, all_scores, all_classes, shape_hw):
        """
        This function aggregate all bbox from augmentation and original image.
        And just put in `fast_rcnn_inference_single_image` function for final detection.
        `fast_rcnn_inference_single_image` function filter results based on detection scores and NMS(Non-Maximum suppression) each bbox.
        => they have score threshold and nms(bbox) threshold.
        """
        num_boxes = len(all_boxes)
        num_classes = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # +1 because fast_rcnn_inference expects background scores as well
        all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
        # formatting
        for idx, cls, score in zip(count(), all_classes, all_scores):
            all_scores_2d[idx, cls] = score

        from .roi_heads.fast_rcnn import fast_rcnn_inference_single_image
        merged_instances, _ = fast_rcnn_inference_single_image(
            all_boxes,
            all_scores_2d,
            shape_hw,
            1e-8,
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            self.cfg.TEST.DETECTIONS_PER_IMAGE,
        )
        return merged_instances
