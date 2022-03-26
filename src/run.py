from typing import List
import subprocess
import argparse
import sys
import os

root_dir = os.getcwd()
sys.path.insert(1, root_dir)
from utils.util import create_random_port


class Detectron2Header(object):
    @classmethod
    def test(cls, args):
        args = args
        dist_url = create_random_port()
        command = cls.tta(args)
        subprocess.Popen(command)
        command = cls.generate_test_command(args, dist_url)
        subprocess.Popen(command)

    @classmethod
    def tta(cls, args) -> List[str]:
        script = f'{args.root_dir}/script/change_func.sh'
        command = ["/bin/bash", script]
        return command

    @classmethod
    def generate_test_command(cls, args, dist_url: str) -> List[str]:
        dt_src = f'{args.root_dir}/core/train_net.py'
        command = ["python3", dt_src,
                   "--num-gpus", args.num_gpu,
                   "--config-file", args.model_config,
                   "--dist-url", dist_url,
                   "--eval-only", "MODEL.WEIGHTS", args.model_weights,
                   "OUTPUT_DIR", args.output_dir,
                   "DATASETS.TEST", args.test_dataset,
                   "DATALOADER.NUM_WORKERS", args.num_worker,
                   "MODEL.ROI_HEADS.NUM_CLASSES", args.num_class,
                   "INPUT.CROP.SIZE", "[1.0, 1.0]",
                   "CUDNN_BENCHMARK", "False"]

        # augmentation options
        if args.TTA is True:
            command.extend(["TEST.AUG.ENABLED", "True"])

        return command


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TTA')
    parser.add_argument('--root_dir', type=str, default=root_dir, help='root directory')
    parser.add_argument('--output_dir', type=str, default=f'{root_dir}/model', help='output path')
    parser.add_argument('--model_weights', type=str, default=f'{root_dir}/model/faster_rcnn_R_50_FPN_3x.pkl',
                        help='model weight path')
    parser.add_argument('--model_config', type=str, default=f'{root_dir}/model/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
                        help='model config file path')
    parser.add_argument('--test_dataset', type=str, default="('coco2017val',)", help='custom test dataset')
    parser.add_argument('--num_worker', type=str, default='4', help='number of cpu threads')
    parser.add_argument('--num_class', type=str, default='80', help='number of classes in given dataset')
    parser.add_argument('--num_gpu', type=str, default='1', help='number of gpu for training')
    # ----------------------------------------TTA options----------------------------------------
    parser.add_argument('--TTA', type=bool, default=True, help='Augmentation option')
    # -------------------------------------------------------------------------------------------
    args = parser.parse_args()

    Detectron2Header.test(args)
