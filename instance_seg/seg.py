import os, cv2
import sys
sys.path.insert(1, os.path.join(sys.path[0], "../", 'detection'))
import tempfile

from detectron2 import model_zoo
from detectron2.config import get_cfg
from utils import setup_cfg, infer, Trainer
from train import parse_args

def fetch_cfg(args):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/mask_rcnn_R_{args.resnet}_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    # from scratch
    cfg.MODEL.WEIGHTS = f"detectron2://ImageNetPretrained/MSRA/R-{args.resnet}.pkl"
    return cfg

if __name__ == "__main__":
    args = parse_args()

    with tempfile.TemporaryDirectory() as td:
        args.td = td
        cfg = fetch_cfg(args)
        cfg = setup_cfg(args, cfg)
        if args.eval_checkpoint is not None:
            infer(cfg)
            sys.exit(0)

        Trainer.data_aug = args.data_aug
        Trainer.debug = args.debug
        Trainer.project_name = "paste-seg-instance"
        trainer = Trainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()