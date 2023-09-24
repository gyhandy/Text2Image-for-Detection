import argparse
import os
import sys, uuid
import tempfile
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import launch
from utils import setup_cfg, infer, Trainer
from pathlib import Path
import ujson as json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", "-s", type=str, required=True, choices=["voc_train", "syn", "coco_train"])
    parser.add_argument("--train_dir", type=str, required=False, help="if unspecified, use default path")
    parser.add_argument("--train_coconame", type=str, required=False, help="if unspecified, use default name")
    parser.add_argument("--syn_dir", type=str, required=False, default="NOT_USED",
                    help="synthic training data folder, contains `images` for images and `COCO.json` for COCO format annotation and `label2id.json` for labels")
    parser.add_argument("--additional_dataset", nargs="+", help="when use multiple dataset other than -s, put more heavy dataset in here")

    parser.add_argument("--test_dataset", "-t", type=str, choices=["voc_val", "coco_val"])
    parser.add_argument("--test_dir", type=str, required=False, help="if unspecified, use default path")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=0.0005)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--freeze", default=False, action="store_true")
    parser.add_argument("--data_aug", default=False, action="store_true", help="data augmentation on synthetic data, RandomContrast etc not including crop, use crop only when --crop")
    parser.add_argument("--crop", default=False, action="store_true")
    parser.add_argument("--epoch", type=int, default=20)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--debug", default=False, action="store_true", help="if true, don't log in wandb")

    parser.add_argument("--resnet", choices=[50, 101], default=50, type=int, help="whether use R101 or R50")

    parser.add_argument("--preview", default=False, action="store_true")
    
    parser.add_argument("--init_checkpoint", default=None, type=str)

    parser.add_argument("--eval_checkpoint", default=None, type=str)
    parser.add_argument("--eval_threshold", default=0.7, type=float)

    parser.add_argument("--num_gpus_per_machine", "-g", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num_machines", type=int, default=1, help="total number of machines")

    args = parser.parse_args()
    return args

def filter(json_file, td):
    """
    filter out instance seg annotation but only object detection one
    """
    with open(json_file) as f:
        data = json.load(f)
    newanno = []
    for anno in data["annotations"]:
        if len(anno["segmentation"]) == 0:
            # already no seg
            return json_file
        anno['segmentation'] = []
        newanno.append(anno)
    data["annotations"] = newanno
    id = str(uuid.uuid4())
    os.makedirs(Path(td) / id)
    json_file = Path(td) / id / "COCO.json"
    with open(json_file, "w") as f:
        json.dump(data, f)
    return json_file

def fetch_cfg(args):
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/faster_rcnn_R_{args.resnet}_FPN_3x.yaml"))
    if args.init_checkpoint is not None:
        print("loading from ckpt:", args.init_checkpoint)
        if "PT_DATA_DIR" in os.environ:
            args.init_checkpoint = os.path.join(os.environ["PT_DATA_DIR"], args.init_checkpoint)
        cfg.MODEL.WEIGHTS = args.init_checkpoint
    else:
        cfg.MODEL.WEIGHTS = f"detectron2://ImageNetPretrained/MSRA/R-{args.resnet}.pkl"
    return cfg

def main(args):
    cfg = fetch_cfg(args)
    cfg = setup_cfg(args, cfg, filter=filter)
    if args.eval_checkpoint is not None:
        infer(cfg)
        sys.exit(0)

    Trainer.data_aug = args.data_aug
    Trainer.debug = args.debug
    Trainer.project_name = "dalle-for-detection"
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    args = parse_args()

    with tempfile.TemporaryDirectory() as td:
        args.td = td
        launch(
            main, num_gpus_per_machine=args.num_gpus_per_machine,
            num_machines=args.num_machines, machine_rank=0, dist_url="auto", args=(args, )
        )