import argparse
import logging
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], "../", 'detection'))
import tempfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import ujson as json
from omegaconf import OmegaConf
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
    AMPTrainer,
    default_writers, default_setup,
    hooks, SimpleTrainer, launch
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.model_zoo import get_config
from detectron2.utils import comm

from wandb_writer import WandbWriter
from utils import COCOAP50Category

# put here for DDP
from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

def parse_args():
    parser = argparse.ArgumentParser(description="all hparams copied from original lazy config")

    parser.add_argument("--train_dataset", "-s", type=str, required=True, choices=["voc_train", "syn", "coco_train"])
    parser.add_argument("--train_dir", type=str, required=False, help="if unspecified, use default path")
    parser.add_argument("--train_coconame", type=str, required=False, help="if unspecified, use default name")
    parser.add_argument("--syn_dir", type=str, required=False, default="NOT_USED",
                        help="synthic training data folder, contains `images` for images and `COCO.json` for COCO format annotation and `label2id.json` for labels")
    parser.add_argument("--blending", default=['gaussian', 'poisson', 'none', 'box', 'motion'],
                        nargs="+", choices=['gaussian', 'poisson', 'none', 'box', 'motion'],
                        help="only used when -s is 'syn'")
    parser.add_argument("--additional_dataset", nargs="+", help="when use multiple dataset other than -s")

    parser.add_argument("--test_dataset", "-t", type=str, required=True, choices=["voc_val", "coco_val"])
    parser.add_argument("--test_dir", type=str, required=False, help="if unspecified, use default path")

    parser.add_argument("--num_gpus_per_machine", "-g", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num_machines", type=int, default=1, help="total number of machines")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=4e-5)
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--freeze", default=False, action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epoch", type=int, default=400)
    parser.add_argument("--data_aug", default=False, action="store_true")
    parser.add_argument("--init_checkpoint", default=None, type=str)
    parser.add_argument("--resnet", choices=[50, 101], default=50, type=int, help="whether use R101 or R50")

    parser.add_argument("--eval_checkpoint", default=None, type=str)

    parser.add_argument("--debug", default=False, action="store_true", help="if true, don't log in wandb")
    args = parser.parse_args()

    # R50 + 16GB => bsz=8
    # R50 + 32GB => bsz=16
    # R101 + 16GB => bsz=6
    # R101 + 32GB => bsz=12
    if args.resnet == 101:
        args.bsz= args.bsz - 4 * args.num_gpus_per_machine
    
    return args


def setup_cfg(args):
    pwd = Path(__file__).resolve().parent
    if args.test_dataset == "voc_val":
        with open(pwd.parent / "data" / "voc2012" / "label2id.json") as f:
            label2id = json.load(f)
        num_labels = len(label2id)
    else:
        label2id = {}
        num_labels = 80

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from PIL import PngImagePlugin
    LARGE_ENOUGH_NUMBER = 100
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

    cfg = get_config(f"new_baselines/mask_rcnn_R_{args.resnet}_FPN_400ep_LSJ.py")

    # only when not eval mode
    TRAIN = tuple()
    if args.eval_checkpoint is None:
        if args.train_dataset == "syn":
            args.syn_dir = Path(args.syn_dir)
            if "PT_DATA_DIR" in os.environ:
                args.syn_dir = Path(os.environ["PT_DATA_DIR"]) / args.syn_dir
            assert args.syn_dir.exists(), f"path {args.syn_dir} doesn't exist!"
            TRAIN = ("synthetic_train",)
            json_file = args.syn_dir / "COCO.json"
            register_coco_instances("synthetic_train", metadata=label2id, json_file=str(json_file),
                                    image_root=str(args.syn_dir))
        elif args.train_dataset == "voc_train":
            TRAIN = ("voc_train",)
            if args.train_dir:
                train_dir = Path(args.train_dir)
            else:
                train_dir = pwd.parent / "data" / "voc2012" / "VOC2012"
            if "PT_DATA_DIR" in os.environ:
                train_dir = Path(os.environ["PT_DATA_DIR"]) / train_dir
            assert train_dir.exists(), f"path {train_dir} doesn't exist!"
            jsonname = "voc_train."
            if args.train_coconame is not None:
                jsonname = f"{args.train_coconame}.json"

            register_coco_instances("voc_train", metadata=label2id, json_file=str(train_dir / jsonname),
                                    image_root=str(train_dir))
        else: # coco_train
            TRAIN = ("coco_train", )
            if args.train_dir:
                train_dir = Path(args.train_dir)
            else:
                train_dir = pwd.parent / "data" / "COCO2017"
            if "PT_DATA_DIR" in os.environ:
                train_dir = Path(os.environ["PT_DATA_DIR"]) / train_dir
            assert train_dir.exists(), f"path {train_dir} doesn't exist!"
            register_coco_instances("coco_train", metadata=label2id, json_file=str(train_dir / "annotations"/ "instances_train2017.json"),
                                    image_root=str(train_dir / "train2017"))

        if args.additional_dataset:
            print("additional dataset:", args.additional_dataset)
            for i, path in enumerate(args.additional_dataset):
                if "dalle" in path:
                    ds = f"dalle_train{i}"
                else:
                    ds = f"syn_train{i}"
                TRAIN = TRAIN + (ds, )
                path = Path(path)
                if "PT_DATA_DIR" in os.environ:
                    path = Path(os.environ["PT_DATA_DIR"]) / path
                assert path.exists(), f"path {path} doesn't exist!"
                # tmpdir = os.path.join(args.td, ds)
                # os.makedirs(tmpdir, exist_ok=True)
                # json_file = filter(path / "COCO.json", args.blending, tmpdir)
                register_coco_instances(ds, metadata=label2id, json_file=str(path/ "COCO.json"),
                                        image_root=str(path))

        cfg.dataloader.train.dataset.names = list(TRAIN)
        total_number_of_train_samples = sum([len(DatasetCatalog.get(trainset)) for trainset in TRAIN])
        num_iter_per_epoch = round(total_number_of_train_samples / args.bsz)
        cfg.train.max_iter = num_iter_per_epoch * args.epoch
        
        # for COCO
        # when maxiter = 184375, milestones = [163889, 177546], i.e. 88.89% and 96.30%
        cfg.lr_multiplier.scheduler.milestones = (
            int(0.8889 * cfg.train.max_iter), int(0.9630 * cfg.train.max_iter)
        )
        cfg.lr_multiplier.scheduler.num_updates = cfg.train.max_iter

    if args.test_dataset == "voc_val":
        if args.test_dir:
            test_dir = Path(args.test_dir)
        else:
            test_dir = pwd.parent / "data" / "voc2012" / "VOC2012"
        if "PT_DATA_DIR" in os.environ:
            test_dir = Path(os.environ["PT_DATA_DIR"]) / test_dir
        assert test_dir.exists(), f"path {args.test_dir} doesn't exist!"

        TEST = ("voc_val",) + TRAIN
        register_coco_instances("voc_val", metadata=label2id,
                                json_file=str(test_dir / "voc_val.json"), image_root=str(test_dir))
                                # json_file=str(test_dir / "voc_val_atleast5.json"), image_root=str(test_dir))
    else: # "coco_val"
        if args.test_dir:
            test_dir = Path(args.test_dir)
        else:
            test_dir = pwd.parent / "data" / "COCO2017"
        if "PT_DATA_DIR" in os.environ:
            test_dir = Path(os.environ["PT_DATA_DIR"]) / test_dir
        assert test_dir.exists(), f"path {args.test_dir} doesn't exist!"

        TEST = ("coco_val",) + TRAIN
        register_coco_instances("coco_val", metadata=label2id,
                                json_file=str(test_dir / "annotations" / "instances_val2017.json"), image_root=str(test_dir / "val2017"))

    cfg = LazyConfig.apply_overrides(cfg, [
        # "model.roi_heads.batch_size_per_image=256"
        f"model.roi_heads.num_classes={num_labels}",

        f"dataloader.train.total_batch_size={args.bsz}",
        f"dataloader.test.dataset.names={list(TEST)}",
        f"dataloader.evaluator._target_=seg_lazy.COCOAP50Category",

        f"train.eval_period={300 if args.debug else 3000}",
        f"train.checkpointer.period={50 if args.debug else 3000}",
        # f"train.eval_period={50}",
        f"train.seed={args.seed}",
        f"train.init_checkpoint='detectron2://ImageNetPretrained/MSRA/R-{args.resnet}.pkl'",

        f"optimizer.lr={args.lr}",
        f"optimizer.weight_decay={args.wd}",

        # only for logging
        f"logging.syn_dir={repr(os.path.basename(args.syn_dir))}",
        f"logging.epoch={args.epoch}",
        f"logging.resnet={args.resnet}",
        f"logging.lr={args.lr}",
        f"logging.mode={'f' if args.freeze else ''}{''.join(b[0] for b in args.blending)}",
        f"logging.additional={args.additional_dataset if args.additional_dataset else []}"
    ])
    if args.init_checkpoint is not None:
        print("loading from ckpt:", args.init_checkpoint)
        if "PT_DATA_DIR" in os.environ:
            args.init_checkpoint = os.path.join(os.environ["PT_DATA_DIR"], args.init_checkpoint)
        cfg.train.init_checkpoint = args.init_checkpoint
    output_dir = os.path.join(cfg.train.output_dir, str(datetime.now().strftime("%d-%m-%y_%H:%M:%S")))
    if "PT_DATA_DIR" in os.environ:
        output_dir = os.path.join(os.environ["PT_DATA_DIR"], output_dir)
    cfg.train.output_dir = output_dir

    if args.num_gpus_per_machine == 1 and args.num_machines == 1:
        # single gpu
        cfg.model.backbone.bottom_up.stem.norm = "BN"
        cfg.model.backbone.bottom_up.stages.norm = "BN"
        cfg.model.backbone.norm = "BN"

    if args.freeze:
        cfg.model.backbone.bottom_up.freeze_at = 6

    if args.data_aug:
        # add additional data aug other than ResizeScale, FixedSizeCrop, RandomFlip (default, i.e. Large Scale Jitter)
        cfg.dataloader.train.mapper.augmentations.extend([
            {"_target_": "detectron2.data.transforms.RandomBrightness", "intensity_max": 1.1, "intensity_min": 0.9},
            {"_target_": "detectron2.data.transforms.RandomApply", "prob": 0.5,
             "tfm_or_aug": {
                 "_target_": "detectron2.data.transforms.RandomContrast", "intensity_min": 0.5, "intensity_max": 1.5
             }},
        ])

    if args.eval_checkpoint is not None:
        assert Path(args.eval_checkpoint).exists()
        cfg.train.init_checkpoint = args.eval_checkpoint
        cfg.train.output_dir = os.path.join(Path("infer"), datetime.now().strftime("%d-%m-%y_%H-%M-%S"))

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    with open(os.path.join(cfg.train.output_dir, "args.txt"), "w") as f:
        f.write('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    return cfg


def do_test(cfg, model, cur_iter):
    if "evaluator" in cfg.dataloader:
        results = OrderedDict()
        logger = logging.getLogger(__name__)

        for dataset_name in cfg.dataloader.evaluator.dataset_name:
            # rarely eval on train
            # if dataset_name in cfg.dataloader.train.dataset.names and cur_iter % 100_000 != 0:
            #     continue
            if dataset_name in cfg.dataloader.train.dataset.names:
                continue
            evaluator = deepcopy(cfg.dataloader.evaluator)
            evaluator.dataset_name = dataset_name
            test = deepcopy(cfg.dataloader.test)
            test.dataset.names = [dataset_name]
            results_i = inference_on_dataset(
                model, instantiate(test), instantiate(evaluator)
            )
            results[dataset_name] = results_i
            if comm.is_main_process():
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
        # do not flat as need the dataset ap
        # if len(results) == 1:
        #     results = list(results.values())[0]
        return results

def setup(cfg, is_test=False, debug=False):
    """
    build necessary component
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    if is_test:
        return model

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)
    model = create_ddp_model(model, **cfg.train.ddp)
    # init here to log dataset stat
    writer = hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter) + \
                    ([WandbWriter(project="dalle-for-detection-seg", config=OmegaConf.to_container(cfg, resolve=True))] if not debug else []), 
                period=cfg.train.log_period,
            ) if comm.is_main_process() else None
    train_loader = instantiate(cfg.dataloader.train)

    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer) if comm.is_main_process() else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model, cur_iter=trainer.iter + 1)),
            writer
        ]
    )
    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=False)
    return trainer

def do_train(cfg, debug=False):
    trainer = setup(cfg, debug=debug)
    start_iter = 0
    # do_test(cfg, trainer.model, cur_iter=0)
    trainer.train(start_iter, cfg.train.max_iter)

def main(args):
    cfg = setup_cfg(args)
    default_setup(cfg, args)
    if args.eval_checkpoint is not None:
        model = setup(cfg, is_test=True)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        do_test(cfg, model, cur_iter=0)
    else:
        do_train(cfg, args.debug)

if __name__ == "__main__":
    args = parse_args()

    with tempfile.TemporaryDirectory() as td:
        args.td = td
        launch(
            main, num_gpus_per_machine=args.num_gpus_per_machine,
            num_machines=args.num_machines, machine_rank=0, dist_url="auto", args=(args, )
        )