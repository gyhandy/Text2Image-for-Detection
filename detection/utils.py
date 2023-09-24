import os, cv2
from typing import Optional
from pathlib import Path
import sys
import ujson as json
from datetime import datetime
import detectron2.data.transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.engine.defaults import DefaultTrainer, default_setup, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode, Visualizer

from wandb_writer import WandbWriter

class COCOAP50Category(COCOEvaluator):
    # overload
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """
        from detectron2.utils.logger import create_small_table
        import numpy as np
        import itertools
        from tabulate import tabulate

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[0, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

class Trainer(DefaultTrainer):
    data_aug: bool = False
    debug: bool=False
    num_of_val_eval: int = 0
    project_name: str = "detectron2"
    wandb_name: Optional[str] = None

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        cls.num_of_val_eval += 1
        if "train_" in dataset_name: # if additonal dataset
            # do train eval every 20 val eval
            if cls.num_of_val_eval % 40 != 0:
                raise NotImplementedError("only do train eval every 40 val eval times!")
        if "_train" in dataset_name:
            # do train eval every 20 val eval
            if cls.num_of_val_eval % 20 != 0:
                raise NotImplementedError("only do train eval every 20 val eval times!")
        print("eval: ", dataset_name)
        return COCOAP50Category(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, "eval", dataset_name))

    def build_writers(self):
        writers =  super().build_writers()
        if not type(self).debug:
            if self.wandb_name:
                writers.append(WandbWriter(config=self.cfg, project=self.project_name, name=self.wandb_name))
            else:
                writers.append(WandbWriter(config=self.cfg, project=self.project_name))
        return writers
    
    @classmethod
    def build_train_loader(cls, cfg):
        if cls.data_aug:
            default_mapper = DatasetMapper(cfg)
            return build_detection_train_loader(cfg, mapper=DatasetMapper(
                cfg, is_train=True,
                augmentations=default_mapper.augmentations.augs + [
                    T.RandomBrightness(0.9, 1.1),
                    T.RandomApply(
                        T.RandomContrast(0.5, 1.5), 0.5),
                ]
            ))
        return build_detection_train_loader(cfg)

def setup_cfg(args, cfg, filter=lambda json_file, td: json_file):
    """
    setup every cfg needed for args
    """
    from PIL import ImageFile, Image
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    pwd = Path(__file__).resolve().parent
    if args.test_dataset == "voc_val":
        with open(pwd.parent / "data" / "voc2012" / "label2id.json") as f:
            label2id = json.load(f)
        num_labels = len(label2id)
    else: # coco_val
        label2id = {}
        num_labels = 80

    # only when not eval mode
    if args.eval_checkpoint is None:
        if args.train_dataset == "syn":
            args.syn_dir = Path(args.syn_dir)
            if "PT_DATA_DIR" in os.environ:
                args.syn_dir = Path(os.environ["PT_DATA_DIR"]) / args.syn_dir
            assert args.syn_dir.exists(), f"path {args.syn_dir} doesn't exist!"
            cfg.DATASETS.TRAIN = ("synthetic_train",)
            json_file = filter(args.syn_dir / "COCO.json", args.td)
            register_coco_instances("synthetic_train", metadata=label2id, json_file=str(json_file),
                                    # image_root=str(args.syn_dir / "images"))
                                    image_root=str(args.syn_dir))
        elif args.train_dataset == "voc_train":
            cfg.DATASETS.TRAIN = ("voc_train",)
            if args.train_dir:
                train_dir = Path(args.train_dir)
            else:
                train_dir = pwd.parent / "data" / "voc2012" / "VOC2012"
            if "PT_DATA_DIR" in os.environ:
                train_dir = Path(os.environ["PT_DATA_DIR"]) / train_dir
            assert train_dir.exists(), f"path {train_dir} doesn't exist!"
            # default 100% voc_train
            # otherwise eg voc_train200
            coconame = args.train_coconame if args.train_coconame else "voc_train.json" 
            # 1464 train
            json_file = filter(train_dir / coconame, args.td)
            register_coco_instances("voc_train", metadata=label2id, json_file=str(json_file),
                                    image_root=str(train_dir))
        elif args.train_dataset == "coco_train":
            cfg.DATASETS.TRAIN = ("coco_train",)
            if args.train_dir:
                train_dir = Path(args.train_dir)
            else:
                train_dir = pwd.parent / "data" / "coco2017"
            if "PT_DATA_DIR" in os.environ:
                train_dir = Path(os.environ["PT_DATA_DIR"]) / train_dir
            assert train_dir.exists(), f"path {train_dir} doesn't exist!"
            # default 100% 
            # otherwise eg instances_train2017_400.json
            coconame = args.train_coconame if args.train_coconame else "instances_train2017_1600.json" 
            # 1600 train
            json_file = filter(train_dir / "annotations" / coconame, args.td)
            register_coco_instances("coco_train", metadata=label2id, json_file=str(json_file),
                                    image_root=str(train_dir / "train2017"))
                                    # image_root=str(train_dir / "train2017_1600"))
        else:
            raise NotImplementedError
        if args.additional_dataset:
            print("additional dataset:", args.additional_dataset)
            for i, path in enumerate(args.additional_dataset):
                ds = f"syn_train_{i}"
                cfg.DATASETS.TRAIN = cfg.DATASETS.TRAIN + (ds, )
                path = Path(path)
                if "PT_DATA_DIR" in os.environ:
                    path = Path(os.environ["PT_DATA_DIR"]) / path
                assert path.exists(), f"path {path} doesn't exist!"
                tmpdir = os.path.join(args.td, ds)
                os.makedirs(tmpdir, exist_ok=True)
                json_file = filter(path / "COCO.json", tmpdir)
                register_coco_instances(ds, metadata=label2id, json_file=str(json_file),
                                            image_root=str(path))
        
        """
        solver config only when training mode
        """
        total_number_of_train_samples = sum([len(DatasetCatalog.get(trainset)) for trainset in cfg.DATASETS.TRAIN])
        num_iter_per_epoch = round(total_number_of_train_samples / args.bsz)
        cfg.SOLVER.MAX_ITER = num_iter_per_epoch * args.epoch
        cfg.SOLVER.STEPS = (cfg.SOLVER.MAX_ITER // 2, cfg.SOLVER.MAX_ITER * 3 // 4)

    if args.preview:
        ds = cfg.DATASETS.TRAIN[0]
        data = DatasetCatalog.get(ds)
        import random, cv2
        import matplotlib.pyplot as plt
        from detectron2.utils.visualizer import Visualizer
        if ds == "synthetic_train":
            ds = os.path.basename(args.syn_dir)
        output = pwd / "preview" / ds
        os.makedirs(output, exist_ok=True)

        for i, d in enumerate(random.sample(data, 30)):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1],
                scale=0.5)
            # metadata=balloon_metadata, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            plt.figure(1, figsize=(10, 8))
            plt.subplot(1,2,1)
            plt.title(d['file_name'])
            plt.imshow(out.get_image())
            plt.subplot(1, 2, 2)
            plt.imshow(img[:, :, ::-1])
            plt.title("RGB")
            plt.tight_layout()
            plt.show()
            plt.savefig(output / f"demo{i}.png")
        sys.exit(0)
    
    if args.test_dataset == "voc_val":
        if args.test_dir:
            test_dir = Path(args.test_dir)
        else:
            test_dir = pwd.parent / "data" / "voc2012" / "VOC2012"

        if "PT_DATA_DIR" in os.environ:
            test_dir = Path(os.environ["PT_DATA_DIR"]) / test_dir
        cfg.DATASETS.TEST = ("voc_val",) + cfg.DATASETS.TRAIN
        json_file = filter(test_dir / "voc_val.json", args.td)
        register_coco_instances("voc_val", metadata=label2id,
                                json_file=str(json_file), image_root=str(test_dir))
    else: # coco
        if args.test_dir:
            test_dir = Path(args.test_dir)
        else:
            test_dir = pwd.parent / "data" / "coco2017"
        if "PT_DATA_DIR" in os.environ:
            test_dir = Path(os.environ["PT_DATA_DIR"]) / test_dir
        cfg.DATASETS.TEST = ("coco_val",) + cfg.DATASETS.TRAIN
        json_file = filter(test_dir / "annotations" / "instances_val2017.json", args.td)
        register_coco_instances("coco_val", metadata=label2id,
            json_file=str(json_file), image_root=str(test_dir / "val2017"))
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_labels
    if args.freeze:
        # freeze all resnet
        cfg.MODEL.BACKBONE.FREEZE_AT = 6

    if args.crop:
        cfg.INPUT.CROP.ENABLED = True
        cfg.INPUT.CROP.TYPE = "relative_range"
        cfg.INPUT.CROP.SIZE = [0.9, 0.9]


    cfg.DATALOADER.NUM_WORKERS = 10
    if args.debug:
        cfg.TEST.EVAL_PERIOD = 100
    else:
        cfg.TEST.EVAL_PERIOD = 3000

    cfg.SEED = args.seed

    cfg.SOLVER.IMS_PER_BATCH = args.bsz
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.WEIGHT_DECAY = args.wd
    # only for logging
    cfg.DATASETS.DATA_AUG = args.data_aug
    cfg.DATASETS.SYN = os.path.basename(args.syn_dir) if args.train_dataset == "syn" else args.train_dataset
    if args.train_dataset == "syn":
        # eg /lab/jxu/paste_segment_philly/artifact/fewshot/VOC/10shot/only_context/[real,syn]-random[5,4,30,512]
        ds_name = "VOC" if "voc" in args.test_dataset else "COCO"
        # eg '10shot/only_context'
        try:
            cfg.DATASETS.SYNNAME = str(args.syn_dir.parent).split(ds_name)[1].strip("/")
        except:
            cfg.DATASETS.SYNNAME = str(args.syn_dir.parent)
    else:
        cfg.DATASETS.SYNNAME = args.train_dataset
    cfg.SOLVER.EPOCH = args.epoch
    cfg.DATASETS.ADD_DS = True if args.additional_dataset else False
    cfg.MODEL.resnet = args.resnet


    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, datetime.now().strftime("%d-%m-%y_%H-%M-%S"))

    if args.eval_checkpoint is not None:
        assert Path(args.eval_checkpoint).exists()
        cfg.MODEL.WEIGHTS = args.eval_checkpoint
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.eval_threshold
        cfg.OUTPUT_DIR = os.path.join(Path("infer"), datetime.now().strftime("%d-%m-%y_%H-%M-%S"))

    if "PT_DATA_DIR" in os.environ:
        cfg.OUTPUT_DIR = os.path.join(os.environ["PT_DATA_DIR"], cfg.OUTPUT_DIR)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, "args.txt"), "w") as f:
        f.write('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    default_setup(cfg, args)
    cfg.freeze()
    return cfg

def infer(cfg):
    predictor = DefaultPredictor(cfg)
    val_ds = cfg.DATASETS.TEST[0]
    data = DatasetCatalog.get(val_ds)
    meta = MetadataCatalog.get(val_ds)
    for i, d in enumerate(data):
        img = cv2.imread(d['file_name'])
        predictions = predictor(img)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = img[:, :, ::-1]
        visualizer = Visualizer(image,
                                meta, instance_mode=ColorMode.IMAGE)

        if "sem_seg" in predictions:
            vis_output = visualizer.draw_sem_seg(
                predictions["sem_seg"].argmax(dim=0)
            )
        if "instances" in predictions:
            instances = predictions["instances"].to("cpu")
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
        vis_output.save(os.path.join(cfg.OUTPUT_DIR, f"{i}.png"))