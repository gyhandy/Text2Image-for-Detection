import json
import os, sys
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

pwd = Path(__file__).parent.resolve()
output = pwd / "out"

voc_dir = pwd.parent / "data/voc2012" / "VOC2012"
with open(voc_dir.parent / "label2id.json") as f:
    label2id = json.load(f)

artifact_dir = Path(sys.argv[1])
assert artifact_dir.exists()
print(artifact_dir)

output = output / artifact_dir.stem
os.makedirs(output, exist_ok=True)
coco_name = "COCO.json"
register_coco_instances("synthetic_train", metadata=label2id, json_file=str(artifact_dir / coco_name),
                        image_root=str(artifact_dir))
setup_logger()

ds = "synthetic_train"
# ds = "VOC_test"
data = DatasetCatalog.get(ds)
for i, d in enumerate(random.sample(data, 30)):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1],
                            scale=0.5)
    # metadata=balloon_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    plt.figure(1, figsize=(10, 8))
    plt.subplot(1, 2, 1)
    plt.title(d['file_name'])
    plt.imshow(out.get_image())
    plt.subplot(1, 2, 2)
    plt.imshow(img[:, :, ::-1])
    plt.title("RGB")
    plt.tight_layout()
    plt.savefig(output / f"demo{i}.png")
    plt.show()