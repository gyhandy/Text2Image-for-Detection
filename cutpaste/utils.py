import glob
import json
import os
import re
from concurrent import futures
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm

from anno import VOCAnno, Anno, EntityAnno
from cutpaste.background import PastedBackground
from foreground import CutObjects, CutObject


####################################################################################################
# foregrounds
def read_real_VOC_foregrounds(data_dir, anno_dir, seg_dir, data_list) -> List[CutObjects]:
    """
    Read real foregrounds from VOC dataset
        Each has VOCAnno
    """
    foregrounds = []
    with open(data_list, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines, total=len(lines), desc="reading real VOC fg"):
        fields = line.strip().split()
        img_path = os.path.join(data_dir, fields[0] + '.jpg')
        anno_path = os.path.join(anno_dir, fields[0] + '.xml')
        seg_img_path = os.path.join(seg_dir, fields[0] + '.png')
        assert os.path.exists(anno_path) and os.path.exists(img_path) and os.path.exists(seg_img_path)
        anno: VOCAnno = Anno.factory(anno_path, seg_img_path)
        foregrounds.extend(CutObjects().add_image(img_path, anno))
    return foregrounds


def read_entity_foregrounds(dataset, rgb_dir, mask_dir) -> List[CutObjects]:
    """
    Read syn foregrounds (processed by entity segmentation, then selected by GradCAM)
        Each has EntityAnno
    """
    rgb_dir, mask_dir = map(Path, [rgb_dir, mask_dir])

    foregrounds = []
    # VOC
    def get_voc_image(mask_file):
        # eg voc2012/foreground/foreground_mask_old/car_mask/a car in a white background30.png
        _, label, filename = mask_file.rsplit("/", 2)
        label = label.replace("_mask", "")
        # infer rgb img_path
        # eg (a car in a white background, 30, _)
        target_caption, target_num, _ = re.split(r'(\d+)', filename)
        img_path = None
        for class_dir in rgb_dir.iterdir():
            for caption in os.listdir(class_dir):
                if caption == target_caption:
                    img_path = class_dir / caption / f"{target_num}.png"
            if img_path is not None:
                break
        assert img_path is not None, f"{str(mask_file)} Not found!"
        assert os.path.exists(img_path)
        anno: EntityAnno = Anno.factory(None, mask_file)
        return CutObjects().add_image(img_path, anno)

    todos = []
    all_mask_files = list(glob.glob(str(mask_dir / "*_mask" / "*.png")))
    with tqdm(total=len(all_mask_files), desc="collecting real fg") as pbar, \
            futures.ThreadPoolExecutor(100) as executor:
        for mask_file in all_mask_files:
            todos.append(executor.submit(get_voc_image, mask_file))
        for future in futures.as_completed(todos):
            res = future.result()
            foregrounds.extend(res)
            pbar.update(1)
    return foregrounds

####################################################################################################
# backgrounds
def read_real_VOC_backgrounds(data_dir, anno_dir, seg_dir, data_list) -> List[PastedBackground]:
    """
    load list of image name and image labels ([i] is img_name_list[i]'s K+1 class boolean vector)
    can be used in reading fg or bg
    """
    backgrounds = []
    with open(data_list, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines, total=len(lines), desc="reading real VOC bg"):
        fields = line.strip().split()
        img_path = os.path.join(data_dir, fields[0] + '.jpg')
        anno_path = os.path.join(anno_dir, fields[0] + '.xml')
        seg_img_path = os.path.join(seg_dir, fields[0] + '.png')
        assert os.path.exists(anno_path)
        assert os.path.exists(img_path)
        assert os.path.exists(seg_img_path)
        backgrounds.append(PastedBackground(
            imagepath=img_path, anno=VOCAnno(anno_path, seg_img_path)
        ))
    return backgrounds
def read_background_template(data_dir: str) -> List[PastedBackground]:
    backgrounds = []
    for rgb_file in glob.iglob(f"{data_dir}/**/*.png",
                               recursive=True):
        if "azDownload" in str(rgb_file): continue
        if "group_0" in str(rgb_file): continue
        backgrounds.append(PastedBackground(rgb_file))
    return backgrounds

def read_dalle_backgrounds(data_dir: str, clip_strategy="use") -> List[PastedBackground]:
    """
    use syn images for background, and ignore possible foreground in syn images
    data_dir: path to get dalle generated syn images
    """
    data_dir = Path(data_dir)
    backgrounds = []
    if clip_strategy in ["use", "reverse"]:
        with open(data_dir / "clip_postprocessed.json") as f:
            data = json.load(f)
        if clip_strategy == "use":
            # keep only in CLIP processed
            for imageid, captions in data.items():
                for caption, selected_ids in captions.items():
                    for id in selected_ids:
                        img = data_dir / f"{imageid}.jpg" / caption / id
                        assert img.exists()
                        backgrounds.append(PastedBackground(str(img)))
        else: # keep only NOT in CLIP processed
            for imgid in data_dir.iterdir():
                if not imgid.name.endswith(".jpg"):
                    continue
                if imgid not in data:
                    for caption in imgid.iterdir():
                        for img in caption.iterdir():
                            backgrounds.append(PastedBackground(str(img)))
    else: # do not use clip but raw
        for img in data_dir.iterdir():
            backgrounds.append(PastedBackground(str(img)))
    return backgrounds

def convert_to_COCO(input_dir,
                    image_id_src, background, blending_list,
                    image_folder="Images", image_suffix="png", output_dir=None):
    """
    image in input_dir / image_folder / image_id_src
    save in output_dir / tmp
    COCO image path in the format of image_folder / image_id_src, relative so that in detectron we can provide input_dir (eg use in remote server)
    """
    if output_dir is None:
        output_dir = input_dir
    output_json_dict = {
        "images": [],
        "annotations": []
    }
    bnd_id = 1
    for blending in blending_list:
        # image_id eg 2007_000515_16
        if blending != "":
            image_id = f"{image_id_src}_{blending}"
        else:
            image_id = image_id_src
        file_name = os.path.join(image_folder, f"{image_id}.{image_suffix}")
        from PIL import PngImagePlugin
        LARGE_ENOUGH_NUMBER = 100
        PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)
        img = Image.open(input_dir / file_name)
        width, height = img.size
        output_json_dict["images"].append({
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": image_id
        })
        for polygons, bbox, area, category in background.to_COCO_ann():
            output_json_dict["annotations"].append({
                "segmentation": polygons,
                "area": area,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": bbox,
                "category_id": int(category),
                "id": f"{image_id}_{bnd_id}"
            })
            bnd_id += 1
    tmpdir = output_dir / "tmp"
    os.makedirs(tmpdir, exist_ok=True)
    tmp_json_path = tmpdir / f"{image_id_src}.json"
    with open(tmp_json_path, "w") as f:
        json.dump(output_json_dict, f)
