import json
import os
import random
from concurrent import futures
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import ujson as json
# from detectron2.utils.env import seed_all_rng
from tqdm import tqdm

from anno import Anno
from background import PastedBackground
from foreground import CutObjects
from utils import convert_to_COCO


class Paster:
    """
    paste @self.foregrounds into @self.backgrounds
    """

    def __init__(self, label2id: dict,
                 out_size: int = 512,
                 repeat_background: int = 1, select_prob: str = "uniform",
                 random_paste=False):
        """
        out_dir/
            foregrounds.csv
            backgrounds.csv # without repeat
            TODO
            xxx
        Args:
            label2id: dict with label text & id
            repeat_background: # times background image is repeated
            select_prob: how to select foreground
            random_paste: whether to use random paste, if False use space maximization paste
        """
        # blending_list = ['gaussian', 'poisson', 'none', 'box', 'motion']
        assert select_prob in ["uniform", "balanced", "supercategory_CDI", "supercategory"]
        # seed_all_rng(seed)
        # TODO
        self.select_prob = select_prob
        self.random_paste = random_paste
        self.out_size = out_size
        self.blending_list: List[str] = ['gaussian', ]
        assert all(b in ['gaussian', 'poisson', 'none', 'box', 'motion'] for b in self.blending_list)
        self.repeat_background = repeat_background  # repeat background only
        self.backgrounds = []
        self.foregrounds = []

        self.id2label = {v: k for k, v in label2id.items()}
        self.label2id = dict(label2id)
        Anno.label2id = label2id
        Anno.id2label = self.id2label

    def aggregate_json(self,
                       input_dir: Path, max_workers=1, json_name="COCO"):
        """
        convert instance mask to COCO format
        input_dir must contain @image_folder folder for pasted images, and json are saved here
        """
        output_json_dict = {
            "images": [],
            "type": "instances",
            "annotations": [],
            "categories": [
                {'supercategory': 'none', 'id': label_id, 'name': label}
                for label, label_id in self.label2id.items()
            ]
        }

        def read_json(path):
            with open(input_dir / "tmp" / path) as f:
                return json.load(f)

        files = list(os.listdir(input_dir / "tmp"))
        with tqdm(total=len(files), desc="COCO agg") as pbar, \
                futures.ThreadPoolExecutor(max_workers) as executor:
            todos = []
            bnd_id = 0  # coco need integer bnd ids
            for file in files:
                todos.append(executor.submit(read_json, file))
            for future in futures.as_completed(todos):
                data = future.result()
                output_json_dict["images"].extend(data["images"])
                anno = data["annotations"]
                for bbox in anno:
                    bbox["id"] = bnd_id
                    bnd_id += 1
                output_json_dict["annotations"].extend(anno)
                pbar.update(1)
        print(f"saving to {input_dir}")
        with open(input_dir / f"{json_name}.json", "w") as f:
            json.dump(output_json_dict, f)
        with open(input_dir / "label2id.json", "w") as f:
            json.dump(dict(self.label2id), f, indent=4)
        # can rm tmp folder if you want
        # shutil.rmtree(input_dir / "tmp")

    def cut_and_paste_one_image(self, i: int, out_dir: Path, out_size: int,
                                num_cut_images: int, blending_list: List[str], probs: List[float],
                                num_cut_lowerbound: Optional[int],
                                max_degree: int, random_paste: bool, scale_factor: int, center_translation_factor: int,
                                use_random_scaling: bool):
        """ return path of background image + all objects (original bg + pasted fg) for this image """
        background: PastedBackground = deepcopy(self.backgrounds[i])
        path_to_save = f"{Path(background.imagepath).stem}_{i}"
        if (out_dir / "tmp" / f"{path_to_save}.json").exists():
            return

        background.resize(out_size)
        if num_cut_lowerbound:
            k = random.randint(num_cut_lowerbound, num_cut_images)
        else:
            k = num_cut_images
        cut_images = random.choices(self.foregrounds, k=k, weights=probs)
        foregrounds = CutObjects(cut_images)
        locations, foregrounds = background.find_paste_location(foregrounds, max_degree=max_degree,
                                                                random_paste=random_paste, scale_factor=scale_factor,
                                                                center_translation_factor=center_translation_factor,
                                                                use_random_scaling=use_random_scaling)

        for blending in blending_list:
            pasted_background = deepcopy(background)
            for paste_location, foreground_object in zip(locations, foregrounds):
                pasted_background.paste(
                    foreground_object=foreground_object, paste_location=paste_location, blending=blending)
            pasted_background.save(path=out_dir, name=f"{path_to_save}_{blending}")

        convert_to_COCO(out_dir, path_to_save, background, blending_list)
    
    def foreground_sample_prob_by_supercategory(self, bg_filename, probs):
        """
        either sample based on supercategory of @bg_filename
        or fallback to @probs
        """
        if "background" in bg_filename and self.select_prob == "supercategory_CDI": # bgtemplate
            return probs # random select fg on bg template images
        return probs

    def save(self, output_dir: Path):
        import torch
        with open(output_dir / "paster.json", "w") as f:
            json.dump({
                "counts": [len(self.foregrounds), len(self)],
                "foreground": [str(fg.img_path) for fg in self.foregrounds],
                "background": [str(bg.imagepath) for bg in self.backgrounds],
            }, f)
        torch.save(self, output_dir / "paster.pt")
        # with open(output_dir / "paster.pt", "w") as f:

    @staticmethod
    def from_save(input_dir: Path):
        import torch
        return torch.load(input_dir / "paster.pt")

    def get_select_prob(self, select_prob) -> np.ndarray:
        """
        return selection prob for each ele of @self.foregrounds
        """
        if select_prob == "uniform":  # uniform over provided data, thus can be balanced and reflect distribution of bg
            probs = np.ones(len(self.foregrounds)) / len(self.foregrounds)
        else:  # balanced based on label st each label shows up equal likely
            labels = np.array([label for _, label, _, _ in self.foregrounds])  # (N, )
            probs = np.zeros_like(labels).astype(float)
            for class_i in np.unique(labels):
                class_indices = labels == class_i  # boolean (N, )
                num_samples_class_i = class_indices.sum()
                assert num_samples_class_i > 0
                probs[class_indices] = 1.0 / num_samples_class_i
            probs /= probs.sum()  # st sum(probs) == 1
            """
            # check if probs work
            labels = []
            for _ in range(10000):
                cut_images = random.choices(self.backgrounds, k=4, weights=probs)
                for _, l, _, _ in cut_images:
                    labels.append(l)
            from collections import Counter
            ct = Counter(labels)
            print(ct) # should be almost same number for each class
            """
        assert len(probs) == len(self.foregrounds)
        return probs

    def __len__(self): # len def as background
        return len(self.backgrounds)

    def truncate(self, slice):
        self.backgrounds = self.backgrounds[slice]
    
    def validate(self):
        assert len(self.foregrounds) > 0 and len(self.backgrounds) > 0

    def cut_and_paste(
        self, out_dir: Path, num_cut_images: int = 2, max_workers=1,
        # rotate
        max_degree: int = 30,
        # variant
        scale_factor=0, center_translation_factor=0, use_random_scaling=False,
        num_cut_lowerbound: Optional[int] = None
    ):
        """
        will create the following in @out_dir:
            Images folder: pasted RGB images
            Masks folder: semantic level segmentation mask
            COCO.json: instance level COCO segmentation annotation
        """
        self.validate()
        self.backgrounds = self.backgrounds * self.repeat_background
        self.save(output_dir=out_dir) # with updated backgrounds
        os.makedirs(out_dir / "Images", exist_ok=True)
        os.makedirs(out_dir / "Masks", exist_ok=True)

        probs = self.get_select_prob(self.select_prob)

        cut_and_paste_one_image = partial(
            self.cut_and_paste_one_image, out_dir=out_dir, out_size=self.out_size, probs=probs,
            num_cut_images=num_cut_images, max_degree=max_degree, blending_list=self.blending_list,
            num_cut_lowerbound=num_cut_lowerbound,
            random_paste=self.random_paste, scale_factor=scale_factor, center_translation_factor=center_translation_factor,
            use_random_scaling=use_random_scaling)
        for i in list(range(len(self))):
            cut_and_paste_one_image(i)

        # todos = []
        # with tqdm(total=len(self), desc="cutpaste") as pbar, \
        #         futures.ThreadPoolExecutor(max_workers) as executor:
        #     for i in list(range(len(self))):
        #         todos.append(executor.submit(
        #             cut_and_paste_one_image, i))
        #     for future in futures.as_completed(todos):
        #         future.result()
        #         pbar.update(1)
        print("converting to COCO format")
        self.aggregate_json(out_dir, max_workers, json_name="COCO")