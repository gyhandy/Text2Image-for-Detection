import random
from typing import List

import cv2
import numpy as np
from PIL import Image

from anno import Anno


def get_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert len(np.where(rows)[0]) > 0
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    assert ymax >= ymin and xmax >= xmin
    return int(xmin), int(xmax), int(ymin), int(ymax)


def get_area(xmin, xmax, ymin, ymax):
    return (xmax - xmin) * (ymax - ymin)


class CutObject:
    """
    mask object
    input mask dummy is 0, positive if occupied (dep on category, 1-20 for VOC)
    """
    def __init__(self, img_path: str, img: Image.Image, mask: Image.Image):
        self.img_path = img_path
        self.img = img
        self.mask = mask
        uniques = set(np.unique(self.mask))
        assert len(uniques) in [2, 1]  # it's possible to get perfect mask, so only positive val (1)
        # if not 0, then it's category for mask
        uniques: set = uniques.difference({0})
        self.category, = uniques
        self.category_name = Anno.id2label[self.category]


    def min_enclosing_circle(self):
        contours, _ = cv2.findContours(np.array(self.mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        center, radius = cv2.minEnclosingCircle(np.concatenate(contours, 0))
        return radius, center

    def scale(self, scaling_factor: float):
        orig_w, orig_h = self.img.size
        o_w, o_h = int(scaling_factor * orig_w), int(scaling_factor * orig_h)
        self.img = self.img.resize((o_w, o_h), Image.ANTIALIAS)
        self.mask = self.mask.resize((o_w, o_h), Image.NEAREST)

    def rotate(self, size, max_degree=60):
        w, h = size
        while True:
            rot_degrees = random.randint(-max_degree, max_degree)
            foreground_tmp = self.img.rotate(rot_degrees, expand=True)
            foreground_mask_tmp = self.mask.rotate(rot_degrees, expand=True)
            o_w, o_h = foreground_tmp.size
            if w - o_w > 0 and h - o_h > 0:
                break
        self.img = foreground_tmp
        self.mask = foreground_mask_tmp

    def save(self, name):
        self.img.save(f"{name}-fg-image.png")
        self.mask.save(f"{name}-fg-mask.png")

class CutObjects(list):
    """
    list of objects (i.e. foregrounds) to cut, and later will be pasted on PastedImage
      can contain multiple foregrounds from the same image
    """
    def __init__(self, *args):
        super().__init__(*args)
        self: List[CutObject]

    def add_image(self, img_path, foreground_anno: Anno, area_threshold=700):
        """
        add per-object mask of the given image
        only add if area exceeds area_threshold
        """
        foreground_img = Image.open(img_path)
        for i, foreground_object in enumerate(foreground_anno.objects(), 1):
            """
            binary mask, 0 is dummy, positive int (label id for ith object) is object mask
            """
            foreground_mask = foreground_anno.create_mask(for_object=i)
            xmin, xmax, ymin, ymax = get_box(foreground_mask)
            if get_area(xmin, xmax, ymin, ymax) < area_threshold:
                continue
            foreground = foreground_img.crop((xmin, ymin, xmax, ymax))
            foreground_mask = foreground_mask.crop((xmin, ymin, xmax, ymax))
            self.append(CutObject(img_path, foreground, foreground_mask))
        return self

    def shuffle(self):
        random.shuffle(self)