import math
import random
from pathlib import Path
from typing import Optional, Iterator

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as cocomask

from anno import Anno
from foreground import CutObject, CutObjects
from pb import create_mask, poisson_blend
from pyblur3 import LinearMotionBlur


def binarize_mask(mask, set_boundary=False) -> np.ndarray:
    """
    make it 255 if occupied, regardless of pixel category
    """
    mask = np.array(mask)
    mask = np.where(mask > 0, 255, 0).astype('uint8')
    if set_boundary:
        mask[:, 0] = 255
        mask[:, -1] = 255
        mask[0, :] = 255
        mask[-1, :] = 255
    return mask


class PastedBackground:
    """
    background image to be pasted on
    """

    def __init__(self, imagepath: str, anno: Optional[Anno] = None):
        """
        if anno is None, no need to find annotation in given image, i.e. we ignore potential foregrounds
        """
        self.image: Image.Image = Image.open(imagepath)
        self.imagepath = imagepath
        if anno:
            self.ignore_foreground = False
            # semantic mask
            # 0 if dummy, positive int (label id for each of the potential object/foreground) is object mask
            self.mask = anno.create_mask(for_object=None)
            # Image Mask (each instance a unique id) & instance id to actual category (starts from 1)
            self.instance_mask, self.instance_mask_id2category = anno.create_instance_mask()
        else:
            self.ignore_foreground = True
            w, h = self.size
            self.mask = Image.fromarray(np.zeros((h, w), dtype="uint8"))
            self.instance_mask, self.instance_mask_id2category = self.mask.copy(), {}

    @property
    def size(self):
        """ w and h """
        return self.image.size

    def resize(self, out_size: int):
        self.image = self.image.resize((out_size, out_size), Image.LANCZOS)
        self.mask = self.mask.resize((out_size, out_size), Image.NEAREST)
        self.instance_mask = self.instance_mask.resize((out_size, out_size), Image.NEAREST)

    def find_paste_location(self,
                            foregrounds: CutObjects, max_degree, random_paste=False,
                            scale_factor=0, center_translation_factor=0, use_random_scaling=False):
        """
        modify self.mask in place (paste with foreground)
        return new foregrounds (after scaling and rotation)
        if random_paste: select random paste location (for abalation) and random scale 0.3-0.7
        """
        # 1. loop through all objects
        foregrounds.shuffle()
        foregrounds_to_paste = []
        locations = []
        for foreground_object in foregrounds:
            w, h = self.size
            if random_paste:
                (x, y) = random.randint(0, h), random.randint(0, w)
                radius2, center2 = foreground_object.min_enclosing_circle()
                scaling = random.uniform(0.3, 0.7)
                radius = scaling * radius2
            else:
                # 2. find max inscribing circle in the background non-occupied area
                radius, (x, y) = self.max_inscribing_circle()
                if center_translation_factor != 0:  # translate center
                    sgn = 1 if random.random() < 0.5 else -1
                    x += sgn * h * (center_translation_factor / 100)
                    y += sgn * w * (center_translation_factor / 100)

                # 3. compute per-object min enclosing circle
                radius2, center2 = foreground_object.min_enclosing_circle()

                # 4. after scale, make foreground_object align with max inscribing circle
                if use_random_scaling:
                    scaling = random.uniform(0.3, 0.7)
                else:
                    scaling = radius / radius2
                    if scale_factor != 0:  # scale by @scale_factor %
                        scaling *= (1 + scale_factor / 100)
            try:
                foreground_object.scale(scaling)
                o_w, o_h = foreground_object.img.size
                assert w - o_w >= 0 and h - o_h >= 0 and o_w > 0 and o_h > 0
            except:
                continue

            foreground_object.rotate(size=self.size, max_degree=max_degree)

            foregrounds_to_paste.append(foreground_object)
            locations.append((
                int(x - radius), int(y - radius)
            ))
            self.mask.paste(foreground_object.mask, locations[-1],
                            Image.fromarray(binarize_mask(foreground_object.mask)))
            new_instance_id = len(self.instance_mask_id2category) + 1
            self.instance_mask_id2category[new_instance_id] = foreground_object.category
            self.instance_mask.paste(
                Image.fromarray(
                    np.where(np.array(foreground_object.mask) == foreground_object.category, new_instance_id, 0).astype(
                        "uint8")),
                locations[-1], Image.fromarray(binarize_mask(foreground_object.mask))
            )
            if self.ignore_foreground:
                # even if ignore foreground first, now it will not since one foreground was pasted
                self.ignore_foreground = False

        return locations, CutObjects(foregrounds_to_paste)

    def save(self, name, path=None):
        """
        before save, mask is label 1 - 20, and 0 if dummy
        change it to 0-19 label (corresponding to labels.txt but minus 1), 255 if dummy
        """
        if path is None:
            img_path = mask_path = Path(".")
        else:
            img_path = path / "Images"
            mask_path = path / "Masks"
        mask = self.mask
        self.image.save(img_path / f"{name}.png")
        mask.save(mask_path / f"{name}.png")

    def max_inscribing_circle(self):
        """
        max inscribing circle that contains all background objects
        """
        if self.ignore_foreground:
            assert list(np.unique(self.mask)) == [0], "should be only 0 i.e. dummy"
            w, h = self.size
            x, y = random.randint(0, h), random.randint(0, w)
            dist_x = min(abs(x), abs(x - h // 2))
            dist_y = min(abs(y), abs(y - w // 2))
            return min(dist_x, dist_y), (x, y)

        background_mask = binarize_mask(self.mask, set_boundary=True)
        dist_map = cv2.distanceTransform(255 - background_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        _, radius, _, center = cv2.minMaxLoc(dist_map)
        return radius, center

    def paste(self, blending: str, paste_location: tuple, foreground_object: CutObject):
        def LinearMotionBlur3C(img):
            """Performs motion blur on an image with 3 channels. Used to simulate
               blurring caused due to motion of camera.
            Args:
                img(NumPy Array): Input image with 3 channels
            Returns:
                Image: Blurred image by applying a motion blur with random parameters
            """

            def randomAngle(kerneldim):
                """Returns a random angle used to produce motion blurring
                Args:
                    kerneldim (int): size of the kernel used in motion blurring
                Returns:
                    int: Random angle
                """
                kernelCenter = int(math.floor(kerneldim / 2))
                numDistinctLines = kernelCenter * 4
                validLineAngles = np.linspace(0, 180, numDistinctLines, endpoint=False)
                angleIdx = np.random.randint(0, len(validLineAngles))
                return int(validLineAngles[angleIdx])

            lineLengths = [3, 5, 7, 9]
            lineTypes = ["right", "left", "full"]
            lineLengthIdx = np.random.randint(0, len(lineLengths))
            lineTypeIdx = np.random.randint(0, len(lineTypes))
            lineLength = lineLengths[lineLengthIdx]
            lineType = lineTypes[lineTypeIdx]
            lineAngle = randomAngle(lineLength)
            blurred_img = img
            for i in range(3):
                blurred_img[:, :, i] = np.asarray(LinearMotionBlur(img[:, :, i], lineLength, lineAngle, lineType))
            blurred_img = Image.fromarray(blurred_img, 'RGB')
            return blurred_img

        x, y = paste_location
        foreground = foreground_object.img
        foreground_mask = Image.fromarray(np.where(np.array(foreground_object.mask) != 0, 255, 0).astype('uint8'))
        background = self.image.copy()
        if blending == 'none':
            background.paste(foreground, (x, y), foreground_mask)
        elif blending == 'motion':
            background.paste(foreground, (x, y), foreground_mask)
            background = LinearMotionBlur3C(np.asarray(background))

        elif blending == 'poisson':
            offset = (y, x)
            img_mask = np.asarray(foreground_mask)
            img_src = np.asarray(foreground).astype(np.float64)
            img_target = np.asarray(background)
            img_mask, img_src, offset_adj \
                = create_mask(img_mask.astype(np.float64),
                              img_target, img_src, offset=offset)
            background_array = poisson_blend(img_mask, img_src, img_target,
                                             method='normal', offset_adj=offset_adj)
            background = Image.fromarray(background_array, 'RGB')
        elif blending == 'gaussian':
            background.paste(foreground, (x, y), Image.fromarray(
                cv2.GaussianBlur(np.asarray(foreground_mask), (5, 5), 2)))
        elif blending == 'box':
            background.paste(foreground, (x, y), Image.fromarray(
                cv2.blur(np.asarray(foreground_mask), (3, 3))))
        else:
            raise NotImplementedError
        self.image = background

    def to_COCO_ann(self) -> Iterator[tuple]:
        """ polygons, bbox, area """
        for id, category in self.instance_mask_id2category.items():
            mask = np.array(self.instance_mask)
            mask = np.where(mask == id, mask, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            polygons = []
            for contour in contours:
                if contour.size >= 6:
                    polygons.append(contour.flatten().tolist())
            if len(polygons) == 0:
                continue
            RLEs = cocomask.frPyObjects(polygons, mask.shape[0], mask.shape[0])
            RLE = cocomask.merge(RLEs)
            area = cocomask.area(RLE)
            [x, y, w, h] = cv2.boundingRect(mask)
            yield polygons, [x, y, w, h], float(area), category