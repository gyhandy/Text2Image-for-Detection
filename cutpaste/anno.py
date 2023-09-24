from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import ujson as json
from PIL import Image
from pycocotools import mask as cocomask


class Anno:
    label2id: dict
    id2label: dict

    @abstractmethod
    def objects(self):
        raise NotImplementedError

    @abstractmethod
    def create_mask(self, for_object: Optional[int] = None) -> Image.Image:
        raise NotImplementedError

    @abstractmethod
    def create_instance_mask(self) -> Tuple[Image.Image, dict]:
        raise NotImplementedError

    @staticmethod
    def factory(anno_path, seg_img_path):
        if anno_path is None:
            return EntityAnno(seg_img_path)
        elif seg_img_path is None:
            return COCOAnno(anno_path)
        return VOCAnno(anno_path, seg_img_path)


class VOCAnno(Anno):
    def __init__(self, anno_path, seg_img_path):
        import xml.etree.ElementTree as ET
        self.anno_path = anno_path
        self.anno = ET.parse(anno_path).getroot()
        self.seg_img_path = seg_img_path

    def size(self):
        size = self.anno.find("size")
        height, width = size.find("./height").text, size.find("./width").text
        return int(height), int(width)

    def filename(self) -> str:
        return self.anno.find("filename").text

    def objects(self):
        objects = self.anno.findall("object")
        # hardcode, remove wrong seg annotation
        if "2009_005069" in self.anno_path:
            objects = objects[:-1]
        return objects

    def create_mask(self, for_object: Optional[int] = None):
        """
        create boolean mask with same shape as .size()
        gt (is object) is positive, dummy is 0
        if for_object = None, OR all mask
        else, mask for this specific object (0 if dummy, positive for this category)
        """
        # consists of: objects (object number in anno), 0 (dummy bg), 255 (white mask outline)
        seg_mask = np.array(Image.open(self.seg_img_path))
        objects = self.objects()
        if for_object is None:
            ids = list(range(1, len(objects) + 1))
            categories = [
                object.find("./name").text
                for object in objects
            ]
            id2categoryid = {
                i: self.label2id[c]
                for i, c in zip(ids, categories)
            }
            # plus mapping to get dummy 255
            id2categoryid[0] = 0
            id2categoryid[255] = 0
            if len(np.unique(seg_mask)) != len(id2categoryid):
                # when seg is wrong, there are mismatch
                seg_mask = np.where(np.isin(seg_mask, list(id2categoryid)), seg_mask, 0)

            # rn if seg_mask == i, it's ith object, make it ith object's category
            mask = np.vectorize(id2categoryid.get)(seg_mask).astype('uint8')
            return Image.fromarray(mask)

        assert type(for_object) is int
        assert 1 <= for_object <= len(objects)
        id = for_object
        category = objects[id - 1].find("./name").text

        mask = np.where(seg_mask == id, self.label2id[category], 0).astype("uint8")
        return Image.fromarray(mask)

    def create_instance_mask(self):
        """
        instance mask where each non-dummy object is positive with id (starts from 1, NOT label id)
        0 if background dummy
        """
        seg_mask = np.array(Image.open(self.seg_img_path))
        instance_mask = np.where(np.isin(seg_mask, [0, 255]), 0, seg_mask).astype("uint8")
        objects = self.objects()
        ids = list(range(1, len(objects) + 1))
        categories = [
            object.find("./name").text
            for object in objects
        ]
        instance_mask_id2category = {
            i: self.label2id[c]
            for i, c in zip(ids, categories)
        }
        return Image.fromarray(instance_mask), instance_mask_id2category


class EntityAnno(Anno):
    def __init__(self, seg_img_path):
        # eg data/voc2012/entity_mask/bottle_mask/2009_000562.png
        self.seg_img_path = seg_img_path
        _, label, filename = seg_img_path.rsplit("/", 2)
        self.label = self.label2id[label.replace("_mask", "")]

    def objects(self):
        return [self.label]

    def create_mask(self, for_object: Optional[int] = None):
        # if for_object is not None:
        #     assert for_object in self.objects()
        # 0 or 255
        mask = np.array(Image.open(self.seg_img_path))
        mask = np.where(mask == 255, self.label, 0).astype("uint8")
        return Image.fromarray(mask)

    def create_instance_mask(self):
        instance_mask = np.array(Image.open(self.seg_img_path))
        # 0 or 255
        instance_mask = np.where(instance_mask == 255, 1, 0).astype("uint8")
        return Image.fromarray(instance_mask), {1: self.label}

class COCOAnno(Anno):
    def __init__(self, anno_path):
        with open(anno_path) as f:
            self.anno = json.load(f)
        
        self.id2annos = {
            id: []
            for id in self.objects()
        }
        for anno in self.anno["annotations"]:
            self.id2annos[anno["category_id"]].append(anno)

    def size(self):
        return int(self.anno['images']['height']), int(self.anno['images']['width'])

    def objects(self):
        return sorted(set([
            anno['category_id']
            for anno in self.anno["annotations"]
        ]))

    def create_mask(self, for_object: Optional[int] = None):
        if for_object: # i-th (1 based)
            category = self.objects()[for_object-1]
            annos = self.id2annos[category]
            mask = np.zeros(self.size(), dtype=int)
            for anno in annos:
                objs = cocomask.frPyObjects(anno["segmentation"], *self.size())
                binary_mask = cocomask.decode(objs) # (h, w, n) binary {0 (dummy), 1 (obj)} where n is \# disjoint anno
                if binary_mask.ndim == 2:
                    binary_mask = binary_mask[:, :, np.newaxis]
                for n in range(binary_mask.shape[-1]): #
                    mask[binary_mask[:, :, n] == 1] = category
                # binary_mask = np.where(binary_mask == 1, category, 0)
                # mask = np.ma.mask_or(mask, binary_mask)
            return Image.fromarray(mask.astype(np.uint8))
        
        mask = np.zeros(self.size(), dtype=int)
        for i, category in enumerate(self.objects(), 1):
            mask2 = self.create_mask(for_object = i)
            mask[np.array(mask2) == category] = category
        return Image.fromarray(mask.astype(np.uint8))

    def create_instance_mask(self):
        instance_mask = np.zeros(self.size(), dtype=int)
        instance_mask_id2category = {}
        for anno in self.anno["annotations"]:
            objs = cocomask.frPyObjects(anno["segmentation"], *self.size())
            binary_mask = cocomask.decode(objs) # (h, w) binary {0 (dummy), 1 (obj)}
            if binary_mask.ndim == 2:
                binary_mask = binary_mask[:, :, np.newaxis]
            next_id = len(instance_mask_id2category) + 1
            for n in range(binary_mask.shape[-1]): #
                instance_mask[binary_mask[:, :, n] == 1] = next_id
            instance_mask_id2category[next_id] = anno['category_id']

        return Image.fromarray(instance_mask.astype(np.uint8)), instance_mask_id2category