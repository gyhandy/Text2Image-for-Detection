import os
import random
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict, ListConfig
import numpy as np

from paster import Paster

from logging import Logger, getLogger

logger = getLogger(__file__)
@hydra.main(config_path="config", config_name="config", version_base="1.2")
def paste(cfg: DictConfig):
    assert cfg.get("dataset") and cfg.get("name")
    assert cfg['paster'].get("select_prob") in ["uniform", "balanced"]
    with open_dict(cfg):
        if not cfg.get("dataset_dir"):
            if cfg.dataset == "VOC":
                cfg.dataset_dir = Path(cfg.work_dir).parent / "data" / "voc2012"
            else:
                cfg.dataset_dir = Path(cfg.work_dir).parent / "data" / "COCO2017"
        else:
            cfg.dataset_dir = Path(cfg.dataset_dir)
        cfg.output_dir = Path(cfg.output_dir)
        if cfg.get("debug") and cfg.output_dir.exists():
            shutil.rmtree(cfg.output_dir)
        os.makedirs(cfg.output_dir, exist_ok=True)

    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    paster = Paster(
        label2id=cfg.ds.label2id,
        out_size=cfg.paster.out_size,
        repeat_background=cfg.paster.repeat_background,
        select_prob=cfg.paster.select_prob,
        random_paste=cfg.paster.use_random_paste,
    )
    if (cfg.output_dir / "paster.pt").exists(): # if paster is saved, load it instead of creating a new one
        paster = paster.from_save(cfg.output_dir)
    else:
        #### foreground
        if cfg.get("fg_real"):
            # from utils import read_real_VOC
            paster.foregrounds.extend(hydra.utils.call(cfg.fg_real))
        if cfg.get("fg_syn"):
            # from utils import read_entity_foregrounds
            paster.foregrounds.extend(hydra.utils.call(cfg.fg_syn))
        #### background
        if cfg.get("bg_real"):
            # from utils import read_real_VOC
            paster.backgrounds.extend(hydra.utils.call(cfg.bg_real))
        if cfg.get("bg_syn"):
            # from utils import read_dalle_backgrounds
            assert isinstance(cfg.bg_syn, ListConfig)
            for syn_cfg in cfg.bg_syn:
                paster.backgrounds.extend(hydra.utils.call(syn_cfg))

        paster.save(cfg.output_dir)

    paster.validate()
    if cfg.paster.debug:
        random.shuffle(paster.backgrounds)
        random.shuffle(paster.foregrounds)
        paster.truncate(slice(0, 40))
        cfg.paster.max_workers = 1
        cfg.output_dir /= "debug"
        if cfg.output_dir.exists():
            shutil.rmtree(cfg.output_dir)
    else:
        random_str = "-random" if cfg.use_random_paste else ""
        prob_str = '' if cfg.select_prob == 'uniform' else "-balanced"
        num_cut = cfg.num_cut_images if cfg.num_cut_lowerbound is None else f"{cfg.num_cut_lowerbound}~{cfg.num_cut_images}"
        cfg.output_dir = cfg.output_dir \
                         / f"[{cfg.foreground},{cfg.background}]{random_str}{prob_str}[{cfg.repeat_each_image},{num_cut},{cfg.max_degree},{cfg.out_size}]"

    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.paster.get("start") and cfg.paster.get("to"):
        slice_idx = slice(cfg.paster.start, cfg.paster.to)
        paster.truncate(slice_idx)

    logger.info(f"size of background {len(paster)}; size of foreground {len(paster.foregrounds)}")
    logger.info(f"saving to {cfg.output_dir}")

    # TODO, move json
    shutil.copy(Path(os.getcwd()) / ".hydra" / "config.yaml", cfg.output_dir / "config.yaml")
    shutil.copy(Path(os.getcwd()) / ".hydra" / "overrides.yaml", cfg.output_dir / "overrides.yaml")
    paster.cut_and_paste(
        out_dir=cfg.output_dir,
        max_workers=cfg.paster.max_workers,
        num_cut_images=cfg.paster.num_cut_images, max_degree=cfg.paster.max_degree,
        num_cut_lowerbound=cfg.paster.num_cut_lowerbound,
        scale_factor=cfg.paster.scale_factor, center_translation_factor=cfg.paster.center_translation_factor,
        use_random_scaling=cfg.paster.use_random_scaling
    )


if __name__ == "__main__":
    paste()