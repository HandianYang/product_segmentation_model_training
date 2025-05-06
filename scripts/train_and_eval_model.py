#!/usr/bin/env python3

"""
train_and_eval_model.py

Description:
    This script trains and evaluates a Detectron2 model on a COCO-style dataset.
    It allows for different dataset versions (v1, v2, v3) to be specified via 
    command-line arguments. The script registers the dataset, sets up the 
    configuration, trains the model, evaluates it, and saves the configuration 
    file.

Usage:
    Train and evaluate the model using the newest dataset version:

        python3 train_and_eval_model.py

    Train and evaluate the model using a specific dataset version:

        python3 train_and_eval_model.py --version {DATASET_VERSION}

    where {DATASET_VERSION} is one of the following:
        - v1
        - v2
        - v3
"""

import argparse
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate model with different dataset versions.")
    parser.add_argument("--version", type=str, default="v3", choices=["v1", "v2", "v3"],
                        help="Dataset version to use (default: v3)")
    return parser.parse_args()

def register_datasets(dataset_root):
    register_coco_instances("supermarket_product_dataset_train", {},
                            str(dataset_root / "train" / "annotations_without_supercategory.coco.json"),
                            str(dataset_root / "train"))
    register_coco_instances("supermarket_product_dataset_val", {},
                            str(dataset_root / "valid" / "annotations_without_supercategory.coco.json"),
                            str(dataset_root / "valid"))
    register_coco_instances("supermarket_product_dataset_test", {},
                            str(dataset_root / "test" / "annotations_without_supercategory.coco.json"),
                            str(dataset_root / "test"))

def setup_config(version):
    cfg = get_cfg()
    cfg.merge_from_file(str(Path(__file__).resolve().parent.parent / "config" / "mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("supermarket_product_dataset_train",)
    cfg.DATASETS.TEST = ("supermarket_product_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.OUTPUT_DIR = str(Path(__file__).resolve().parent.parent / "output" / version)
    cfg.MODEL.WEIGHTS = str(Path(__file__).resolve().parent.parent / "backbone" / "R-50.pkl")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return cfg

def train_model(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return trainer

def evaluate_model(cfg, trainer):
    evaluator = COCOEvaluator("supermarket_product_dataset_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "supermarket_product_dataset_test")
    inference_on_dataset(trainer.model, val_loader, evaluator)

def save_config(cfg):
    with open(cfg.OUTPUT_DIR + "/config.yaml", "w") as f:
        f.write(cfg.dump())

def main():
    args = parse_arguments()
    dataset_root = Path(__file__).resolve().parent.parent / "datasets" / f"supermarket_products_{args.version}"
    register_datasets(dataset_root)
    cfg = setup_config(args.version)
    trainer = train_model(cfg)
    evaluate_model(cfg, trainer)
    save_config(cfg)

if __name__ == "__main__":
    main()
