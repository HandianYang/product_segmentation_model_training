#!/usr/bin/env python3

"""
display_result.py

Description:
    This script demonstrates the segmentation results of a trained model on an
    example image. It uses Detectron2's visualization tools to display the
    results and save them to PNG files.

Usage:
    Display the segmentation results using the newest dataset version:

        python3 display_result.py

    Display the segmentation results using a specific dataset version:

        python3 display_result.py --version {DATASET_VERSION}

    where {DATASET_VERSION} is one of the following:
        - v1
        - v2
        - v3
"""

import argparse
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description="Demonstrate segmentation results with different dataset versions.")
    parser.add_argument("--version", type=str, default="v3", choices=["v1", "v2", "v3"],
                        help="Dataset version to use (default: v3)")
    return parser.parse_args()

def setup_predictor(version):
    output_dir = Path(__file__).resolve().parent.parent / "output" / version
    cfg = get_cfg()
    cfg.merge_from_file(str(output_dir / "config.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.WEIGHTS = str(output_dir / "model_final.pth")
    return DefaultPredictor(cfg), cfg

def process_image(predictor, cfg):
    image_path = Path(__file__).resolve().parent / "example_image.JPG"
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.imread(str(image_path))
    outputs = predictor(image)
    return image, outputs

def visualize_and_save(image, outputs, cfg, version):
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_path = Path(__file__).resolve().parent / f"example_output_{version}.png"
    cv2.imwrite(str(output_path), out.get_image()[:, :, ::-1])

def main():
    args = parse_arguments()
    predictor, cfg = setup_predictor(args.version)
    image, outputs = process_image(predictor, cfg)
    visualize_and_save(image, outputs, cfg, args.version)

if __name__ == "__main__":
    main()