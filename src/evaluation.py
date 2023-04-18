import argparse
import os
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import json
from detectron2.utils.logger import setup_logger
import shutil
from training_utils import get_cfg
# Set up logger
setup_logger()


def evaluate(yaml_url: str or os.PathLike,
             model_weights: str or os.PathLike,
             test_data_file: str or os.PathLike,
             output_dir: str or os.PathLike,
             device: str):
    """
    Evaluate a trained model on a test dataset.

    Args:
        yaml_url (str or os.PathLike): The URL or path to the YAML file containing the model configuration.
        model_weights (str or os.PathLike): The path to the file containing the trained model weights.
        test_data_file (str or os.PathLike): The path to the JSON file containing the test data in COCO format.
        output_dir (str or os.PathLike): The path to the directory where the evaluation results will be saved.
        device (str): The device to use for evaluation ('cpu' or 'cuda').
    """
    if device == 'cpu':
        # Set environment variable to disable CUDA
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # Set device to CPU
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

    # Set device
    device = torch.device(device)

    # Remove existing output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Load test data in COCO format from a JSON file
    test_data = json.load(open(test_data_file, 'r'))
    test_dataset_name = "test_data"

    # Register test dataset
    try:
        DatasetCatalog.register(test_dataset_name, lambda: test_data)
    except AssertionError:
        print('Data is already registered')

    # Set the class names
    MetadataCatalog.get(test_dataset_name).set(thing_classes=["FACE", "LP"])

    # Load trained model
    cfg = get_cfg(network_base_name=yaml_url.replace('COCO-Detection/', '').replace('.yaml', ''),
                  yaml_url=yaml_url,
                  train_datasets=(),
                  test_datasets=(),
                  output_directory=output_dir)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    # Set output directory and device
    os.rmdir(cfg.OUTPUT_DIR)
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = str(device)

    # Create predictor and evaluator objects
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(test_dataset_name, output_dir=output_dir)

    # Evaluate model on test dataset
    val_loader = build_detection_test_loader(cfg, test_dataset_name)
    inference_on_dataset(predictor.model, val_loader, evaluator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_url', type=str, required=True)
    parser.add_argument('--model_weights', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_data_file', type=str, default=os.path.join('data', 'test.json'))
    parser.add_argument('--device', type=str, default='cuda')
    args = vars(parser.parse_args())
    evaluate(**args)
