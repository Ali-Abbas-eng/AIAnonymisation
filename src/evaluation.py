import argparse
import os
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.model_zoo import get_config_file
import json
from detectron2.utils.logger import setup_logger
setup_logger()


def evaluate(yaml_url: str or os.PathLike,
             model_weights: str or os.PathLike,
             test_data_file: str or os.PathLike,
             output_dir: str or os.PathLike,
             device: str):
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # Set device to CPU
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

    device = torch.device(device)

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
    cfg = get_cfg()
    base_name = yaml_url.replace('.yaml', '').replace('COCO-Detection/', '')
    cfg.merge_from_file(get_config_file(yaml_url))
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = str(device)
    predictor = DefaultPredictor(cfg)

    # Evaluate model on test dataset
    evaluator = COCOEvaluator(test_dataset_name, output_dir=os.path.join(output_dir, base_name, 'test'))
    val_loader = build_detection_test_loader(cfg, test_dataset_name)
    inference_on_dataset(predictor.model, val_loader, evaluator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_url', type=str, required=True)
    parser.add_argument('--model_weights', type=str, required=True)
    parser.add_argument('--test_data_file', type=str, default=os.path.join('data', 'test.json'))
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--device', type=str, default='cuda')
    args = vars(parser.parse_args())
    evaluate(**args)
