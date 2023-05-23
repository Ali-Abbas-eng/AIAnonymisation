import argparse
import os
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger
from inference import get_predictor

setup_logger()


def evaluate(network: str or os.PathLike,
             model_weights: str or os.PathLike,
             test_data_file: str or os.PathLike,
             output_dir: str or os.PathLike,
             device: str):
    """
    Evaluates a trained object detection model on a test dataset.

    Args:
        yaml_url (str or os.PathLike): The URL or path to the YAML file containing the model configuration.
        model_weights (str or os.PathLike): The path to the file containing the trained model weights.
        test_data_file (str or os.PathLike): The path to the JSON file containing the test data in COCO format.
        output_dir (str or os.PathLike): The path to the directory where the output will be saved.
        device (str): The device to use for prediction ('cpu' or 'cuda').
    """
    # Get predictor object
    predictor, cfg = get_predictor(network=network,
                                   model_weights=model_weights,
                                   test_data_file=test_data_file,
                                   output_dir=output_dir,
                                   device=device,
                                   return_cfg=True)

    # Evaluate model on test dataset
    evaluator = COCOEvaluator('test_data', output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, 'test_data')
    inference_on_dataset(predictor.model, val_loader, evaluator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--model_weights', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_data_file', type=str, default=os.path.join('data', 'test.json'))
    parser.add_argument('--device', type=str, default='cuda')
    args = vars(parser.parse_args())
    evaluate(**args)
