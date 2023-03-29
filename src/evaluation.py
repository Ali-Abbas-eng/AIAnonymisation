import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from data_tools import register_datasets
import matplotlib.pyplot as plt
import json


def initialize(yaml_url: str,
               weights_path: str or os.PathLike,
               output_directory: str or os.PathLike,
               data_directory: str or os.PathLike = 'data'):
    """
    Initializes the model and evaluates it on the validation dataset.

    Args:
        yaml_url (str): URL to the YAML configuration file.
        weights_path (str or os.PathLike): Path to the model weights file.
        output_directory (str or os.PathLike): Path to the output directory.
        data_directory (str or os.PathLike): Path to the data directory.

    Returns:
        None
    """
    try:
        # Register datasets if they haven't been registered already
        register_datasets(data_directory='data', thing_classes=['FACE', 'LP'])
    except AssertionError:
        pass

    # Load configuration from YAML file
    cfg = get_cfg()
    cfg.merge_from_file(yaml_url)

    # Load model weights
    cfg.MODEL.WEIGHTS = weights_path

    # Set dataset to use for testing
    cfg.DATASETS.TEST = ('val', )

    # Create predictor object for making predictions on test dataset
    predictor = DefaultPredictor(cfg)

    # Create evaluator object for evaluating model performance on test dataset
    evaluator = COCOEvaluator('val', cfg, False, output_dir=output_directory)

    # Build test dataset loader
    val_loader = build_detection_test_loader(cfg, "my_dataset_val")

    # Evaluate model performance on test dataset and save metrics to file
    inference_on_dataset(predictor.model, val_loader, evaluator)
    with open(os.path.join(output_directory, 'metrics.json'), "r") as f:
        metrics = json.load(f)

    # Plot AP and AR curves using metrics from evaluation
    ap = metrics["bbox"]["AP"]
    ar = metrics["bbox"]["AR"]
    plt.plot(ap)
    plt.plot(ar)
    plt.xlabel("Iteration")
    plt.ylabel("Metric")
    plt.legend(["AP", "AR"])
    plt.show()