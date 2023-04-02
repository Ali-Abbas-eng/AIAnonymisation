import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import matplotlib.pyplot as plt
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


def run_evaluation(yaml_url: str,
                   weights_path: str or os.PathLike,
                   output_directory: str or os.PathLike,
                   evaluation_dataset_name: str):
    """
    Initializes the model and evaluates it on the validation dataset.

    Args:
        yaml_url (str): URL to the YAML configuration file.
        weights_path (str or os.PathLike): Path to the model weights file.
        output_directory (str or os.PathLike): Path to the output directory.
        evaluation_dataset_name (str): Name of the dataset to be used for evaluation.

    Returns:
        None
    """
    # Load configuration from YAML file
    cfg = get_cfg()
    cfg.merge_from_file(yaml_url)

    # Load model weights
    cfg.MODEL.WEIGHTS = weights_path

    # Create predictor object for making predictions on test dataset
    predictor = DefaultPredictor(cfg)

    # Create evaluator object for evaluating model performance on test dataset
    evaluator = COCOEvaluator(evaluation_dataset_name, output_dir=output_directory)

    # Build test dataset loader
    val_loader = build_detection_test_loader(cfg, dataset=evaluation_dataset_name)

    # Evaluate model performance on test dataset and save metrics to file
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)

    # Plot AP and AR curves using metrics from evaluation
    ap = metrics["bbox"]["AP"]
    ar = metrics["bbox"]["AR"]
    plt.plot(ap)
    plt.plot(ar)
    plt.xlabel("Iteration")
    plt.ylabel("Metric")
    plt.legend(["AP", "AR"])
    plt.show()
