import argparse
import os
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import json
from detectron2.utils.logger import setup_logger
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import shutil
from detectron2.utils.visualizer import Visualizer

try:
    from training_utils import get_cfg
    from data_tools import read_video
except ModuleNotFoundError:
    from src.training_utils import get_cfg
    from src.data_tools import read_video
setup_logger()


def get_predictor(yaml_url: str or os.PathLike,
                  model_weights: str or os.PathLike,
                  test_data_file: str or os.PathLike,
                  output_dir: str or os.PathLike,
                  device: str):
    """
    Returns a predictor object for object detection using a trained model.

    Args:
        yaml_url (str or os.PathLike): The URL or path to the YAML file containing the model configuration.
        model_weights (str or os.PathLike): The path to the file containing the trained model weights.
        test_data_file (str or os.PathLike): The path to the JSON file containing the test data in COCO format.
        output_dir (str or os.PathLike): The path to the directory where the output will be saved.
        device (str): The device to use for prediction ('cpu' or 'cuda').

    Returns:
        DefaultPredictor: A predictor object for object detection.
    """
    # Set device to CPU if specified
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

    # Set device
    device = torch.device(device)

    # Remove existing output directory
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

    # Set model weights and other parameters
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    os.rmdir(cfg.OUTPUT_DIR)
    cfg.OUTPUT_DIR = output_dir
    cfg.MODEL.DEVICE = str(device)

    # Create predictor object
    predictor = DefaultPredictor(cfg)
    os.rmdir(cfg.OUTPUT_DIR)
    return predictor


def evaluate(yaml_url: str or os.PathLike,
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
    predictor = get_predictor(yaml_url=yaml_url,
                              model_weights=model_weights,
                              test_data_file=test_data_file,
                              output_dir=output_dir,
                              device=device)

    # Evaluate model on test dataset
    evaluator = COCOEvaluator('test', output_dir=output_dir)
    val_loader = build_detection_test_loader(dataset='test')
    inference_on_dataset(predictor.model, val_loader, evaluator)


def predict_on_video(video_object: str or os.PathLike or np.ndarray,
                     output_path: str or os.PathLike,
                     yaml_url: str or os.PathLike,
                     model_weights: str or os.PathLike,
                     test_data_file: str or os.PathLike,
                     output_dir: str or os.PathLike,
                     device: str,
                     scale: float = 1.,
                     generator: bool = False):
    """
    Runs object detection on a video and saves the output.

    Args:
        video_object (str or os.PathLike or np.ndarray): The path to the video file or a numpy array containing the video frames.
        output_path (str or os.PathLike): The path to the file where the output video will be saved.
        yaml_url (str or os.PathLike): The URL or path to the YAML file containing the model configuration.
        model_weights (str or os.PathLike): The path to the file containing the trained model weights.
        test_data_file (str or os.PathLike): The path to the JSON file containing the test data in COCO format.
        output_dir (str or os.PathLike): The path to the directory where the output will be saved.
        device (str): The device to use for prediction ('cpu' or 'cuda').
        scale (float): the ratio of the returned video to the original video
    """
    # Get predictor object
    predictor = get_predictor(yaml_url=yaml_url,
                              model_weights=model_weights,
                              test_data_file=test_data_file,
                              output_dir=output_dir,
                              device=device)
    # Load video frames
    if type(video_object) == str:
        buf = cap = cv2.VideoCapture(video_object)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    elif type(video_object) == np.ndarray:
        buf = video_object
        frame_count = len(video_object)
        frame_height = video_object.shape[1]
        frame_width = video_object.shape[2]

    else:
        raise TypeError(f'Data of type {type(video_object)} is not supported, supported types are str and np.ndarray')
    # Save result as video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    # Get metadata
    metadata = MetadataCatalog.get("test")
    metadata.set(thing_classes=["FACE", "Lic. Plate"])

    # Run prediction on each frame
    result = []

    for i in tqdm(range(frame_count), desc='Inference'):
        _, im = cap.read()
        outputs = predictor(im[:, :, ::-1])

        # Visualize predictions
        v = Visualizer(im,
                       metadata=metadata,
                       scale=1.0
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Save frame with predictions
        new_frame = out.get_image()
        new_frame = np.array(new_frame, dtype='uint8')
        video_writer.write(new_frame)

    video_writer.release()
    del predictor
    return result


def predict_on_directory(images_directory: str or os.PathLike or np.ndarray,
                         output_directory: str or os.PathLike,
                         yaml_url: str or os.PathLike,
                         model_weights: str or os.PathLike,
                         test_data_file: str or os.PathLike,
                         output_dir: str or os.PathLike,
                         device: str,
                         scale: float = 1.):
    # Get predictor object
    predictor = get_predictor(yaml_url=yaml_url,
                              model_weights=model_weights,
                              test_data_file=test_data_file,
                              output_dir=output_dir,
                              device=device)
    # Get metadata
    metadata = MetadataCatalog.get("test")
    for file in tqdm(os.listdir(images_directory)):
        if os.path.isdir(file) or os.path.exists(os.path.join(output_directory, file)):
            continue
        im = plt.imread(os.path.join(images_directory, file))
        outputs = predictor(im)

        # Visualize predictions
        v = Visualizer(im,
                       metadata=metadata,
                       scale=1.0
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # Save frame with predictions
        new_frame = out.get_image()
        plt.imsave(os.path.join(output_directory, file), new_frame)

    shutil.make_archive(output_directory, 'zip', output_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_url', type=str, required=True)
    parser.add_argument('--model_weights', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_data_file', type=str, default=os.path.join('data', 'test.json'))
    parser.add_argument('--device', type=str, default='cuda')
    args = vars(parser.parse_args())
    evaluate(**args)
