import warnings
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
import os
import json
from detectron2.utils.logger import setup_logger
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from data_tools import path_fixer
from detectron2.utils.visualizer import Visualizer
import argparse
try:
    from training_utils import get_cfg
except ModuleNotFoundError:
    from src.training_utils import get_cfg
setup_logger()


def get_predictor(yaml_url: str or os.PathLike,
                  model_weights: str or os.PathLike,
                  test_data_file: str or os.PathLike,
                  device: str,
                  output_dir: str,
                  threshold: float) -> DefaultPredictor:
    """Returns a DefaultPredictor object for the given model.

    Args:
        yaml_url (str or os.PathLike): The URL or path of the YAML configuration file for the model.
        model_weights (str or os.PathLike): The path to the model weights file.
        test_data_file (str or os.PathLike): The path to the file containing the test data.
        output_dir (str or os.PathLike): The path to the output directory.
        device (str): The device on which the model should run.
        threshold (float): The score threshold for predictions.

    Returns:
        DefaultPredictor: A DefaultPredictor object for the given model.
    """

    # If device is CPU, set CUDA_VISIBLE_DEVICES to empty string and turn off gradient computation
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)

    # Set the device for the model
    device = torch.device(device)

    # Load the test data and register it with Detectron2
    test_data = json.load(open(test_data_file, 'r'))
    test_dataset_name = "test_data"
    try:
        DatasetCatalog.register(test_dataset_name, lambda: test_data)
    except AssertionError:
        print('Data is already registered')

    # Set the metadata for the test dataset
    MetadataCatalog.get(test_dataset_name).set(thing_classes=["FACE", "LP"])

    # Set the configuration for the model
    cfg = get_cfg(network_base_name=yaml_url.replace('COCO-Detection/', '').replace('.yaml', ''),
                  yaml_url=yaml_url,
                  train_datasets=(),
                  test_datasets=(),
                  output_directory=output_dir)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = str(device)
    # Create a DefaultPredictor object with the given configuration and return it
    predictor = DefaultPredictor(cfg)

    return predictor


def predict_on_video(video_object: str or os.PathLike or np.ndarray,
                     predictor: DefaultPredictor,
                     output_path: str or os.PathLike,
                     metadata: MetadataCatalog,
                     scale: float = 1.):
    """
    Applies object detection on a video and saves the resulting video to the specified output path.

    Args:
        video_object (str or os.PathLike or np.ndarray): The path to the input video file or a numpy array containing the frames.
        predictor (DefaultPredictor): The object detection model to use.
        output_path (str or os.PathLike): The path to save the resulting video file.
        metadata (MetadataCatalog): Metadata about the input data.
        scale (float, optional): The scaling factor to apply to the input frames.

    Returns:
        cv2.VideoCapture: A handle to the input video.
    """

    # Open the video capture object and get the video properties
    # noinspection PyUnresolvedReferences
    cap = cv2.VideoCapture(video_object)
    # noinspection PyUnresolvedReferences
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # noinspection PyUnresolvedReferences
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # noinspection PyUnresolvedReferences
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # noinspection PyUnresolvedReferences
    # Create a video writer object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # noinspection PyUnresolvedReferences
    video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    for _ in tqdm(range(frame_count), desc='Inference'):
        # Read the current frame from the input video
        _, im = cap.read()
        im = im[:, :, ::-1]

        # Apply the object detection model to the current frame
        new_frame = inference_step(predictor, im, metadata, scale)

        # Write the resulting frame to the output video
        video_writer.write(new_frame[:, :, ::-1])

    # Release the video writer object
    video_writer.release()

    # Clean up resources
    del predictor


def infer(target_path: str or os.PathLike or np.ndarray,
          output_path: str or os.PathLike,
          yaml_url: str or os.PathLike,
          model_weights: str or os.PathLike,
          test_data_file: str or os.PathLike,
          device: str,
          scale: float = 1.,
          threshold: float = .7):
    """
    Infers object detection from given input path and saves the output to output_path.

    Args:
    path (str or os.PathLike or np.ndarray): The input file path or array of images
    output_path (str or os.PathLike): The output file path.
    yaml_url (str or os.PathLike): The url or path to the yaml file.
    model_weights (str or os.PathLike): The path to the model weights file.
    test_data_file (str or os.PathLike): The path to the test data file.
    device (str): The device to run the inference on.
    scale (float, optional): The scale factor for the image size. Default is 1.
    threshold (float, optional): The detection threshold. Default is 0.7.
    """
    target_path = path_fixer(target_path)
    output_path = path_fixer(output_path)
    test_data_file = path_fixer(test_data_file)
    model_weights = path_fixer(model_weights)
    # Get predictor object
    predictor = get_predictor(yaml_url=yaml_url,
                              model_weights=model_weights,
                              test_data_file=test_data_file,
                              device=device,
                              output_dir='temp',
                              threshold=threshold)

    # Get metadata
    metadata = MetadataCatalog.get("test")

    # Check if the input path exists
    if not os.path.exists(target_path):
        raise FileNotFoundError(f'No such file or directory: {target_path}')

    # Check if the input path is a directory
    if os.path.isdir(target_path):
        files = [file for file in os.listdir(target_path) if os.path.isfile(file)]
        os.makedirs(output_path)
        predict_on_directory(directory=target_path,
                             predictor=predictor,
                             output_directory=output_path,
                             metadata=metadata,
                             scale=scale)

    # Check if the input path is a video file
    if os.path.isfile(target_path):
        extension = target_path[-3:]
        if extension in ['mov', 'mp4', 'avi']:
            predict_on_video(video_object=target_path,
                             output_path=output_path,
                             metadata=metadata,
                             predictor=predictor)
        # Check if the input path is an image file
        if target_path[-4:] in ['png', 'jpg']:
            image = plt.imread(target_path)
            output = inference_step(predictor, image, metadata, scale)
            plt.imsave(output_path, output)


def predict_on_directory(directory: str or os.PathLike or np.ndarray,
                         predictor: DefaultPredictor,
                         output_directory: str or os.PathLike,
                         metadata: MetadataCatalog,
                         scale: float = 1.):
    """
    Perform object detection on all image files in a directory and save the output to another directory.

    Args:
        directory (str or os.PathLike): The directory containing the input image files.
        predictor (DefaultPredictor): The object detection model predictor.
        output_directory (str or os.PathLike): The directory to save the output images.
        metadata: The metadata for the input images.
        scale (float): The scale factor for the images.

    Returns:
        None
    """
    for file in tqdm(os.listdir(directory)):
        image = plt.imread(os.path.join(directory, file))
        output = inference_step(predictor, image, metadata, scale)
        plt.imsave(os.path.join(output_directory, file), output)


def inference_step(predictor, image, metadata, scale):
    """
    Performs a single step of inference on a given image using the given predictor.

    Args:
        predictor (DefaultPredictor): The predictor to use for inference.
        image (np.ndarray): The image to perform inference on.
        metadata: The metadata for the image.
        scale (float): The scale of the image.

    Returns:
        np.ndarray: The output image after inference has been performed.
    """
    # Get the output predictions for the image
    outputs = predictor(image)

    # Visualize predictions
    visualizer = Visualizer(image, metadata=metadata, scale=scale)
    out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

    # Convert the output image to a numpy array and return it
    out = np.array(out.get_image(), dtype='uint8')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--yaml_url', type=str, required=True)
    parser.add_argument('--model_weights', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--test_data_file', type=str, default=os.path.join('data', 'test.json'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scale', type=float, default=1.)

    args = vars(parser.parse_args())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        infer(**args)
