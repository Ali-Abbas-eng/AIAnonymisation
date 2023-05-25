import shutil
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
supported_formats = {
    'image': ['png', 'jpg', 'jpeg'],
    'video': ['avi', 'mp4', 'mov']
}


def get_predictor(network: str or os.PathLike,
                  model_weights: str or os.PathLike = None,
                  test_data_file: str or os.PathLike = os.path.join('data', 'test.json'),
                  device: str = 'cuda',
                  output_dir: str = 'temp',
                  threshold: float = 0.7,
                  return_cfg: bool = False) -> DefaultPredictor:
    """Returns a DefaultPredictor object for the given model.

    Args:
        network (str or os.PathLike): The URL or path of the YAML configuration file for the model.
        model_weights (str or os.PathLike): The path to the model weights file.
        test_data_file (str or os.PathLike): The path to the file containing the test data.
        output_dir (str or os.PathLike): The path to the output directory.
        device (str): The device on which the model should run.
        threshold (float): The score threshold for predictions.
        return_cfg (bool): Whether to return the ConfigurationNode Object (used in the evaluation code).

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

    if model_weights is None:
        model_weights = os.path.join('output', network, 'model_final.pth')
    # Load the test data and register it with Detectron2
    test_data = json.load(open(test_data_file, 'r'))
    test_dataset_name = "test_data"
    try:
        DatasetCatalog.register(test_dataset_name, lambda: test_data)
    except AssertionError:
        print('Data is already registered')

    # Set the metadata for the test dataset
    MetadataCatalog.get(test_dataset_name).set(thing_classes=["FACE", "Car Plate"])

    # Set the configuration for the model
    cfg = get_cfg(network_base_name=network,
                  yaml_url=f'COCO-Detection/{network}.yaml',
                  train_datasets=(),
                  test_datasets=(),
                  output_directory=output_dir)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = str(device)
    # Create a DefaultPredictor object with the given configuration and return it
    predictor = DefaultPredictor(cfg)
    if return_cfg:
        return predictor, cfg
    return predictor


def predict_on_video(video_object: str or os.PathLike,
                     output_path: str or os.PathLike,
                     predictor: DefaultPredictor,
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

    for _ in tqdm(range(frame_count), desc=f'Inference on Video {video_object}'):
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


def predict_on_directory(directory: str or os.PathLike or list,
                         output_directory: str or os.PathLike,
                         predictor: DefaultPredictor,
                         metadata: MetadataCatalog,
                         scale: float = 1.):
    """
    Perform object detection on all image files in a directory and save the output to another directory.

    Args:
        directory (str or os.PathLike or list): The directory or list of directories containing the input image files.
        output_directory (str or os.PathLike): The directory to save the output images.
        predictor (DefaultPredictor): The object detection model predictor.
        metadata (MetadataCatalog): The metadata for the input images.
        scale (float): The scale factor for the images.

    Returns:
        None
    """
    # If the directory argument is a string, convert it to a list of file paths
    if type(directory) == str:
        directory = [os.path.join(directory, file) for file in os.listdir(directory)]
    if len(directory) > 0:
        base_dir = directory[0][:directory[0].index(os.path.basename(directory[0]))]
        # Iterate over each file in the directory and perform object detection on each image
        for file in tqdm(directory, desc=f'Performing Inference on Images at {base_dir}'):
            # Read in the image using matplotlib
            image = plt.imread(file)
            try:
                # Perform object detection on the image using the predictor and metadata
                output = inference_step(predictor, image, metadata, scale)
                # Save the output image to the output directory with the same filename as the input file
                plt.imsave(os.path.join(output_directory, os.path.basename(file)), output)
            except ValueError:
                pass


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


def supported(file, file_type):
    for extension in supported_formats[file_type]:
        if file.endswith(extension):
            return True
    return False


def predict_on_file(path: str or os.PathLike,
                    output_path: str or os.PathLike,
                    predictor: DefaultPredictor,
                    metadata: MetadataCatalog,
                    scale: float):
    # Set up the 'support error message'
    support_error = 'File type is not supported by the inference module'

    # Determine the file type
    file_type = 'image' if supported(path, 'image') else 'video' if supported(path, 'video') else exit(support_error)

    # Invoke the appropriate function w.r.t file type
    if file_type == 'video':
        predict_on_video(path, output_path, predictor, metadata, scale)
    elif file_type == 'image':
        # Read in the image using matplotlib
        image = plt.imread(path)
        # Perform object detection on the image using the predictor and metadata
        output = inference_step(predictor, image, metadata, scale)
        # Save the output image to the output directory with the same filename as the input file
        plt.imsave(output_path, output)


def inference_manager(network: str or os.PathLike or DefaultPredictor,
                      target_path: str or os.PathLike,
                      output_path: str or os.PathLike,
                      model_weights: str or os.PathLike = None,
                      test_data_file: str or os.PathLike = os.path.join('data', 'test.json'),
                      device: str = 'cuda',
                      cache_dir: str = 'temp',
                      threshold: float = 0.7,
                      scale: float = 1.0):
    """
    Perform object detection on a directory of images and videos, and save the results in another directory.
    Recursively traverses subdirectories as well.

    Args:
        network (str or os.PathLike or DefaultPredictor): The path to the model file or the predictor object.
        target_path (str or os.PathLike): The path to the directory containing the input files.
        output_path (str or os.PathLike): The path to the directory to save the output files.
        model_weights (str or os.PathLike, optional): The path to the model weights file. Defaults to None.
        test_data_file (str or os.PathLike, optional): The path to the test data file. Defaults to 'data/test.json'.
        device (str, optional): The device to run the inference on. Defaults to 'cuda'.
        cache_dir (str, optional): The directory to store intermediate files. Defaults to 'temp'.
        threshold (float, optional): The confidence threshold for object detection. Defaults to 0.7.
        scale (float, optional): The scale factor for the input images. Defaults to 1.0.

    Returns:
        None
    """
    # Fix the paths to make them consistent
    target_path = path_fixer(target_path)
    output_path = path_fixer(output_path)
    test_data_file = path_fixer(test_data_file)
    if model_weights:
        model_weights = path_fixer(model_weights)

    # Get the predictor object
    if not type(network) == DefaultPredictor:
        predictor = get_predictor(network=network,
                                  model_weights=model_weights,
                                  test_data_file=test_data_file,
                                  device=device,
                                  output_dir=cache_dir,
                                  threshold=threshold)
        # Remove the cache directory if it exists
        shutil.rmtree(cache_dir)
    else:
        predictor = network

    # Get the metadata for the input images
    metadata = MetadataCatalog.get('test_data')

    # In the case of a single file
    if os.path.isfile(target_path):
        predict_on_file(target_path, output_path, predictor, metadata, scale)
    else:
        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

        # Get the list of files and directories to process
        contents = os.listdir(target_path)
        images = [os.path.join(target_path, file) for file in contents if supported(file, file_type='image')]
        videos = [os.path.join(target_path, file) for file in contents if supported(file, file_type='video')]
        directories = [os.path.join(target_path, file) for file in contents if
                       os.path.isdir(os.path.join(target_path, file))]

        # Process the images
        predict_on_directory(images, output_path, predictor, metadata, scale)

        # Process the videos
        [predict_on_video(file, os.path.join(output_path, os.path.basename(file)), predictor, metadata, scale)
         for file in videos]

        # Process the subdirectories recursively
        [inference_manager(predictor,
                           os.path.join(subdir),
                           os.path.join(output_path, subdir.split(os.path.sep)[-1]),
                           model_weights,
                           test_data_file,
                           device,
                           cache_dir,
                           threshold,
                           scale)
         for subdir in directories]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=.7)
    parser.add_argument('--test_data_file', type=str, default=os.path.join('data', 'test.json'))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scale', type=float, default=1.)

    args = vars(parser.parse_args())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        inference_manager(**args)
