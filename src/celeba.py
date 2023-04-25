import os
import json
from tqdm.auto import tqdm
from typing import Union, Callable
import gdown
import py7zr
import multivolumefile
import matplotlib.pyplot as plt
from data_tools import adaptive_resize, IMAGE_SIZE


def download_and_extract(download_directory: Union[str, os.PathLike] = os.path.join('../data', 'zipped'),
                         unzipped_directory: Union[str, os.PathLike] = os.path.join('../data', 'raw', 'CelebA')):
    """
    Downloads and extracts CelebA dataset from Google Drive.

    Args:
        download_directory (Union[str, os.PathLike]): Directory to save downloaded files. Defaults to 'data/zipped'.
        unzipped_directory (Union[str, os.PathLike]): Directory to extract files. Defaults to 'data/raw'.
    """
    # Download CelebA dataset from Google Drive
    url = 'https://drive.google.com/drive/folders/1eyJ52hX5KpniS9MB-MFXBRtJUDpeldqx?usp=share_link'
    gdown.download_folder(url=url,
                          output=download_directory,
                          quiet=False,
                          proxy=None,
                          speed=None,
                          use_cookies=True)
    os.makedirs(unzipped_directory, exist_ok=True)
    zipped_file_path = os.path.join(download_directory, 'img_celeba.7z')

    with multivolumefile.open(zipped_file_path, mode='rb') as target_archive:
        with py7zr.SevenZipFile(target_archive, 'r') as archive:
            archive.extractall(unzipped_directory)

    annotations_output_path = os.path.join(unzipped_directory, 'anno')
    os.makedirs(annotations_output_path)
    # download annotations
    gdown.download('https://drive.google.com/uc?id=19X0GE3kP6tNatS9kZ2-Ks2_OeeCtqeFI',
                   output=os.path.join(annotations_output_path, 'list_bbox_celeba.txt'))


def generate_dataset_registration_info(data_directory: str or os.PathLike,
                                       annotations_file: str or os.PathLike,
                                       info_path: str or os.PathLike,
                                       create_record: Callable,
                                       pre_process: Callable):
    """
    Generates dataset records (list of dictionaries) and saves them to a json file.

    Args:
        data_directory (str or os.PathLike): The directory where the images are stored.
        annotations_file (str or os.PathLike): The path to the annotations file.
        info_path (str or os.PathLike): The path to the information file.
        create_record (Callable): a function to creat information object (dictionary) for one data file.
        pre_process (Callable): pre-processing function to be used before writing the final dataset to disk.

    Returns:
        None
    """
    # Read the annotations file and split it into lines
    annotations = open(annotations_file, 'r').read().split('\n')[2:]

    # Remove empty lines and extra spaces
    annotations = [item.replace("  ", ' ') for item in annotations if len(item) > 2]

    # Convert each line into a list of integers and strings
    annotations = [[int(item) if item.isnumeric() else item for item in line.split()] for line in annotations]

    with tqdm(total=len(annotations), desc='Generating Data From CelebA') as progress_bar:
        def generate_data_point(index, example):
            # Get the image path
            image_path = os.path.join(data_directory, 'img_celeba', example[0])

            # Get the bounding box coordinates
            x_min = example[1]
            y_min = example[2]
            x_max = example[1] + example[3]
            y_max = example[2] + example[4]
            bboxes = [x_min, y_min, x_max, y_max]
            if pre_process is not None:
                bboxes = pre_process(image_path, bboxes)
            # Create a record for the example
            example_record = create_record(image_path=image_path,
                                           index=index,
                                           bounding_boxes=bboxes,
                                           category_id=0)
            progress_bar.update()
            # progress_bar.update()
            return example_record
        # Create an empty list to store the dataset records
        dataset_records = [generate_data_point(index, line) for index, line in enumerate(annotations)]

    # Save the dataset records to a json file
    json.dump(dataset_records, open(info_path, 'w'))
