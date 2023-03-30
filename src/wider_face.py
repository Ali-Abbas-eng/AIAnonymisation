import itertools
import data_tools
import os
from tqdm.auto import tqdm
from zipfile import ZipFile
import json
import gdown
from typing import Union, Callable


def download_and_extract(download_directory: Union[str, os.PathLike],
                         unzipped_directory: Union[str, os.PathLike]):
    """
    Downloads and extracts CCPD2019 dataset from Google Drive.

    Args:
        download_directory (Union[str, os.PathLike]): Directory to save downloaded files. Defaults to 'data/zipped'.
        unzipped_directory (Union[str, os.PathLike]): Directory to extract files. Defaults to 'data/raw/CCPD2019'.
    """
    # create the directories
    os.makedirs(download_directory, exist_ok=True)
    os.makedirs(unzipped_directory, exist_ok=True)

    def unzip(file, unzipped_dir):
        os.makedirs(unzipped_dir, exist_ok=True)
        with ZipFile(file, 'r') as zip_file:
            for member in tqdm(zip_file.namelist()):
                zip_file.extract(member, unzipped_dir)

    # Download WIDER FACE train dataset from Google Drive
    zipped_train_file_path = os.path.join(download_directory, 'WIDER_train.zip')
    gdown.download(url='https://drive.google.com/uc?id=1w6lLpq6Sh10okRA6bSBqcDEDb-2fK_nc',
                   output=zipped_train_file_path)
    # extract downloaded file
    unzip(zipped_train_file_path, os.path.join(unzipped_directory, 'WIDER_train'))

    # Download WIDER FACE val dataset from Google Drive
    zipped_val_file_path = os.path.join(download_directory, 'WIDER_val.zip')
    gdown.download(url='https://drive.google.com/uc?id=1wb5jtFTHd9yBZpYpUVO50hb5ofa2NOm3',
                   output=zipped_val_file_path)
    # extract downloaded file
    unzip(zipped_val_file_path, os.path.join(unzipped_directory, 'WIDER_val'))

    # Download WIDER FACE annotations file
    zipped_annotations_files = os.path.join(download_directory, 'wider_face_split.zip')
    gdown.download(url='https://drive.google.com/uc?id=1KcRtgcLprJBnhKpkEkC-FwBdXrdb_nsv',
                   output=zipped_val_file_path)

    # extract downloaded file
    unzip(zipped_val_file_path, unzipped_directory)


def generate_dataset_registration_info(data_directory: str or os.PathLike,
                                       annotations_file: str or os.PathLike,
                                       create_record: Callable):
    """
    This function generates dataset registration information from the annotations file and the data directory.

    Args:
        data_directory (str or os.PathLike): The directory containing the data.
        annotations_file (str or os.PathLike): The annotations file.
        create_record (Callable): a function to creat information object (dictionary) for one data file.

    Returns:
        dataset_dicts (list): A list of dictionaries containing the dataset registration information.
    """
    # Read the annotations file and split it by new line
    annotations = open(annotations_file, 'r').read().split('\n')

    # Initialize the dataset dictionary and file id
    dataset_dicts = []
    file_id = 0

    # Loop through the annotations
    for index, line in tqdm(enumerate(annotations), total=len(annotations)):
        # Check if the line ends with '.jpg'
        if line.endswith('.jpg'):
            # Get the file internal path and image path
            file_internal_path = os.path.join(*line.split('/'))
            image_path = os.path.join(data_directory, file_internal_path)

            # Get the number of images and bounding boxes
            num_faces = int(annotations[index + 1])
            bboxes = []
            for i in range(index + 2, index + 2 + num_faces):
                bbox = [int(coordinate) for coordinate in annotations[i].split()[:4]]
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bboxes.append(bbox)
            bboxes = list(itertools.chain.from_iterable(bboxes))

            # Create the record and append it to the dataset dictionary
            record = create_record(image_path=image_path,
                                   bounding_boxes=bboxes,
                                   index=file_id,
                                   category_id=0)
            dataset_dicts.append(record)

    return dataset_dicts


def write_data(data_directory_train: Union[str, os.PathLike],
               data_directory_valid: Union[str, os.PathLike],
               annotation_file_train: Union[str, os.PathLike],
               annotation_file_valid: Union[str, os.PathLike],
               create_record: Callable,
               info_path: str = data_tools.WIDER_FACE_INFORMATION_FILE):
    """
    This function writes the data to the information file.

    Args:
        data_directory_train (str or os.PathLike): The directory containing the training data.
        data_directory_valid (str or os.PathLike): The directory containing the validation data.
        annotation_file_train (str or os.PathLike): The training data annotations file.
        annotation_file_valid (str or os.PathLike): The validation data annotations file.
        create_record (Callable): a function to creat information object (dictionary) for one data file.
        info_path (str): The path of the information file.
    """
    # Generate the dataset registration information for the training and validation data
    data1 = generate_dataset_registration_info(data_directory=data_directory_train,
                                               annotations_file=annotation_file_train,
                                               create_record=create_record)
    data2 = generate_dataset_registration_info(data_directory=data_directory_valid,
                                               annotations_file=annotation_file_valid,
                                               create_record=create_record)

    # Combine the training and validation data
    data1.extend(data2)

    # Write the data to the information file
    json.dump(data1, open(info_path, 'w'))
