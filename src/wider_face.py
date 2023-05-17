import itertools
import os
from tqdm.auto import tqdm
from zipfile import ZipFile
import json
import gdown
from typing import Union, Callable


def download_and_extract(download_directory: Union[str, os.PathLike],
                         unzipped_directory: Union[str, os.PathLike]):
    """
    Downloads and extracts WIDER FACE dataset from Google Drive.

    Args:
        download_directory (Union[str, os.PathLike]): Directory to save downloaded files. Defaults to 'data/zipped'.
        unzipped_directory (Union[str, os.PathLike]): Directory to extract files. Defaults to 'data/raw/WIDER_FACE'.
    """
    # create the directories
    os.makedirs(download_directory, exist_ok=True)
    os.makedirs(unzipped_directory, exist_ok=True)

    def unzip(file, unzipped_dir):
        os.makedirs(unzipped_dir, exist_ok=True)
        with ZipFile(file, 'r') as zip_file:
            # noinspection PyTypeChecker
            for member in tqdm(zip_file.namelist()):
                # noinspection PyTypeChecker
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
                   output=zipped_annotations_files)

    # extract downloaded file
    unzip(zipped_annotations_files, unzipped_directory)


def generate_dataset_registration_info(data_directory: str or os.PathLike,
                                       annotations_file: str or os.PathLike,
                                       create_record: Callable,
                                       pre_process: Callable):
    """
    This function generates dataset registration information from the annotations file and the data directory.

    Args:
        data_directory (str or os.PathLike): The directory containing the data.
        annotations_file (str or os.PathLike): The annotations file.
        create_record (Callable): a function to creat information object (dictionary) for one data file.
        pre_process (Callable): a function to pre-process the dataset before writing to disk.

    Returns:
        dataset_dicts (list): A list of dictionaries containing the dataset registration information.
    """
    # Read the annotations file and split it by new line
    annotations = open(annotations_file, 'r').read().split('\n')

    lines_of_interest = [line for line in annotations if line.endswith('.jpg')]
    with tqdm(total=len(lines_of_interest), desc='Generating Data From Wider Face') as progress_bar:
        def generate_data_point(row, image_id):
            # Check if the line ends with '.jpg'
            if row.endswith('.jpg'):
                idx = annotations.index(row)
                # Get the file internal path and image path
                file_internal_path = os.path.join(*row.split('/'))
                image_path = os.path.join(data_directory, file_internal_path)

                # Get the number of images and bounding boxes
                num_faces = int(annotations[idx + 1])
                bboxes = []
                for i in range(idx + 2, idx + 2 + num_faces):
                    bbox = [int(coordinate) for coordinate in annotations[i].split()[:4]]
                    bbox[2] += bbox[0]
                    bbox[3] += bbox[1]
                    bboxes.append(bbox)
                bboxes = list(itertools.chain.from_iterable(bboxes))
                if pre_process is not None:
                    bboxes = pre_process(image_path, bboxes)

                # Create the record and append it to the dataset dictionary
                record = create_record(image_path=image_path,
                                       bounding_boxes=bboxes,
                                       index=image_id,
                                       category_id=0)
                progress_bar.update()
                return record

        dataset_dicts = [generate_data_point(line, index) for index, line in enumerate(lines_of_interest)]
    return dataset_dicts


def write_data(data_directory_train: Union[str, os.PathLike],
               data_directory_valid: Union[str, os.PathLike],
               annotation_file_train: Union[str, os.PathLike],
               annotation_file_valid: Union[str, os.PathLike],
               create_record: Callable,
               info_path: str or os.PathLike,
               pre_process: Callable):
    """
    This function writes the data to the information file.

    Args:
        data_directory_train (str or os.PathLike): The directory containing the training data.
        data_directory_valid (str or os.PathLike): The directory containing the validation data.
        annotation_file_train (str or os.PathLike): The training data annotations file.
        annotation_file_valid (str or os.PathLike): The validation data annotations file.
        create_record (Callable): a function to creat information object (dictionary) for one data file.
        info_path (str): The path of the information file.
        pre_process (Callable): the function to pre-process dataset before writing to disk
    """
    # Generate the dataset registration information for the training and validation data
    data1 = generate_dataset_registration_info(data_directory=data_directory_train,
                                               annotations_file=annotation_file_train,
                                               create_record=create_record,
                                               pre_process=pre_process)
    data2 = generate_dataset_registration_info(data_directory=data_directory_valid,
                                               annotations_file=annotation_file_valid,
                                               create_record=create_record,
                                               pre_process=pre_process)

    # Combine the training and validation data
    data1.extend(data2)

    # Write the data to the information file
    json.dump(data1, open(info_path, 'w'))
