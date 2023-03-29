import os
import matplotlib.pyplot as plt
import itertools
from detectron2.structures import BoxMode
import requests
from tqdm.auto import tqdm
import json
from detectron2.data import DatasetCatalog, MetadataCatalog


CELEB_A_IMAGES_DIRECTORY = os.path.join('../data', 'raw', 'CelebA', 'img_celeba')
CELEB_A_ANNOTATIONS_FILE = os.path.join('../data', 'raw', 'CelebA', 'anno', 'list_bbox_celeba.txt')
CELEB_A_INFORMATION_FILE = os.path.join('../data', 'raw', 'CelebA', 'celeba_info.json')

WIDER_FACE_IMAGES_DIRECTORY_TRAIN = os.path.join('../data', 'raw', 'WIDER FACE', 'WIDER_train', 'images')
WIDER_FACE_ANNOTATIONS_FILE_TRAIN = os.path.join('../data', 'raw', 'WIDER FACE', 'wider_face_split',
                                                 'wider_face_train_bbx_gt.txt')
WIDER_FACE_IMAGES_DIRECTORY_VALID = os.path.join('../data', 'raw', 'WIDER FACE', 'WIDER_val', 'images')
WIDER_FACE_ANNOTATIONS_FILE_VALID = os.path.join('../data', 'raw', 'WIDER FACE', 'wider_face_split',
                                                 'wider_face_val_bbx_gt.txt')

WIDER_FACE_INFORMATION_FILE = os.path.join('../data', 'raw', 'WIDER FACE', 'wider_face.json')

CELEB_A_NUM_CANDIDATES = {
    'train': 30_000,
    'test': 10_000,
    'val': 10_000
}

WIDER_FACE_NUM_CANDIDATES = {
    'train': 10_000,
    'test': 2000,
    'val': 2000
}

CCPD_NUM_CANDIDATES = {
    'train': 40_000,
    'test': 12_000,
    'val': 12_000
}

FINAL_DATA_PATH = '../data'
IMAGES_DATA_DIRECTORY = os.path.join(FINAL_DATA_PATH, 'images')

DATASET_INFO_FILE = os.path.join(FINAL_DATA_PATH, 'info.json')

CCPD_IMAGES_DIRECTORY = os.path.join('../data', 'raw', 'CCPD2019')
CCPD_INFORMATION_FILE = os.path.join('../data', 'raw', 'CCPD2019', 'CCPD2019.json')


def create_record(image_path: str,
                  bounding_boxes: list,
                  index: int,
                  category_id: int = 0):
    """
    This function creates a record for a single image.

    Args:
        image_path (str): The path of the image.
        bounding_boxes (list): The list of bounding boxes.
        index (int): The index of the image.
        category_id (int): The category id.
    Returns:
        record (dict): The record for the image.
    """
    # Initialize the record dictionary and read the image
    record = {}
    image = plt.imread(image_path)

    # Add the image path, height, width, and index to the record dictionary
    record['file_name'] = image_path
    record['height'], record['width'] = image.shape[:2]
    record['image_id'] = index

    # Add the annotations to the record dictionary and return the record
    record['annotations'] = get_annotations(bounding_boxes=bounding_boxes,
                                            category_id=category_id)
    return record


def get_annotations(bounding_boxes, category_id):
    # Initialize the annotations list and loop through the bounding boxes
    annotations = []
    for i in range(0, len(bounding_boxes), 4):
        # Get the bounding box coordinates
        x_min, y_min, x_max, y_max = bounding_boxes[i: i + 4]

        # Create the polygon and flatten it
        poly = [
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max)
        ]
        poly = list(itertools.chain.from_iterable(poly))

        # Create the annotation dictionary and append it to the annotations list
        annotation = {
            'bbox': [x_min, y_min, x_max, y_max],
            'bbox_mode': BoxMode.XYXY_ABS,
            'segmentation': [poly],
            'category_id': category_id,
            'iscrowd': 0
        }
        annotations.append(annotation)
    return annotations


def download_files(urls: dict, directory: str = 'models'):
    """
    Downloads files from the given URLs and saves them to the given directory.

    Args:
        urls (dict): A dictionary of URLs to download.
        directory (str): The directory where the files should be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Loop through each URL and download the file
    for key, url in urls.items():
        # Get the filename from the URL
        filename = os.path.join(directory, url.split('/')[-1])

        # Download the file
        response = requests.get(url, stream=True)

        # Get the size of the file and set the block size for the progress bar
        file_size = int(response.headers.get('Content-Length', 0))
        block_size = 1024

        # Create a progress bar for the download
        progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=f'Downloading {key.capitalize()}')

        # Loop through the response data and write it to a file while updating the progress bar
        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        # Close the progress bar
        progress_bar.close()

        # Print an error message if the file was not downloaded successfully
        if file_size != 0 and progress_bar.n != file_size:
            print(f"ERROR: Failed to download {filename}")


def create_dataset_dicts(split: str, data_directory: str = 'data'):
    """
    a functionality to merge json files that contain portions of the total dataset (each downloaded individually)
    Args:
        split: str, the set of the data which separate files are to be merged, could be ['train', 'test', 'val']
        data_directory: str, the directory to flip around looking for the data to use

    Returns: list[dict]
        a list of dictionaries containing information about the dataset as a whole

    """
    total_data = os.path.join(data_directory, f'{split}_info.json')
    if os.path.isfile(total_data):
        data = json.load(open(total_data, 'r'))
        return data

    else:
        # initialise an empty list to hold the data objects
        data = []

        # recursively iterate through the data directory and look for the json files
        for root, dirs, files in os.walk(data_directory):
            # iterate through all the files contained the current directory
            for file in files:
                # is the current file is a json file
                if file.endswith('.json'):
                    # if the file name contains the name of the split of interest
                    if split in file:
                        # load the content of the file (which is a list) and extend the data list we initialised earlier
                        data.extend(json.load(open(os.path.join(root, file))))
        json.dump(data, open(total_data, 'w'))
        # return the list of dictionaries that represents the datasets
        return data


def register_datasets(data_directory: str,
                      thing_classes: list):
    """
    Registers datasets for training, testing, and validation in Detectron2.

    This function registers datasets for training, testing, and validation in Detectron2 using the provided data
    directory and thing classes. The data directory should contain the data for all three splits (train, test, valid)
    in separate subdirectories. The thing classes should be a list of class names corresponding to the classes present
    in the dataset.

    :param data_directory: The path to the directory containing the data for all three splits (train, test, valid).
    :type data_directory: str
    :param thing_classes: A list of class names corresponding to the classes present in the dataset.
    :type thing_classes: list
    """

    # Define a function that returns a dataset dictionary for the current split using create_dataset_dicts()
    def data_getter(data_split: str):
        return create_dataset_dicts(split=data_split, data_directory=data_directory)

    # Loop over all three splits (train, test, valid)
    for split in ['train', 'test', 'valid']:
        # Register the current split with Detectron2 using DatasetCatalog.register()
        DatasetCatalog.register(split, lambda data_split=split: data_getter(data_split=data_split))

        # Set thing_classes for the current split using MetadataCatalog.get().set()
        MetadataCatalog.get(split).set(thing_classes=thing_classes)

