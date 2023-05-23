import matplotlib.pyplot as plt
import itertools
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm
import json
import os
import shutil
import numpy as np
from typing import List, Dict
import cv2
import gdown
import zipfile
import tarfile
import lzma


CELEB_A_DATASET_DIRECTORY = os.path.join('data', 'raw', 'CelebA')
CELEB_A_IMAGES_DIRECTORY = os.path.join('data', 'raw', 'CelebA', 'img_celeba')
CELEB_A_ANNOTATIONS_FILE = os.path.join('data', 'raw', 'CelebA', 'anno', 'list_bbox_celeba.txt')
CELEB_A_INFORMATION_FILE = os.path.join('data', 'raw', 'CelebA', 'celeba_info.json')

WIDER_FACE_IMAGES_DIRECTORY = os.path.join('data', 'raw', 'WIDER_FACE')
WIDER_FACE_IMAGES_DIRECTORY_TRAIN = os.path.join('data', 'raw', 'WIDER_FACE', 'WIDER_train', 'images')
WIDER_FACE_ANNOTATIONS_FILE_TRAIN = os.path.join('data', 'raw', 'WIDER_FACE', 'wider_face_split',
                                                 'wider_face_train_bbx_gt.txt')
WIDER_FACE_IMAGES_DIRECTORY_VALID = os.path.join('data', 'raw', 'WIDER_FACE', 'WIDER_val', 'images')
WIDER_FACE_ANNOTATIONS_FILE_VALID = os.path.join('data', 'raw', 'WIDER_FACE', 'wider_face_split',
                                                 'wider_face_val_bbx_gt.txt')

WIDER_FACE_INFORMATION_FILE = os.path.join('data', 'raw', 'WIDER_FACE', 'wider_face.json')

CELEB_A_NUM_CANDIDATES = {
    'train': 160_000,
    'test': 10000,
    'val': 5000
}

# WIDER_FACE_NUM_CANDIDATES = {
#     'train': 10_0,
#     'test': 1000,
#     'val': 1000
# }

CCPD_NUM_CANDIDATES = {
    'train': 160_000,
    'test': 10000,
    'val': 5000
}

FINAL_DATA_PATH = 'data'
IMAGES_DATA_DIRECTORY = os.path.join(FINAL_DATA_PATH, 'images')

DATASET_INFO_FILE_TRAIN = os.path.join(FINAL_DATA_PATH, 'train.json')
DATASET_INFO_FILE_TEST = os.path.join(FINAL_DATA_PATH, 'test.json')
DATASET_INFO_FILE_VAL = os.path.join(FINAL_DATA_PATH, 'val.json')

CCPD_IMAGES_DIRECTORY = os.path.join('data', 'raw', 'CCPD2019')
CCPD_INFORMATION_FILE = os.path.join('data', 'raw', 'CCPD2019', 'CCPD2019.json')

IMAGE_SIZE = (360, 580)


def pre_process_data(image_path: str, bounding_boxes: list) -> list:
    """
    Pre-processes an image by resizing it and its corresponding bounding boxes.

    Args:
        image_path (str): The path to the image file.
        bounding_boxes (list): A list of bounding boxes in the format [x_min, y_min, x_max, y_max].

    Returns:
        list: A list of updated bounding boxes.
    """
    # Load the image from the given path
    image = plt.imread(image_path)

    # Resize the image and its corresponding bounding boxes
    image, bounding_boxes = adaptive_resize(image, bounding_boxes, new_size=IMAGE_SIZE)

    # Save the resized image to the same path
    plt.imsave(image_path, image)

    # Return the updated bounding boxes
    return bounding_boxes


def plot_images(images, show: bool):
    """
    Plots the input images using matplotlib.

    Args:
        images (numpy.ndarray): A 4D numpy array containing the images to plot.
        show (bool): whether to show the plot or not.
    Returns:
        None
    """
    if images.shape[0] > 4:
        # Create a grid of subplots with 4 columns
        fig, axs = plt.subplots(nrows=images.shape[0] // 4, ncols=4, figsize=(15, 10))
        for i in range(images.shape[0]):
            # Plot the current image in the appropriate subplot
            axs[i // 4, i % 4].imshow(images[i])
            axs[i // 4, i % 4].axis('off')
    else:
        # Create a grid of subplots with 1 column
        fig, axs = plt.subplots(nrows=images.shape[0], ncols=1, figsize=(15, 10))
        for i in range(images.shape[0]):
            # Plot the current image in the appropriate subplot
            axs[i].imshow(images[i])
            axs[i].axis('off')
    if show:
        plt.show()
    return fig


def visualize_sample(info_file: str, n_samples: int = 8, show=True, save_path: str = None):
    """
    Visualize a sample from a custom dataset for detectron2.

    This function takes in a list of dataset records and an index for the sample to be visualized.
    It reads the image and annotations for the specified sample and displays them using detectron2's
    Visualizer class.

    Args:
        info_file (str): The name (path) of the file containing the dataset information to draw from.
        n_samples (int): The number of images to draw and view.
        show (bool): whether to show the images plot or not
        save_path (str): path to a non-existent png file to which the plot will be saved, None equals don't save
    """
    # retrieve the registered dataset
    dataset_dicts = json.load(open(info_file))
    dataset_name = 'visualization_only'
    register_dataset(info_file, dataset_name)

    # generate random indexes to select images
    indexes = np.random.permutation(len(dataset_dicts))

    images = []
    for i in range(n_samples):
        # Get record for specified sample
        record = dataset_dicts[indexes[i]]

        # Read image using cv2's imread function
        # noinspection PyUnresolvedReferences
        img = cv2.imread(record['file_name'])

        # Convert image from BGR to RGB format
        # noinspection PyUnresolvedReferences
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create Visualizer object
        v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset_name), scale=1)

        # Use draw_dataset_dict method to draw annotations on image
        v = v.draw_dataset_dict(record)

        # Add the image to the list
        images.append(cv2.resize(v.get_image()[:, :, ::-1], (256, 256)))

    # Get annotated images using matplotlib imshow function
    fig = plot_images(np.array(images), show=show)
    # Save the plot to the specified path
    if save_path is not None:
        fig.savefig(save_path)


def adaptive_resize(image, bounding_boxes, new_size):
    """
    Resizes an image and its corresponding bounding boxes.

    Args:
        image (numpy.ndarray): The image to resize.
        bounding_boxes (list): A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        new_size (tuple): The new size of the image in the format (width, height).

    Returns:
        numpy.ndarray: The resized image.
        list: A list of bounding boxes with updated coordinates.
    """
    # Get the old size of the image
    old_size = image.shape[:2]

    # Calculate the scaling factor for each dimension
    scale_x = new_size[0] / old_size[0]
    scale_y = new_size[1] / old_size[1]

    # Resize the image
    # noinspection PyUnresolvedReferences
    resized_image = cv2.resize(image, new_size[::-1])

    # Update the bounding box coordinates
    new_bounding_boxes = []
    for i in range(0, len(bounding_boxes), 4):
        bbox = bounding_boxes[i: i + 4]
        x_min = int(bbox[0] * scale_x)
        y_min = int(bbox[1] * scale_y)
        x_max = int(bbox[2] * scale_x)
        y_max = int(bbox[3] * scale_y)
        new_bounding_boxes.extend([x_min, y_min, x_max, y_max])

    return resized_image, new_bounding_boxes


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


def register_dataset(info_file, dataset_name) -> None:
    """
    registers the datasets into detectron2's acceptable format
    :param info_file: str (path-like), the json file holding the dataset information.
    :param dataset_name: str, the name of the dataset to be registered.
    :return: None
    """
    try:
        # read the file representing the current split
        dataset_dicts = json.load(open(info_file))

        # register the current data split
        DatasetCatalog.register(dataset_name, lambda: dataset_dicts)

        # Set thing_classes for the current split using MetadataCatalog.get().set()
        MetadataCatalog.get(dataset_name).set(thing_classes=['Face', 'LP'])
    except AssertionError as ex:
        print(ex)


def modify_record(record, new_index, new_path):
    """
    Modify the record with new index and new path.

    Args:
        record: A dictionary containing the record information.
        new_index: An integer representing the new index.
        new_path: A string representing the new path.

    Returns:
        A dictionary containing the modified record information.
    """
    record['file_name'] = new_path
    record['image_id'] = new_index
    return record


def generate_data_split(info: List[Dict],
                        split: str,
                        indexes: np.ndarray,
                        start_index: int,
                        file_base_name: str,
                        data_directory: str):
    """
    Generates data for a given split.

    Args:
        info: list of dictionaries representing the dataset
        split: A string representing the split.
        indexes: List, a list of indexes to get from the dataset.
        start_index: int, an integer representing the number from which to start giving incremental IDs
        file_base_name: str, the beginning of the new file name to be saved in another folder
        data_directory: str, the directory to which new files will be saved

    Returns:
        A list containing the data for the given split.
    """
    data = []
    # noinspection PyTypeChecker
    for i, index in tqdm(enumerate(indexes), total=len(indexes), desc=f'Generating Data ({file_base_name}_{split})'):
        # Get the file name and path.
        file_name = info[index]['file_name'].split(os.path.sep)[-1]
        file_name = f'{file_base_name}_{file_name}'
        file_path = os.path.join(data_directory, file_name)

        # Copy the file to the face data path.
        shutil.copyfile(info[index]['file_name'], file_path)

        # Modify the record and append it to the data list.
        new_record = modify_record(record=info[index], new_index=i + start_index, new_path=file_path)
        data.append(new_record)

    return data


def select_from_data(json_file: str,
                     start_indexes: dict,
                     new_file_base_name: str,
                     num_examples: dict,
                     output_directory: str,
                     final_data_path: str or os.PathLike):
    """
        Selects data from CelebA dataset.

        Args:
            json_file: A string representing the path to the CelebA information file.
            start_indexes: A dictionary containing the start index for each split.
            new_file_base_name: str, the string at the beginning of each new dataset file.
            num_examples: dict, a dictionary containing the number of examples in each data split.
            output_directory: str, the directory to which new data files will be saved.
            final_data_path: str or os.PathLike, the directory to which the final json file will be saved.

        Returns:
            Three lists containing the data for train, test and validation splits.
        """
    assert type(start_indexes) == dict
    assert list(start_indexes.keys()) == ['train', 'test', 'val'], f'Unknown keys in start_indexes\n' \
                                                                   f'\t{start_indexes.keys()}'
    assert list(num_examples.keys()) == ['train', 'test', 'val'], f'Unknown keys in num_examples\n' \
                                                                  f'\t{num_examples.keys()}'

    # Load the dataset information file.
    info = json.load(open(json_file, 'r'))

    # creat an indexer to randomly select files from the dataset
    indexer = np.random.permutation(len(info))

    # Create the face data path if it doesn't exist.
    os.makedirs(output_directory, exist_ok=True)

    # initialise an empty list to hold the file names that hold the data splits temporarily
    files = {}

    for split in start_indexes.keys():
        # Generate data for train, test and validation splits.
        indexes = indexer[:num_examples[split]]
        data = generate_data_split(info=info,
                                   split=split,
                                   indexes=indexes,
                                   start_index=start_indexes[split],
                                   file_base_name=new_file_base_name,
                                   data_directory=output_directory)
        indexer = indexer[num_examples[split]:]
        temp_file = os.path.join(final_data_path, f'{new_file_base_name}_{split}_temp.json')
        json.dump(data, open(temp_file, 'w'))
        files[split] = temp_file

    return files


def merge(files: list, output_file: str) -> str:
    """
    Merges multiple JSON files into a single file.

    Args:
        files (list): A list of file paths to JSON files to be merged.
        output_file (str): The path to the output file.

    Returns:
        str: The path to the output file.
    """
    # Initialize an empty list to store the data from all files
    data = []

    # Iterate over each file in the input list
    for file in files:
        # Load the data from the current file and extend the data list
        data.extend(json.load(open(file)))

    # Dump the merged data into the output file
    json.dump(data, open(output_file, 'w'))

    # Return the path to the output file
    return output_file


def generate_splits(directory: str or os.PathLike,
                    original_json: str or os.PathLike,
                    num_examples: dict,
                    dataset_name: str,
                    shuffle: bool = True):
    """
    Generates train, test and validation splits from a given dataset.

    Args:
        directory (str or os.PathLike): The directory where the generated splits will be saved.
        original_json (str or os.PathLike): The path to the original dataset in JSON format.
        num_examples (dict): A dictionary containing the number of examples for each split.
                             Must have keys 'train', 'test' and 'val'.
        dataset_name (str): The name of the dataset.
        shuffle (bool): If True, shuffles the data before generating the splits. Default is True.

    Returns:
        None
    """
    # Check that the num_examples dictionary has the correct keys
    assert list(num_examples.keys()) == ['train', 'test', 'val']

    # Load the data from the original JSON file
    data = json.load(open(original_json))

    # Shuffle the data if shuffle is True
    if shuffle:
        indexes = np.random.permutation(len(data))
    else:
        indexes = np.arange(len(data))

    # Get the indexes for each split
    train_indexes = indexes[:num_examples['train']]
    test_indexes = indexes[len(train_indexes): len(train_indexes) + num_examples['test']]
    val_indexes = indexes[len(train_indexes) + len(test_indexes):
                          len(train_indexes) + len(test_indexes) + num_examples['val']]

    # Get the data for each split
    train_data = [data[int(index)] for index in train_indexes]
    test_data = [data[int(index)] for index in test_indexes]
    val_data = [data[int(index)] for index in val_indexes]

    # Save the splits to their respective JSON files
    json.dump(train_data, open(os.path.join(directory, dataset_name + '_train.json'), 'w'))
    json.dump(test_data, open(os.path.join(directory, dataset_name + '_test.json'), 'w'))
    json.dump(val_data, open(os.path.join(directory, dataset_name + '_val.json'), 'w'))


def path_fixer(path: str) -> str:
    """
    Replaces forward slashes, backslashes and multiple slashes in a path with the appropriate separator for the
    operating system.

    Args:
    path (str): The path to be fixed.

    Returns:
    str: The fixed path with the correct OS-relative separators.
    """

    # Replace double and single forward slashes with a temporary separator symbol ('$')
    path = path.replace('//', '$')
    path = path.replace('/', '$')

    # Replace backslashes with the temporary separator symbol ('$')
    path = path.replace('\\', '$')

    # Replace the temporary separator symbol ('$') with the correct separator for the operating system
    path = path.replace('$', os.path.sep)

    return path


def download(urls, directory):
    """
    downloads the files at the specified dataset-specific urls
    Args:
        urls: dict, (key, value) pairs representing file names (keys) and their corresponding urls (values)
        directory: str or os.PathLike, the path to the directory to which all downloads will be saved.

    Returns:
        None
    """
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # iterate through provided urls
    for key, value in urls.items():
        if '.' in key:
            gdown.download(url=value,
                           output=os.path.join(directory, key))
        # otherwise, it's probably a folder
        else:
            gdown.download_folder(url=value,
                                  output=directory,
                                  quiet=False,
                                  proxy=None,
                                  speed=None,
                                  use_cookies=True)


# Set a dictionary of supported compression formats
SUPPORTED_EXTENSIONS = {
    '.zip': {'file': lambda path: zipfile.ZipFile(path),
             'members': lambda file: file.namelist()},

    '.tar.gz': {'file': lambda path: tarfile.open(path),
                'members': lambda file: file.getmembers()},

    '.tar.xz': {'file': lambda path: tarfile.open(fileobj=lzma.open(path)),
                'members': lambda file: file.getmembers()},
    '.tgz': {'file': lambda path: tarfile.open(path, 'r:gz'),
             'members': lambda file: file.getmembers()}
}


def extract(path: Union[str, os.PathLike], output_directory: Union[str, os.PathLike]):
    """
    a function that extracts the file at the specified path
    Args:
        path: str or os.PathLike, the path to the file to be extracted.
        output_directory: str or os.PathLike, the path to the output file.

    Returns:
        None

    """
    # get the extension of the file
    extension = path[path.index('.'):]

    # get the corresponding information based on the file type
    file_info = SUPPORTED_EXTENSIONS.get(extension, None)

    # if the dictionary doesn't contain the specified file types are available
    if file_info is None:
        # raise a key error explaining what formats are
        raise KeyError(f'Extension {extension} is not supported')

    # get the file handle
    file = file_info['file'](path)
    # get a list of members the file contains
    members = file_info['members'](file)
    # noinspection PyTypeChecker
    # iterate through the list of members
    for member in tqdm(members, total=len(members), desc=f'Extracting files from {path} to {output_directory}'):
        # noinspection PyUnresolvedReferences
        # extract current member
        try:
            file.extract(member, path=output_directory)
        except PermissionError:
            # Most likely it's a duplicated file trying to be written again
            pass
