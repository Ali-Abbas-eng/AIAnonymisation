import copy

import numpy as np
import torch
from utils import download, extract, modify_record
import os
import json
from typing import Callable
from os import PathLike
import shutil
import random
from utils import recompute_bounding_boxes
from detectron2.data import transforms
from detectron2.data import detection_utils
from typing import List, Union


class ImagesDataset:
    """
    Encapsulation of the Dataset classes, all dataset used in this project MUST use this class as a super class, or
    ideally, instantiate an object from this class with the dataset-specific information as parameters
    """

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, path):
        self.__path = path
        self.coco_file = os.path.join(path, self.name + '.json')

    @property
    def coco_file(self):
        return self.__coco_file

    @coco_file.setter
    def coco_file(self, path):
        self.__coco_file = path

    def __init__(self,
                 name: str,
                 path: Union[str, PathLike],
                 coco_file: Union[str, PathLike] = None,
                 urls: dict = None,
                 cache_directory: Union[str, PathLike] = None,
                 auto_remove_cache: bool = True,
                 shuffle: bool = True) -> None:
        """
        initializer of the dataset
        Args:
            name: str, name of the dataset.
            path: str or PathLike, the path to the directory holding the dataset as is inside it.
            coco_file: str or PathLike, the path to the JSON file that will represent the dataset in COCO format.
            urls: dict, keys are file names and values are corresponding urls pointing to the file .
            cache_directory: str or PathLike, directory in which the downloaded files will be saved.
            auto_remove_cache: bool, set to True in case you want to delete downloaded files upon extraction.
            shuffle: bool, whether to shuffle the dataset upon creation.
        """
        # Set dataset attributes
        self.name = name
        self.path = path
        self.urls = urls
        self.cache_directory = cache_directory
        self.auto_remove_cache = auto_remove_cache

        # Set the path to the coco file (specify a default value for the path)
        self.coco_file = coco_file if coco_file else os.path.join(self.path, self.name + '.json')

        # Set the current value of the registration_info_generator parameter to None
        self.generate_dataset_registration_info_params = None

        # noinspection PyTypeChecker
        # Set the current value of the registration_info_generator to None (since it's most likely to be used only once)
        self.registration_info_generator: Callable = None

        # Set the value to the readiness of the dataset relevant to the existence of the coco file
        self.ready_to_use = os.path.exists(self.coco_file)

        # Set the value of the shuffle attribute
        self.shuffle = shuffle

    def get_data_list(self):
        """
        Helper function that gets the dataset in the same coco format saved to disk (generates the file if nonexistent)
        Returns:
            list, list of dictionaries representing the dataset

        """
        # if the coco file exists
        if self.ready_to_use:
            data_list = json.load(open(self.coco_file))
            if self.shuffle:
                random.shuffle(data_list)

            # read the file, return the content
            return data_list
        # If the coco file doesn't exist
        else:
            # Generate the file
            self.generate_dataset_registration_info()
            # Recursive call to make things easier
            return self.get_data_list()

    def generate_dataset_registration_info(self):
        """
        this function will call the registration info generator which is a function provided with the dataset, since
        each dataset can have a mind-numbingly tedious process to get to the bounding box
        Returns:
            None, this function writes the dataset registration info (COCO format) as a JSON file.
        """
        # get the list of dictionaries
        records = self.registration_info_generator(**self.generate_dataset_registration_info_params)
        # Dump the dataset_dicts to the info file
        json.dump(records, open(self.coco_file, 'w'))

        # Set the value of ready_to_use attribute to True since the dataset coco file was just created
        self.ready_to_use = True

    def extract_dataset_files(self) -> None:
        """
        a helper function to extract the dataset from the specified path
        Returns:
            None
        """
        # If the directory to the compressed files are not None
        if self.cache_directory:
            # Generate a list of paths pointing to the compressed files
            files = [os.path.join(self.cache_directory, file) for file in os.listdir(self.cache_directory)]
            # Iterate through the list of files
            for file in files:
                # noinspection PyTypeChecker
                # extract the file (if it's one of the supported file types)
                try:
                    # noinspection PyTypeChecker
                    extract(path=file, output_directory=self.path)
                except KeyError:
                    print(f'Looks like one of the contents of the directory {self.cache_directory} is not supported, '
                          f'the file that threw the exception was {file}')
            # If the instantiated object has auto_remove_cache == True
            if self.auto_remove_cache:
                # Delete the files in the Cache directory
                shutil.rmtree(self.cache_directory)

    def download_dataset_files(self):
        """
        helper function to download the dataset from the specified urls
        Returns:
            None, the function saves the downloaded files to the self.cache_directory
        """
        # Check if the attributes are not None
        if self.urls and self.cache_directory:
            # download the dataset (if the urls are supported (full url to file (ends with the file name) or GDrive url)
            download(urls=self.urls, directory=self.cache_directory)
        # extract downloaded files
        self.extract_dataset_files()

    def split(self, splits: dict):
        """
        helper function to split the dataset into multiple subsets (generally, train, test, and validation splits)
        Args:
            splits: dict, keys are strings representing the paths to the new subsets (json files),
             values are size of the split
            relative to the whole dataset
        Returns:
            None, writes len(splits.keys) new files each with the name convention that is (dataset_name_key.json)
        """
        # Make sure there is nothing fishy in the provided parameters (sum of the proportion MUST equal 1)
        assert sum(list(splits.values())) == 1., 'Provided proportions of the dataset does not add up to one.'
        # Generate a list of dictionaries representing the dataset in COCO format
        data_list = self.get_data_list()
        # If the coco file exists
        if self.ready_to_use:
            # Start from the very beginning of the dataset
            start_index = 0
            # Iterate through the provided subset information
            for split, proportion in splits.items():
                # Calculate the end to of the new subset as the len(data_list) * relative size of subset (proportion)
                end_index = int(len(data_list) * proportion)
                # Create the new subset files
                subset = data_list[start_index: start_index + end_index]
                # Reset the indexes of the records in the generated data split (file indexes in new split starts from 0)
                subset = [modify_record(record=record, new_index=index, new_path=record['file_name'])
                          for index, record in enumerate(subset)]
                # Set the new value of the start_index to be the end index of the current subset
                start_index = len(subset)
                # Write the new subset to the provided path
                json.dump(subset, open(split, 'w'))


def custom_data_mapper(dataset_dict):
    """
    Re-Computes the bounding boxes after performing the transformations in the pipeline
    Arguments:
        dataset_dict: dict, a dictionary representing the data point of interest
    """
    # Deep copy of the dataset dict since it will be changed later (recommended in the documentation)
    dataset_dict = copy.deepcopy(dataset_dict)

    # Get the original dimensions of the image
    original_size = (dataset_dict['height'], dataset_dict['width'])

    # Set the new size of the image
    shortest_edge_range = [256, 512]

    # Read the image
    image = detection_utils.read_image(dataset_dict['file_name'], format='BGR')

    # Augment the input
    aug_input = transforms.AugInput(image)

    # Create the transform
    transform = transforms.ResizeShortestEdge(shortest_edge_range)(aug_input)

    # Get the image
    image = torch.from_numpy(aug_input.image.transpose(2, 0, 1).copy())

    # Save the new image size
    new_size = image.shape[1:3]

    # Loop through the annotations of the instances in the current data point
    for annotation in dataset_dict['annotations']:
        # Recompute the bounding boxes according to the new image size
        annotation['bbox'] = recompute_bounding_boxes(annotation['bbox'], original_size, new_size)

    # Set the image value
    dataset_dict['image'] = image

    # Set the instances
    dataset_dict['instances'] = detection_utils.annotations_to_instances(dataset_dict['annotations'], new_size)

    return dataset_dict
