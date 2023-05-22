from .data_tools import download, extract
import os
import json
from typing import Union, Callable
from os import PathLike
import shutil

class ImagesDataset:
    """
    Encapsulation of the Dataset classes, all dataset used in this project MUST use this class as a super class, or
    ideally, instantiate an object from this class with the dataset-specific information as parameters
    """
    def __init__(self,
                 name: str,
                 path: Union[str, PathLike],
                 coco_file: Union[str, PathLike] = None,
                 urls: dict = None,
                 cache_directory: Union[str, PathLike] = None,
                 auto_remove_cache: bool = True,
                 registration_info_generator: Callable = None,
                 registration_info_generator_parameters: dict = None) -> None:
        """
        initializer of the dataset
        Args:
            name: str, name of the dataset.
            path: str or PathLike, the path to the directory holding the dataset as is inside it.
            coco_file: str or PathLike, the path to the JSON file that will represent the dataset in COCO format.
            urls: dict, keys are file names and values are corresponding urls pointing to the file .
            cache_directory: str or PathLike, directory in which the downloaded files will be saved.
            auto_remove_cache: bool, set to True in case you want to delete downloaded files upon extraction.
            registration_info_generator: Callable, function that returns a list of dicts representing the dataset
            registration_info_generator_parameters: dict, parameters to the registry generating function.
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
                extract(path=file, output_directory=self.path)
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
