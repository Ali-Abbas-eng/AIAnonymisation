import random
from abc import ABC
import tarfile
import os
import cv2
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import requests
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer


def download_files(urls: dict = None, directory: str = 'data'):
    """
    Downloads files from the given URLs and saves them to the given directory.

    Args:
        urls (dict): A dictionary of URLs to download.
        directory (str): The directory where the files should be saved.
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    if urls is None:
        urls = {'images': 'http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz',
                'annotations': 'http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz',
                'readme': 'http://vis-www.cs.umass.edu/fddb/README.txt'}
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


def extract_data_from(text_content: str, root: str = 'data'):
    """
    Extracts data from text content.

    Parameters:
    -----------
    text_content: str
        String containing the data to be extracted.

    root: str, optional (default = 'data')
        Root directory where the image data is located.

    Returns:
    --------
    data: dict
        Dictionary containing the extracted data. The dictionary contains two keys:
        'image': a list of image file paths,
        'annotation': a list of lists where each sub-list contains annotations for the corresponding image.
    """
    data_ = {'image': [],
             'annotation': []}

    # Split the input string by new line characters
    lines = text_content.split('\n')

    i = 0
    # Process the lines until the end of the string is reached or a non-existent image is encountered
    while True:
        # Create the path to the image file
        path = os.path.join(root, *lines[i].split('/')) + '.jpg'

        # If the image file does not exist, stop processing
        if not os.path.isfile(path):
            break

        # Move to the next line in the input string to get the number of faces
        i += 1
        number_of_faces = int(lines[i])

        # Move to the next line in the input string to start reading the annotations
        i += 1
        annotations = []
        for n in range(i, i + number_of_faces, 1):
            annotations.append([float(x) for x in lines[n].split()])

        # Move to the next line in the input string to get to the next image
        i += number_of_faces

        # Append the image file path and its annotations to the data dictionary
        data_['image'].append(path)
        data_['annotation'].append(annotations)

    return data_


def extract_text_data(data_directory):
    """
    Extracts text data from a directory.

    Parameters:
    -----------
    data_directory: str
        Directory where the text data is located.

    Returns:
    --------
    data: dict
        Dictionary containing the extracted text data. The dictionary contains two keys:
        'image': a list of image file paths,
        'annotation': a list of lists where each sub-list contains annotations for the corresponding image.
    """

    data = {'image': [], 'annotation': []}

    # Iterate through all files in the directory tree
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            # Skip files that do not end with 'List.txt'
            if not file.endswith('List.txt'):
                continue

            # Read the text data from the file
            text = open(os.path.join(root, file)).read()

            # Extract the data from the text and add it to the main data dictionary
            data_part = extract_data_from(text, data_directory)
            data['image'].extend(data_part['image'])
            data['annotation'].extend(data_part['annotation'])
    return data


def unzip(zip_file: str, data_directory: str = 'data'):
    """
    Unzips a .tar file into a specified directory.

    Parameters:
    -----------
    zip_file: str
        Path to the .tar file to be unzipped.
    data_directory: str, optional
        Directory where the contents of the .tar file will be extracted to. Defaults to 'data'.

    Returns:
    --------
    None
    """
    # Create the directory if it does not exist
    if not os.path.isdir(data_directory):
        os.makedirs(data_directory)

    # Open the tar file
    compressed_images_file = tarfile.open(name=zip_file)
    contents = compressed_images_file.getmembers()

    # Extract each file in the tar file
    for file in tqdm(contents, total=len(contents), desc='Extracting Images'):
        compressed_images_file.extract(file, path=data_directory)


class FaceData(Dataset, ABC):
    def __init__(self,
                 data_directory: str = 'data',
                 split: str = 'test',
                 train_proportion: float = .6,
                 zipped_images_file: str = 'data/originalPics.tar.gz',
                 zipped_annotations_file: str = 'data/FDDB-folds.tgz',
                 unpack: bool = False,
                 download: bool = False,
                 ):
        """
        Class constructor.

        Parameters:
        ----------
        data_directory : str, optional (default='data')
            The directory to store the extracted files and annotations.
        split : str, optional (default='test')
            The data split to use, either 'train', 'val', or 'test'.
        train_proportion : float, optional (default=0.6)
            The proportion of the data to use for training.
        zipped_images_file : str, optional (default='originalPics.tar.gz')
            The path to the zipped images file.
        zipped_annotations_file : str, optional (default='FDDB-folds.tgz')
            The path to the zipped annotations file.
        unpack : bool, optional (default=False)
            Whether to extract the zipped files or not.

        Raises:
        ------
        AssertionError:
            If there is an error in unpacking the files.
            If there is a mismatch in the length of the extracted images and annotations.
            If the annotations text files are empty or do not exist.
            If an unknown data split is provided.
        """
        self.data_directory = data_directory
        if not os.path.isdir(self.data_directory):
            os.makedirs(self.data_directory)
        self.split = split
        if unpack:
            if download:
                download_files()
            # Check if the zipped images file path is valid
            assert zipped_images_file is not None, "Expected a path to the images' zip file, got None instead"
            assert os.path.isfile(zipped_images_file), 'Specified path for zipped images file was not found'
            # Extract the images
            unzip(zipped_images_file, self.data_directory)

            # Check if the zipped annotations file path is valid
            assert zipped_annotations_file is not None, "Expected a path to the annotations' zip file, got None instead"
            assert os.path.isfile(zipped_annotations_file), 'Specified path for zipped annotations file was not found'
            # Extract the annotations
            unzip(zipped_annotations_file, self.data_directory)

        # Extract the data from the extracted annotations
        data_df = extract_text_data(self.data_directory)

        # Check if the length of the images and annotations match
        assert len(data_df['image']) == len(data_df['annotation']), 'Lengths mismatched, make sure the extracted ' \
                                                                    'files are not corrupted, the length of the ' \
                                                                    'images list should be the same as ' \
                                                                    'the annotations list'
        # Check if there is data present in the annotations text files
        assert len(data_df['image']) > 0, 'Looks like the annotations text files were empty or they do not exist at all'

        # Store the extracted data in a pandas dataframe
        self.data_df = pd.DataFrame(data_df, index=None)
        # Calculate the end indices for each split
        self.train_end = int(self.data_df.shape[0] * train_proportion)
        self.val_end = self.train_end + int(self.data_df.shape[0] * (1 - train_proportion) / 2)

        # Select the data split based on the split parameter
        if split == 'train':
            self.data_df = self.data_df.loc[0:self.train_end, :].reset_index()
        elif split == 'val':
            self.data_df = self.data_df.loc[self.train_end:self.val_end, :].reset_index()
        elif split == 'test':
            self.data_df = self.data_df.loc[self.val_end:, :].reset_index()
        else:
            raise KeyError(f'Unknown data split, expected one of ["train", "test", "val"], got {split}.')

    def __len__(self):
        """
        Return the number of data points in the dataset.

        Returns:
            int: The number of data points in the dataset.
        """
        return self.data_df.shape[0]

    def get_record(self, index):
        """
        Returns a dictionary containing the image and annotations data at a given index.

        Args:
            index (int): The index of the record to retrieve.

        Returns:
            A dictionary containing the following keys:
                - 'file_name': The name of the image file.
                - 'image_id': The unique ID of the image.
                - 'height': The height of the image.
                - 'width': The width of the image.
                - 'annotations': A list of annotations for the image, where each annotation is a dictionary containing
                                 the keys 'bbox' (bounding box coordinates) and 'category_id' (category ID).

        """

        # Retrieve the row at the given index from the DataFrame.
        row = self.data_df.loc[index]

        # Retrieve the height and width of the image using its file path.
        height, width = plt.imread(row['image']).shape[:2]
        annotations = []
        # Generate a list of annotations for the image using the row's ellipse data.
        for ellipse in row['annotation']:
            bbox = list(FaceData.get_bounding_box(FaceData.image_info(ellipse)))
            x_min = bbox[0]
            x_max = bbox[2]
            y_min = bbox[1]
            y_max = bbox[3]
            poly = [
                (x_min, y_min), (x_max, y_min),
                (x_max, y_max), (x_min, y_max)
            ]
            poly = list(itertools.chain.from_iterable(poly))
            annotations.append({'bbox': bbox,
                                'category_id': 0,
                                'segmentation': [poly],
                                'iscrowd': 0,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                })

        # Calculate the unique image ID based on the dataset split.
        image_id = index
        if self.split == 'test':
            image_id += self.val_end
        elif self.split == 'val':
            image_id += self.train_end

        # Return the dictionary containing the image and annotations data.
        return {
            'file_name': row['image'],
            'image_id': image_id,
            'height': height,
            'width': width,
            'class_name': 'face',
            'annotations': annotations
        }

    def __getitem__(self, index):
        """
        Return a single data point from the dataset corresponding to the given index.

        Args:
            index (int): The index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding annotation.
        """
        row = self.data_df.loc[index]
        # Read in the image file using plt.imread.
        image = plt.imread(row['image'])
        # Retrieve the annotation for the corresponding image.
        label = row['annotation']

        # Return a tuple containing the image and its corresponding annotation.
        return image, label

    @staticmethod
    def get_bounding_box(ellipse):
        """
        Calculates the bounding box coordinates for an ellipse.

        Args:
            ellipse (tuple): A tuple containing the center point of the ellipse as a tuple (x, y) and its width and height
                             as a tuple (width, height).

        Returns:
            A tuple containing the coordinates of the bounding box as (x_min, y_min, x_max, y_max).
        """

        # Extract the center point and dimensions of the ellipse.
        x, y = ellipse[0]
        height = ellipse[1][0]
        width = ellipse[1][1]

        # Calculate the minimum and maximum x and y values of the bounding box.
        x_min = x - width
        y_min = y - height
        x_max = x + width
        y_max = y + height

        # Return the bounding box coordinates as a tuple.
        return x_min, y_min, x_max, y_max

    @staticmethod
    def image_info(ellipse):
        """
        Returns information about an ellipse that is used to annotate a face in an image.

        Args:
            ellipse: A list of values representing the parameters of an ellipse. The list should have 6 elements, which
            are (in order): major_axis_radius, minor_axis_radius, angle, center_x, center_y, and score.

        Returns:
            A tuple containing the following information about the ellipse: center point (x, y), axes length (major, minor),
            and angle (in degrees).

        """
        # Extract the parameters of the ellipse
        major_axis_radius, minor_axis_radius, angle, center_x, center_y, _ = ellipse

        # Convert the axes lengths to integers
        axes_length = (int(major_axis_radius), int(minor_axis_radius))

        # Convert the center point coordinates to integers
        center_point = (int(center_x), int(center_y))

        # Convert the angle from radians to degrees and round to the nearest integer
        angle_degrees = int(angle * 180 / np.pi)

        # Return the extracted information as a tuple
        return center_point, axes_length, angle_degrees

    @staticmethod
    def plot_image(image, ellipses, bounding_boxes):
        """
        Plot ellipses on the given image.

        Parameters:
        ----------
        image: numpy.ndarray
            An image represented as a numpy ndarray.

        ellipses: list
            A list of ellipse objects to be drawn on the image.

        Returns:
        -------
        numpy.ndarray
            The input image with ellipses drawn on it.
        """
        # Define the color and thickness of the ellipse outlines
        color = (255, 0, 0)
        thickness = 2

        # Draw an ellipse for each ellipse object in the list
        for ellipse in ellipses:
            info = FaceData.image_info(ellipse)
            center, axes_length, angle_degrees = info
            if bounding_boxes:
                x_min, y_min, x_max, y_max = FaceData.get_bounding_box(info)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
            else:
                cv2.ellipse(image, center, axes_length, angle_degrees, 0, 360, color, thickness)

        # Return the image with ellipses drawn on it
        return image

    def plot(self, batch=None, bounding_boxes: bool = False):
        """
        Plot a batch of images with ellipses on them.

        Parameters:
        ----------
        batch: list, optional (default=None)
            A list of data items to plot. If None, a batch of 8 random items will be plotted.

        Returns:
        -------
        None
        """
        # If no batch is provided, generate a random batch of 8 items from the data generator
        if batch is None:
            batch = [self.__getitem__(k) for k in np.random.randint(low=0, high=len(self), size=4)]

        # Process each item in the batch and store the resulting images in a list
        processed_images = []
        for item in batch:
            image, ellipses = item
            processed_images.append(FaceData.plot_image(image, ellipses, bounding_boxes))

        # Calculate the number of rows needed to plot all the images
        num_rows = int(np.ceil(len(batch) / 2))

        # Create a figure with subplots for each row
        fig, axs = plt.subplots(2, num_rows)

        # Plot each processed image in the appropriate subplot
        for i in range(num_rows):
            try:
                axs[0, i].imshow(processed_images[i])
                axs[1, i].imshow(processed_images[i + 1])
            except IndexError:
                # If an IndexError occurs, it means that there is only one row left, so only plot on the first row
                axs[0, i].imshow(processed_images[i])

        # Show the plot
        plt.show()


def get_data_records(data_directory: str = 'data',
                     split: str = 'train',
                     train_proportion: float = .6,
                     zipped_images_file: str = 'data/originalPics.tar.gz',
                     zipped_annotations_file: str = 'data/FDDB-folds.tgz',
                     unpack: bool = False, ):
    """
    Generates a list of dataset records for a given dataset split.

    Args:
        data_directory (str): The path to the directory containing the dataset.
        split (str): The split of the dataset to generate records for (i.e. 'train', 'val', or 'test').
        train_proportion (float): The proportion of the dataset to use for training (only used if split='train').
        zipped_images_file (str): The name of the compressed file containing the dataset images.
        zipped_annotations_file (str): The name of the compressed file containing the dataset annotations.
        unpack (bool): Whether to unpack the compressed dataset files.

    Returns:
        A list of dataset records, where each record is a dictionary containing the keys:
        - 'file_name': The name of the image file.
        - 'image_id': The unique ID of the image.
        - 'height': The height of the image.
        - 'width': The width of the image.
        - 'annotations': A list of annotations for the image, where each annotation is a dictionary containing
                         the keys 'bbox' (bounding box coordinates) and 'category_id' (category ID).

    """

    # Initialize a new instance of the FaceData class with the given parameters.
    dataset = FaceData(
        data_directory=data_directory,
        split=split,
        train_proportion=train_proportion,
        zipped_images_file=zipped_images_file,
        zipped_annotations_file=zipped_annotations_file,
        unpack=unpack
    )

    # Generate a list of dataset records by calling the get_record() method on each index in the dataset.
    records = [dataset.get_record(i) for i in tqdm(range(len(dataset)), desc='Generating Dataset Records')]

    # Return the list of dataset records.
    return records


def register(show: bool = False):
    """
    Registers a custom dataset named "FDDB" with three splits "train", "test", and "val" using detectron2 library.
    Also visualizes three random images from the test set with their ground truth bounding boxes.

    Args:
        show (bool): Whether to show the visualization or not. Default is True.
    """
    # Register the dataset splits
    for split in ["train", "test", "val"]:
        DatasetCatalog.register("FDDB_" + split, lambda split=split: get_data_records(split=split))
        MetadataCatalog.get("FDDB_" + split).set(thing_classes=["face"])

    # Get the testing data and data dictionaries
    testing_data = MetadataCatalog.get("FDDB_test")
    test_data_dicts = get_data_records(split='test')

    # Visualize random images from the test set
    if show:
        for data_point in random.sample(test_data_dicts, 3):
            img = plt.imread(data_point["file_name"])
            visualizer = Visualizer(img, metadata=testing_data, scale=0.5)
            out = visualizer.draw_dataset_dict(data_point)
            plt.imshow(out.get_image())
            plt.show()


