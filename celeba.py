import os

import detectron2.structures

import data_tools
import json
from tqdm.auto import tqdm


def generate_dataset_registration_info(data_directory: str or os.PathLike = data_tools.CELEB_A_IMAGES_DIRECTORY,
                                       annotations_file: str or os.PathLike = data_tools.CELEB_A_ANNOTATIONS_FILE,
                                       info_path: str or os.PathLike = data_tools.CELEB_A_INFORMATION_FILE):
    """
    Generates dataset records (list of dictionaries) and saves them to a json file later to be imported by detectron2 and registered.

    Args:
        data_directory (str or os.PathLike): The directory where the images are stored.
        annotations_file (str or os.PathLike): The path to the annotations file.
        info_path (str or os.PathLike): The path to the information file.

    Returns:
        None
    """
    # Read the annotations file and split it into lines
    annotations = open(annotations_file, 'r').read().split('\n')[2:]

    # Remove empty lines and extra spaces
    annotations = [item.replace("  ", ' ') for item in annotations if len(item) > 2]

    # Convert each line into a list of integers and strings
    annotations = [[int(item) if item.isnumeric() else item for item in line.split()] for line in annotations]

    # Create an empty list to store the dataset records
    dataset_records = []

    # Loop through each example in the annotations
    for index, example in tqdm(enumerate(annotations), total=len(annotations)):
        # Get the image path
        image_path = os.path.join(data_directory, example[0])

        # Get the bounding box coordinates
        x_min = example[1]
        y_min = example[2]
        x_max = example[1] + example[3]
        y_max = example[2] + example[4]

        # Create a record for the example
        example_record = data_tools.create_record(image_path=image_path,
                                                  index=index,
                                                  bounding_boxes=[x_min, y_min, x_max, y_max],
                                                  bbox_format=detectron2.structures.BoxMode.XYXY_ABS,
                                                  category_id=0)

        # Add the example record to the dataset records list
        dataset_records.append(example_record)

    # Save the dataset records to a json file
    json.dump(dataset_records, open(info_path, 'w'))


if __name__ == '__main__':
    generate_dataset_registration_info()