import os

import detectron2.structures
from tqdm.auto import tqdm
import data_tools
import json


def decode_file_name(file_path: str) -> list or None:
    """
    This function decodes the file name and extracts the coordinates of the bounding box.

    Args:
    file_path (str): The path of the file.

    Returns:
    list: A list containing the coordinates of the bounding box.
    """
    try:
        # Split the file path and extract the file name
        file_name = file_path.split(os.path.sep)[-1][:-4]

        # Split the file name and extract the fields containing the coordinates
        fields = file_name.split('-')
        fields = fields[2].split('_')

        # Extract the coordinates from the fields
        x1, y1 = map(int, fields[0].split('&'))
        x2, y2 = map(int, fields[1].split('&'))

        # Return the coordinates as a list
        return [x1, y1, x2, y2]
    except IndexError:
        return None


def generate_dataset_registration_info(data_directory: str = data_tools.CCPD_IMAGES_DIRECTORY,
                                       info_path: str = data_tools.CCPD_INFORMATION_FILE) -> None:
    """
    This function generates the dataset registration info.

    Args:
    data_directory (str): The directory containing the data.
    info_path (str): The path of the info file.

    Returns:
    None
    """
    # Initialize the index and dataset_dicts
    index = 0
    dataset_dicts = []

    # Calculate the total number of progress bar bars
    progress_bar_bars = sum(len(files) for _, __, files in os.walk(data_directory))

    # Create a progress bar
    with tqdm(total=progress_bar_bars) as progress_bar:
        # Traverse through the data directory
        for root, _, files in os.walk(data_directory):
            # Traverse through the files in the directory
            for file in files:
                # Check if the file is an image file
                if file.endswith('.jpg'):
                    # Decode the file name to extract the coordinates
                    coordinates = decode_file_name(file)
                    if coordinates is not None:
                        file_path = os.path.join(root, file)

                        # Create a record and append it to the dataset_dicts
                        record = data_tools.create_record(image_path=file_path,
                                                          bounding_boxes=coordinates,
                                                          bbox_format=detectron2.structures.BoxMode.XYXY_ABS,
                                                          category_id=1,
                                                          index=index)
                        dataset_dicts.append(record)
                        index += 1

                    # Update the progress bar
                    progress_bar.update()

    # Dump the dataset_dicts to the info file
    json.dump(dataset_dicts, open(info_path, 'w'))


if __name__ == '__main__':
    # Call the generate_dataset_registration_info function
    generate_dataset_registration_info()
