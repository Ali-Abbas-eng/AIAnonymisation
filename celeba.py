import os
import data_tools
import json
from tqdm.auto import tqdm
from typing import Union
import gdown
import py7zr
import multivolumefile


def download_and_extract(download_directory: Union[str, os.PathLike] = os.path.join('data', 'zipped'),
                         unzipped_directory: Union[str, os.PathLike] = os.path.join('data', 'raw')):
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

    filenames = [os.path.join(download_directory, file) for file in os.listdir(download_directory)]
    os.makedirs(unzipped_directory, exist_ok=True)
    with multivolumefile.open('/content/drive/MyDrive/CelebA/img_celeba.7z', mode='rb') as target_archive:
        with py7zr.SevenZipFile(target_archive, 'r') as archive:
            archive.extractall(unzipped_directory)


def generate_dataset_registration_info(data_directory: str or os.PathLike = data_tools.CELEB_A_IMAGES_DIRECTORY,
                                       annotations_file: str or os.PathLike = data_tools.CELEB_A_ANNOTATIONS_FILE,
                                       info_path: str or os.PathLike = data_tools.CELEB_A_INFORMATION_FILE):
    """
    Generates dataset records (list of dictionaries) and saves them to a json file.

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
                                                  category_id=0)

        # Add the example record to the dataset records list
        dataset_records.append(example_record)

    # Save the dataset records to a json file
    json.dump(dataset_records, open(info_path, 'w'))


if __name__ == '__main__':
    generate_dataset_registration_info()
