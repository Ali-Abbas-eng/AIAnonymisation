import os
from tqdm.auto import tqdm
import data_tools
import lzma
import tarfile
import json
import gdown
from typing import Union


def download_and_extract(download_directory: Union[str, os.PathLike] = os.path.join('data', 'zipped'),
                         unzipped_directory: Union[str, os.PathLike] = os.path.join('data', 'raw', 'CCPD2019')):
    """
    Downloads and extracts CCPD2019 dataset from Google Drive.

    Args:
        download_directory (Union[str, os.PathLike]): Directory to save downloaded files. Defaults to 'data/zipped'.
        unzipped_directory (Union[str, os.PathLike]): Directory to extract files. Defaults to 'data/raw/CCPD2019'.
    """
    # Download CCPD2019 dataset from Google Drive
    zipped_file_path = os.path.join(download_directory, 'CCPD2019.tar.xz')
    gdown.download(url='https://drive.google.com/uc?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc',
                   output=zipped_file_path)

    # Extract downloaded files
    with lzma.open(zipped_file_path) as decompressed_xz:
        with tarfile.open(fileobj=decompressed_xz) as decompressed_tar:
            members = decompressed_tar.getmembers()
            for member in tqdm(members, total=len(members)):
                decompressed_tar.extract(member=member, path=unzipped_directory)


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
