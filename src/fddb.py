import argparse
import json
from data_tools import download_files, path_fixer, create_record
import os
import tarfile
from tqdm import tqdm
from typing import Callable


def download_and_extract(urls=None,
                         zipped_directory: str or os.PathLike = os.path.join('data', 'zipped', 'FDDB'),
                         unzipped_directory: str or os.PathLike = os.path.join('data', 'raw', 'FDDB')):
    """
    Downloads and extracts files from the specified URLs to the given directories.

    Args:
        urls (dict): Dictionary containing URLs for 'images' and 'annotations' files.
                     Defaults to None.
        zipped_directory (str or os.PathLike): Directory path to store the downloaded zip files.
                                               Defaults to 'data/zipped/FDDB'.
        unzipped_directory (str or os.PathLike): Directory path to extract the files.
                                                 Defaults to 'data/raw/FDDB'.
    """
    # Set default URLs if not provided
    if urls is None:
        urls = {'images': 'https://vis-www.cs.umass.edu/fddb/originalPics.tar.gz',
                'annotations': 'https://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz'}

    # Create zipped directory if it doesn't exist
    if not os.path.exists(zipped_directory):
        os.makedirs(zipped_directory)

    # Download the files to the zipped directory
    download_files(urls, directory=zipped_directory)

    # Extract files from each downloaded archive
    for file in os.listdir(zipped_directory):
        tar_file = tarfile.open(os.path.join(zipped_directory, file))
        members = tar_file.getmembers()
        for member in tqdm(members, total=len(members), desc=f'Extracting {file}'):
            tar_file.extract(member, path=unzipped_directory)


def get_bbox(details):
    """
    Calculates the bounding box coordinates from the given face details.

    Args:
        details (list): List containing face details.

    Returns:
        tuple: Bounding box coordinates (x1, y1, x2, y2).
    """
    # Extract the face details and convert them to integers
    face = [int(float(details[j])) for j in range(5)]

    # Calculate the center coordinates of the bounding box
    centre = (face[3], face[4])

    # Calculate the lengths of the axes of the bounding box
    axes_length = (face[1], face[0])

    # Calculate the corner coordinates of the bounding box
    x1, y1 = centre[0] + axes_length[0], centre[1] - axes_length[1]
    x2, y2 = centre[0] - axes_length[0], centre[1] + axes_length[1]

    # Return the bounding box coordinates
    return x1, y1, x2, y2


def extract_file_info(data_directory,
                      text_file: str or os.PathLike,
                      create_rec: Callable,
                      progress_bar):
    """
    Extracts file information from a text file and generates records based on the information.

    Args:
        data_directory (str or os.PathLike): Directory path of the data files.
        text_file (str or os.PathLike): Path to the text file containing file information.
        create_rec (Callable): A function that creates a record based on given parameters.
        progress_bar: A progress bar object used to track progress.

    Returns:
        list: List of records generated from the file information.
    """
    # Read the text file and split it into lines
    with open(text_file) as file_handle:
        info = file_handle.read().split('\n')
        info = [item for item in info if item != '\n']

    image_index = 0
    records = []
    for index, line in enumerate(info):
        # Construct the full path of the image file
        path = path_fixer(os.path.join(data_directory, line)) + '.jpg'

        coordinates = []
        if os.path.exists(path):
            num_faces = int(info[index + 1])
            faces = info[index + 2: index + 2 + num_faces]
            for face in faces:
                details = face.split()
                # Extract the bounding box coordinates using the get_bbox function
                x1, y1, x2, y2 = get_bbox(details=details)
                coordinates.extend([x1, y1, x2, y2])

            # Create a record using the provided create_record function
            record = create_rec(image_path=path,
                                bounding_boxes=coordinates,
                                category_id=0,
                                index=image_index)
            image_index += 1
            progress_bar.update()
            records.append(record)

    return records


def generate_dataset_registration_info(data_directory: str or os.PathLike,
                                       info_path: str or os.PathLike):
    """
    Generates dataset registration information and saves it as a JSON file.

    Args:
        data_directory (str or os.PathLike): Directory path of the dataset.
        info_path (str or os.PathLike): Path to save the generated registration information as a JSON file.
    """
    # Calculate the total progress based on the number of files to process
    total_progress = sum([len(open(os.path.join(data_directory, 'FDDB-folds', file)).read().split('\n'))
                          for file in os.listdir(os.path.join(data_directory, 'FDDB-folds'))
                          if 'ellipse' not in file]) - 10

    # Initialize a progress bar with the total progress
    with tqdm(total=total_progress, desc='Generating FDDB Registration Info') as progress_bar:
        total_records = []
        for file in os.listdir(os.path.join(data_directory, 'FDDB-folds')):
            if 'ellipse' in file:
                # Extract file information and create records using the extract_file_info function
                records = extract_file_info(data_directory,
                                            os.path.join(data_directory, 'FDDB-folds', file),
                                            create_record,
                                            progress_bar)
                total_records.extend(records)

    # Save the generated records as a JSON file
    json.dump(total_records, open(info_path, 'w'))


def main(download: int,
         images_download_url: str,
         annotations_download_url: str,
         zipped_directory: str or os.PathLike,
         unzipped_directory: str or os.PathLike):
    """
    Encapsulation of the data getting and formatting functionalities, the default output of this function is a directory that contain the dataset in the following hirearchy
    data:
    ----raw:
    --------FDDB:
    ------------2002
    ------------2003
    ------------FDDB-folds
    ------------FDDB.json
    Args:
        download: int, whether to download the dataset or not, activates the download and extract behaviour if > 0
        unzipped_directory: str or os.PathLike, the directory to which to unpack the zipped dataset
        images_download_url: str, url to the images zipped file to be downloaded.
        annotations_download_url: str, url to the zipped images' annotations file to be downloaded
        zipped_directory: str or os.PathLike, the path to the zipped dataset file.

    Returns:
        None

    """
    if download > 0:
        urls = {
            'images': images_download_url,
            'annotations': annotations_download_url
        }
        download_and_extract(urls=urls, zipped_directory=zipped_directory, unzipped_directory=unzipped_directory)
    generate_dataset_registration_info(data_directory=unzipped_directory,
                                       info_path=os.path.join(unzipped_directory, 'FDDB.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', type=int, default=0)
    parser.add_argument('--unzipped_directory', type=str, default=os.path.join('data', 'raw', 'FDDB'))
    parser.add_argument('--zipped_directory', type=str, default=os.path.join('data', 'zipped', 'FDDB'))
    parser.add_argument('--images_download_url', type=str,
                        default='https://vis-www.cs.umass.edu/fddb/originalPics.tar.gz')
    parser.add_argument('--annotations_download_url', type=str,
                        default='https://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz')
    args = vars(parser.parse_args())
    main(**args)
