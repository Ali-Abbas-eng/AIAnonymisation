from .dataset import ImagesDataset
import os
from .data_tools import create_record
from tqdm.auto import tqdm
from typing import Union, Callable
import itertools


def get_ccpd2019_dataset():
    def generate_ccpd2019_registration_file(data_files_directory):
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

        files = [os.path.join(root, file)
                 for root, _, files in os.walk(data_files_directory)
                 for file in files
                 if file.endswith('.jpg')]

        # noinspection PyTypeChecker
        records = [create_record(image_path=file,
                                 bounding_boxes=decode_file_name(file),
                                 category_id=1,
                                 index=index)
                   for index, file in tqdm(enumerate(files), total=len(files), desc='Generating data from CCPD2019')
                   if decode_file_name(file) is not None]
        return records

    CCPD2019_DEFAULT = ImagesDataset(name='ccpd2019',
                                     path=os.path.join('data', 'raw'),
                                     coco_file=None,
                                     urls={
                                         'CCPD2019.tar.xz':
                                             'https://drive.google.com/uc?id=1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc'},
                                     cache_directory=os.path.join('data', 'cache', 'ccpd_2019'))
    CCPD2019_DEFAULT.generate_dataset_registration_info_params = {
        'data_files_directory': CCPD2019_DEFAULT.path
    }
    CCPD2019_DEFAULT.registration_info_generator = generate_ccpd2019_registration_file

    return CCPD2019_DEFAULT


def get_celeba_dataset():
    def generate_dataset_registration_info(data_directory: str or os.PathLike,
                                           annotations_file: str or os.PathLike):
        """
        Generates dataset records (list of dictionaries) and saves them to a json file.

        Args:
            data_directory (str or os.PathLike): The directory where the images are stored.
            annotations_file (str or os.PathLike): The path to the annotations file.
        Returns:
            None
        """
        # Read the annotations file and split it into lines
        annotations = open(annotations_file, 'r').read().split('\n')[2:]

        # Remove empty lines and extra spaces
        annotations = [item.replace("  ", ' ') for item in annotations if len(item) > 2]

        # Convert each line into a list of integers and strings
        annotations = [[int(item) if item.isnumeric() else item for item in line.split()] for line in annotations]

        with tqdm(total=len(annotations), desc='Generating Data From CelebA') as progress_bar:
            def generate_data_point(index, example):
                # Get the image path
                image_path = os.path.join(data_directory, 'img_celeba', example[0])

                # Get the bounding box coordinates
                x_min = example[1]
                y_min = example[2]
                x_max = example[1] + example[3]
                y_max = example[2] + example[4]
                bboxes = [x_min, y_min, x_max, y_max]
                # Create a record for the example
                example_record = create_record(image_path=image_path,
                                               index=index,
                                               bounding_boxes=bboxes,
                                               category_id=0)
                progress_bar.update()
                # progress_bar.update()
                return example_record

            # Create an empty list to store the dataset records
            dataset_records = [generate_data_point(index, line) for index, line in enumerate(annotations)]
            return dataset_records
    urls = {
        'img_celba.7z': 'https://drive.google.com/drive/folders/1eyJ52hX5KpniS9MB-MFXBRtJUDpeldqx?usp=share_link',
        'list_bbox_celeba.txt': 'https://drive.google.com/uc?id=19X0GE3kP6tNatS9kZ2-Ks2_OeeCtqeFI'
    }

    celeba_default = ImagesDataset(name='celeba',
                                   path=os.path.join('data', 'raw', 'CelebA'),
                                   coco_file=os.path.join('data', 'raw', 'celeba.json'),
                                   urls=urls,
                                   cache_directory=os.path.join('data', 'cache', 'celeba'))
    celeba_default.generate_dataset_registration_info_params = {
        'data_directory': os.path.join('data', 'raw', 'CelebA'),
        'annotations_file': os.path.join('data', 'raw', 'CelebA', 'anno', 'list_bbox_celeba.txt')
    }
    celeba_default.registration_info_generator = generate_dataset_registration_info
    return celeba_default


def get_wider_face_dataset():
    def generate_dataset_registration_info(data_directory: str or os.PathLike,
                                           annotation_files: str or os.PathLike):
        """
        This function generates dataset registration information from the annotations file and the data directory.

        Args:
            data_directory (str or os.PathLike): The directory containing the data.
            annotation_files (list), list of paths to the annotation files

        Returns:
            dataset_dicts (list): A list of dictionaries containing the dataset registration information.
        """
        annotations = open(annotation_files[0], 'r').read() + '\n' + open(annotation_files[1], 'r').read()
        # Read the annotations file and split it by new line
        annotations = annotations.split('\n')

        lines_of_interest = [line for line in annotations if line.endswith('.jpg')]
        with tqdm(total=len(lines_of_interest), desc='Generating Data From Wider Face') as progress_bar:
            def generate_data_point(row, image_id):
                # Check if the line ends with '.jpg'
                if row.endswith('.jpg'):
                    idx = annotations.index(row)
                    # Get the file internal path and image path
                    file_internal_path = os.path.join(*row.split('/'))
                    image_path = os.path.join(data_directory, file_internal_path)

                    # Get the number of images and bounding boxes
                    num_faces = int(annotations[idx + 1])
                    bboxes = []
                    for i in range(idx + 2, idx + 2 + num_faces):
                        bbox = [int(coordinate) for coordinate in annotations[i].split()[:4]]
                        bbox[2] += bbox[0]
                        bbox[3] += bbox[1]
                        bboxes.append(bbox)
                    bboxes = list(itertools.chain.from_iterable(bboxes))

                    # Create the record and append it to the dataset dictionary
                    record = create_record(image_path=image_path,
                                           bounding_boxes=bboxes,
                                           index=image_id,
                                           category_id=0)
                    progress_bar.update()
                    return record

            dataset_dicts = [generate_data_point(line, index) for index, line in enumerate(lines_of_interest)]
        return dataset_dicts

    urls = {
        'WIDER_train.zip': 'https://drive.google.com/uc?id=1w6lLpq6Sh10okRA6bSBqcDEDb-2fK_nc',
        'WIDER_val.zip': 'https://drive.google.com/uc?id=1wb5jtFTHd9yBZpYpUVO50hb5ofa2NOm3',
        'wider_face_split.zip': 'https://drive.google.com/uc?id=1KcRtgcLprJBnhKpkEkC-FwBdXrdb_nsv',
    }

    wider_face_default = ImagesDataset(name='wider_face',
                                       path=os.path.join('data', 'raw', 'WiderFace'),
                                       coco_file=os.path.join('data', 'raw', 'wider_face.json'),
                                       urls=urls,
                                       cache_directory=os.path.join('data', 'cache', 'WiderFace'),
                                       )
    wider_face_default.generate_dataset_registration_info_params = {
        'data_directory': os.path.join('data', 'raw', 'WiderFace'),
        'annotation_files': [os.path.join(wider_face_default.path, 'wider_face_split', 'wider_face_train_bbx_gt.txt'),
                             os.path.join(wider_face_default.path, 'wider_face_split', 'wider_face_val_bbx_gt.txt')]
    }
    wider_face_default.registration_info_generator = generate_dataset_registration_info
    return wider_face_default
