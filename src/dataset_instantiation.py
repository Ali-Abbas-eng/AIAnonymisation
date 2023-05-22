from .dataset import ImagesDataset
import os
from .data_tools import create_record
from tqdm.auto import tqdm


def get_ccpd2019_data():
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
                                   coco_file=None,
                                   urls=urls,
                                   cache_directory=os.path.join('data', 'cache', 'celeba'))
    celeba_default.generate_dataset_registration_info_params = {
        'data_directory': os.path.join('data', 'raw', 'CelebA'),
        'annotations_file': os.path.join('data', 'raw', 'CelebA', 'anno', 'list_bbox_celeba.txt')
    }
    celeba_default.registration_info_generator = generate_dataset_registration_info
    return celeba_default
