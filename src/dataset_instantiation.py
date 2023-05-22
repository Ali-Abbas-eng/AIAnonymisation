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
                   for index, file in tqdm(enumerate(files), total=len(files))]
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

