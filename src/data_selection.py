import random
import data_tools
from tqdm.auto import tqdm
import json
import os
import shutil
import numpy as np
from typing import List, Dict


def modify_record(record, new_index, new_path):
    """
    Modify the record with new index and new path.

    Args:
        record: A dictionary containing the record information.
        new_index: An integer representing the new index.
        new_path: A string representing the new path.

    Returns:
        A dictionary containing the modified record information.
    """
    record['file_name'] = new_path
    record['image_id'] = new_index
    return record


def generate_data_split(info: List[Dict],
                        split: str,
                        indexes: np.ndarray,
                        start_index: int,
                        file_base_name: str,
                        data_directory: str):
    """
    Generates data for a given split.

    Args:
        info: list of dictionaries representing the dataset
        split: A string representing the split.
        indexes: List, a list of indexes to get from the dataset.
        start_index: int, an integer representing the number from which to start giving incremental IDs
        file_base_name: str, the beginning of the new file name to be saved in another folder
        data_directory: str, the directory to which new files will be saved

    Returns:
        A list containing the data for the given split.
    """
    data = []
    for i, index in tqdm(enumerate(indexes), total=len(indexes), desc=f'Generating Data ({file_base_name}_{split})'):
        # Get the file name and path.
        file_name = info[index]['file_name'].split(os.path.sep)[-1]
        file_name = f'{file_base_name}_{file_name}'
        file_path = os.path.join(data_directory, file_name)

        # Copy the file to the face data path.
        shutil.copyfile(info[index]['file_name'], file_path)

        # Modify the record and append it to the data list.
        new_record = modify_record(record=info[index], new_index=i + start_index, new_path=file_path)
        data.append(new_record)

    return data


def select_from_data(json_file: str,
                     start_indexes: dict,
                     new_file_base_name: str,
                     num_examples: dict,
                     output_directory: str):
    """
        Selects data from CelebA dataset.

        Args:
            json_file: A string representing the path to the CelebA information file.
            start_indexes: A dictionary containing the start index for each split.
            new_file_base_name: str, the string at the beginning of each new dataset file.
            num_examples: dict, a dictionary containing the number of examples in each data split.
            output_directory: str, the directory to which new data files will be saved.

        Returns:
            Three lists containing the data for train, test and validation splits.
        """
    assert type(start_indexes) == dict
    assert list(start_indexes.keys()) == ['train', 'test', 'val'], f'Unknown keys in start_indexes\n' \
                                                                   f'\t{start_indexes.keys()}'
    assert list(num_examples.keys()) == ['train', 'test', 'val'], f'Unknown keys in num_examples\n' \
                                                                  f'\t{num_examples.keys()}'

    # Load the dataset information file.
    info = json.load(open(json_file, 'r'))

    # creat an indexer to randomly select files from the dataset
    indexer = np.random.permutation(len(info))

    # Create the face data path if it doesn't exist.
    os.makedirs(output_directory, exist_ok=True)

    # initialise an empty list to hold the file names that hold the data splits temporarily
    files = {}

    for split in start_indexes.keys():
        # Generate data for train, test and validation splits.
        indexes = indexer[:num_examples[split]]
        data = generate_data_split(info=info,
                                   split=split,
                                   indexes=indexes,
                                   start_index=start_indexes[split],
                                   file_base_name=new_file_base_name,
                                   data_directory=output_directory)
        indexer = indexer[num_examples[split]:]
        temp_file = os.path.join(data_tools.FINAL_DATA_PATH, f'{new_file_base_name}_{split}_temp.json')
        json.dump(data, open(temp_file, 'w'))
        files[split] = temp_file

    return files


def merge(files):
    for key, list_of_files in files.items():
        data = []
        for file in list_of_files:
            new_data = json.load(open(file, 'r'))
            data.extend(new_data)
        json.dump(data, open(os.path.join(data_tools.FINAL_DATA_PATH, f'{key}_info.json'), 'w'))


def select_candidates():
    celeb_a_files = select_from_data(json_file=data_tools.CELEB_A_INFORMATION_FILE,
                                     start_indexes={'train': 0, 'test': 0, 'val': 0},
                                     new_file_base_name='celeb_a',
                                     num_examples=data_tools.CELEB_A_NUM_CANDIDATES,
                                     output_directory=data_tools.IMAGES_DATA_DIRECTORY)

    wider_face_files = select_from_data(json_file=data_tools.WIDER_FACE_INFORMATION_FILE,
                                        start_indexes=data_tools.CELEB_A_NUM_CANDIDATES,
                                        new_file_base_name='wider_face',
                                        num_examples=data_tools.WIDER_FACE_NUM_CANDIDATES,
                                        output_directory=data_tools.IMAGES_DATA_DIRECTORY)

    ccpd_start_indexes = {
        key: data_tools.CELEB_A_NUM_CANDIDATES[key] + data_tools.WIDER_FACE_NUM_CANDIDATES[key]
        for key in ['train', 'test', 'val']
    }
    ccpd_2019_files = select_from_data(json_file=data_tools.CCPD_INFORMATION_FILE,
                                       start_indexes=ccpd_start_indexes,
                                       new_file_base_name='ccpd_2019',
                                       num_examples=data_tools.CCPD_NUM_CANDIDATES,
                                       output_directory=data_tools.IMAGES_DATA_DIRECTORY)

    merge({key: [celeb_a_files[key], wider_face_files[key], ccpd_2019_files[key]]
           for key in ['train', 'test', 'val']})

    [os.remove(os.path.join(data_tools.FINAL_DATA_PATH, file))
     for file in os.listdir(data_tools.FINAL_DATA_PATH) if 'temp' in file]


def visualize(json_file: str or os.PathLike = os.path.join('../data', 'val_info.json')):
    import matplotlib.pyplot as plt
    import cv2
    data = json.load(open(json_file))
    for i in range(10):
        data_point = random.choice(data)
        image = plt.imread(data_point['file_name'])
        for a in data_point['annotations']:
            cv2.rectangle(image, tuple(a['bbox'][:2]), tuple(a['bbox'][2:]), (0, 0, 255), 2)
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    select_candidates()
    visualize()
