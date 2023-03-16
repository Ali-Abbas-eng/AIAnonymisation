import json
import deeplake
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools
from detectron2.structures import BoxMode


def download_dataset(url: str,
                     split: str,
                     dataset_name: str,
                     starting_index: int = 0,
                     category_id: int = 0,
                     data_directory: str = 'data'):
    """
    Download a dataset from a given URL, split, and dataset name.

    :param url: URL of the dataset
    :param split: Split of the dataset to download
    :param dataset_name: Name of the dataset
    :param starting_index: int, the index of the last element in the previously added dataset (in case of using multiple
    datasets in the process).
    :param category_id: int, the identification number of the class present in this dataset (0 for face 1 for license
    plate)
    :param data_directory: str, the directory to which data will be saved
    :return: None
    """
    dataset_dicts = []
    # Load the dataset using deeplake
    dataset = deeplake.load(url).pytorch(num_workers=2, batch_size=1, shuffle=False)

    # Create a directory to store the downloaded dataset if it does not exist
    data_directory = os.path.join(data_directory, dataset_name)
    os.makedirs(data_directory, exist_ok=True)
    os.makedirs(os.path.join(data_directory, split), exist_ok=True)

    # Iterate over the dataset, extract image data, and save it to the data directory
    for index, batch in tqdm(enumerate(dataset), total=len(dataset)):
        record = {}
        # Extract the image data and bounding boxes from the batch
        image = np.array(batch['images'][0]).astype(np.uint8)
        bboxes = np.array(batch['boxes']).astype(np.int32).flatten().tolist()

        # Create a path to save the image and save it using matplotlib
        image_path = os.path.join(data_directory, split, f'{index}.png')
        plt.imsave(image_path, image)

        record['file_name'] = image_path
        record['height'], record['width'] = image.shape[:2]
        record['image_id'] = index + starting_index
        annotations = []

        for i in range(0, len(bboxes), 4):
            x_min, y_min, h, w = bboxes[i: i + 4]
            x_max = x_min + h
            y_max = y_min + w
            poly = [
                (x_min, y_min), (x_max, y_min),
                (x_max, y_max), (x_min, y_max)
            ]
            poly = list(itertools.chain.from_iterable(poly))
            annotation = {
                'bbox': [x_min, y_min, x_max, y_max],
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': [poly],
                'category_id': category_id,
                'iscrowd': 0
            }
            annotations.append(annotation)
        record['annotations'] = annotations
        dataset_dicts.append(record)
    dataset_info_path = os.path.join(data_directory, f'{dataset_name}_{split}.json')
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_dicts, f)
        f.close()


if __name__ == '__main__':
    # Define command-line arguments using argparse
    parser = argparse.ArgumentParser(description='Download a dataset')
    parser.add_argument('--url', type=str, help='URL of the dataset')
    parser.add_argument('--split', type=str, help='Split of the dataset to download')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset')
    parser.add_argument('--starting_index', type=int, help='Last index in the previous dataset', default=0)
    parser.add_argument('--category_id', type=int, help='0 for faces, 1 for license plates', default=0)
    parser.add_argument('--data_directory', type=str, help='Where to save downloaded dataset', default='data')
    args = parser.parse_args()

    # Call the download_dataset function with named arguments
    download_dataset(url=args.url,
                     split=args.split,
                     dataset_name=args.dataset_name,
                     starting_index=args.starting_index,
                     data_directory=args.data_directory)
