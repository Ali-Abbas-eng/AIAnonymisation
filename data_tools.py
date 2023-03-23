from tqdm.auto import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from detectron2.structures import BoxMode

CELEB_A_IMAGES_DIRECTORY = os.path.join('data', 'raw', 'CelebA', 'img_celeba')
CELEB_A_ANNOTATIONS_FILE = os.path.join('data', 'raw', 'CelebA', 'anno', 'list_bbox_celeba.txt')
CELEB_A_INFORMATION_FILE = os.path.join('data', 'raw', 'CelebA', 'celeba_info.json')

WIDER_FACE_IMAGES_DIRECTORY_TRAIN = os.path.join('data', 'raw', 'WIDER FACE', 'WIDER_train', 'images')
WIDER_FACE_ANNOTATIONS_FILE_TRAIN = os.path.join('data', 'raw', 'WIDER FACE', 'wider_face_split',
                                                 'wider_face_train_bbx_gt.txt')
WIDER_FACE_IMAGES_DIRECTORY_VALID = os.path.join('data', 'raw', 'WIDER FACE', 'WIDER_val', 'images')
WIDER_FACE_ANNOTATIONS_FILE_VALID = os.path.join('data', 'raw', 'WIDER FACE', 'wider_face_split',
                                                 'wider_face_val_bbx_gt.txt')

WIDER_FACE_INFORMATION_FILE = os.path.join('data', 'raw', 'WIDER FACE', 'wider_face.json')


CELEB_A_NUM_TRAIN_CANDIDATES = 200
CELEB_A_NUM_TEST_CANDIDATES = 100
CELEB_A_NUM_VAL_CANDIDATES = 100

WIDER_FACE_NUM_TRAIN_CANDIDATES = 200
WIDER_FACE_NUM_TEST_CANDIDATES = 100
WIDER_FACE_NUM_VAL_CANDIDATES = 100

CCPD_NUM_TRAIN_CANDIDATES = 200
CCPD_NUM_TEST_CANDIDATES = 100
CCPD_NUM_VAL_CANDIDATES = 100


FINAL_DATA_PATH = 'data'
FACE_DATA_PATH = os.path.join(FINAL_DATA_PATH, 'face')
CCPD_DATA_PATH = os.path.join(FINAL_DATA_PATH, 'license_plates')
DATASET_INFO_FILE = os.path.join(FINAL_DATA_PATH, 'data.json')


CCPD_IMAGES_DIRECTORY = os.path.join('data', 'raw', 'CCPD2019')
CCPD_INFO_PATH = os.path.join('data', 'raw', 'CCPD2019', 'CCPD2019.json')



def create_record(image_path: str, bounding_boxes: list, index: int, category_id: int = 0):
    """
    This function creates a record for a single image.

    Args:
        image_path (str): The path of the image.
        bounding_boxes (list): The list of bounding boxes.
        index (int): The index of the image.
        category_id (int): The category id.

    Returns:
        record (dict): The record for the image.
    """
    # Initialize the record dictionary and read the image
    record = {}
    image = plt.imread(image_path)

    # Add the image path, height, width, and index to the record dictionary
    record['file_name'] = image_path
    record['height'], record['width'] = image.shape[:2]
    record['image_id'] = index

    # Initialize the annotations list and loop through the bounding boxes
    annotations = []
    for i in range(0, len(bounding_boxes), 4):
        # Get the bounding box coordinates
        x_min, y_min, h, w = bounding_boxes[i: i + 4]
        x_max = x_min + h
        y_max = y_min + w

        # Create the polygon and flatten it
        poly = [
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max)
        ]
        poly = list(itertools.chain.from_iterable(poly))

        # Create the annotation dictionary and append it to the annotations list
        annotation = {
            'bbox': [x_min, y_min, x_max, y_max],
            'bbox_mode': BoxMode.XYXY_ABS,
            'segmentation': [poly],
            'category_id': category_id,
            'iscrowd': 0
        }
        annotations.append(annotation)

    # Add the annotations to the record dictionary and return the record
    record['annotations'] = annotations
    return record