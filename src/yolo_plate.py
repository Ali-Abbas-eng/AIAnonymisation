import json
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
from utils import create_record, path_fixer
from argparse import ArgumentParser
from typing import Union, List, Tuple, AnyStr
from os import PathLike


def convert_text_information_to_bounding_box(text: List, shape: Tuple) -> List:
    """
    converts a string that contains the information about multiple bounding boxes to a list of coordinates representing
    those bounding boxes in a way easily readable by detectron2
    Args:
        text: list, list of one-line strings, each line MUST contain a list of 5 space-separated values.
        shape: tuple, the shape (height, width) of the image.

    Returns:
        List: a list of 4 * n elements where n is the number of lines (elements in the provided list),
        and 4 represents the XYXY_ABS format of the bounding box coordinates
    """
    # initiate an empty list to store the bounding boxes
    bounding_boxes = []

    # reverse the shape of the image from (height, width) to (width, height)
    shape = [shape[1], shape[0]]

    # iterate through lines of text
    for line in text:
        # split the line by the space to get the 5 elements representing the bounding box in YOLO format
        line = line.split()

        # get the x of the center of the bounding box
        x = int(shape[0] * float(line[1]))

        # get the y of the center of the bounding box
        y = int(shape[1] * float(line[2]))

        # get the height of the bounding box
        h = int(shape[0] * float(line[3]))

        # get the width of the bounding box
        w = int(shape[1] * float(line[4]))

        # add the newly formatted bounding box to the list of bounding boxes
        bounding_boxes.extend([x + h // 2, y + w // 2, x - h // 2, y - w // 2])
    return bounding_boxes


def get_info(data_directory: Union[AnyStr, PathLike]) -> None:
    """
    loops through the directory which contains the images and their corresponding text file holding the bounding boxes
    information, reformat the bounding box into the COCO format to be readable by detectron2, writes the new information
    to a json file
    Args:
        data_directory: str, the directory which contains the dataset, one level before the directory of the images
            i.e.:
                data
                    raw
                        YOLO_PLATE  <---- this is the directory to be passed
                            yolo_plate_dataset
                                image1.jpg
                                image1.txt
                                image2.jpg
                                image2.txt
                                ....
                            yolo_plate.json
    Returns:
        None, write the information directly to a json file
    """
    # get the images directory by appending 'yolo_plate_dataset' to the data_directory path
    images_directory = os.path.join(data_directory, 'yolo_plate_dataset')

    # get the files of interest (for each .jpg file there a .txt file with the same base name)
    files = [file for file in os.listdir(images_directory) if file.endswith('.jpg')]

    # initialise an empty list of records
    records = []

    # noinspection PyTypeChecker
    # iterate through the list of files (images)
    for index, file in tqdm(enumerate(files), total=len(files), desc='Generating Information from YOLO Plate dataset'):
        # get the image path
        image_path = path_fixer(os.path.join(images_directory, file))
        # read the image
        image = plt.imread(image_path)
        # read the corresponding text file
        bounding_boxes = open(os.path.join(images_directory, file[:-4] + '.txt')).read()
        # split line by line
        bounding_boxes = bounding_boxes.split('\n')
        # remove the last line that is just a new line with an empty string
        bounding_boxes.remove('')
        # get the new format of the bounding boxes
        bounding_boxes = convert_text_information_to_bounding_box(bounding_boxes, image.shape)
        # create a record that incorporates the COCO format
        record = create_record(image_path=path_fixer(image_path),
                               index=index,
                               bounding_boxes=bounding_boxes,
                               category_id=1)
        # add the new record to the list of records
        records.append(record)

    # write the list of COCO-Formatted records to a file with the same name of the dataset
    json.dump(records, open(os.path.join(data_directory, 'yolo_plate.json'), 'w'))


if __name__ == '__main__':
    # initialise and argument parser
    parser = ArgumentParser()
    # add the data_directory to the command line arguments with a default and preferred value
    parser.add_argument('--data_directory', type=str, default=os.path.join('data', 'raw', 'YOLO_PLATE'))
    # generate a dictionary with the corresponding keys
    args = vars(parser.parse_args())
    # invoke the dataset generation function
    get_info(**args)
