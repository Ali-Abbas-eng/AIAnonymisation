import itertools
import os
import detectron2.structures
import data_tools
import json
from tqdm.auto import tqdm


def generate_dataset_registration_info(data_directory: str or os.PathLike = None,
                                       annotations_file: str or os.PathLike = None):
    """
    This function generates dataset registration information from the annotations file and the data directory.

    Args:
        data_directory (str or os.PathLike): The directory containing the data.
        annotations_file (str or os.PathLike): The annotations file.

    Returns:
        dataset_dicts (list): A list of dictionaries containing the dataset registration information.
    """
    # Read the annotations file and split it by new line
    annotations = open(annotations_file, 'r').read().split('\n')

    # Initialize the dataset dictionary and file id
    dataset_dicts = []
    file_id = 0

    # Loop through the annotations
    for index, line in tqdm(enumerate(annotations), total=len(annotations)):
        # Check if the line ends with '.jpg'
        if line.endswith('.jpg'):
            # Get the file internal path and image path
            file_internal_path = os.path.join(*line.split('/'))
            image_path = os.path.join(data_directory, file_internal_path)

            # Get the number of images and bounding boxes
            num_faces = int(annotations[index + 1])
            bboxes = []
            for i in range(index + 2, index + 2 + num_faces):
                bbox = [int(coordinate) for coordinate in annotations[i].split()[:4]]
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                bboxes.append(bbox)
            bboxes = list(itertools.chain.from_iterable(bboxes))

            # Create the record and append it to the dataset dictionary
            record = data_tools.create_record(image_path=image_path,
                                              bounding_boxes=bboxes,
                                              index=file_id,
                                              bbox_format=detectron2.structures.BoxMode.XYXY_ABS,
                                              category_id=0)
            dataset_dicts.append(record)

    return dataset_dicts


def write_data(info_path: str = data_tools.WIDER_FACE_INFORMATION_FILE):
    """
    This function writes the data to the information file.

    Args:
        info_path (str): The path of the information file.
    """
    # Generate the dataset registration information for the training and validation data
    data1 = generate_dataset_registration_info(data_directory=data_tools.WIDER_FACE_IMAGES_DIRECTORY_TRAIN,
                                               annotations_file=data_tools.WIDER_FACE_ANNOTATIONS_FILE_TRAIN)
    data2 = generate_dataset_registration_info(data_directory=data_tools.WIDER_FACE_IMAGES_DIRECTORY_VALID,
                                               annotations_file=data_tools.WIDER_FACE_ANNOTATIONS_FILE_VALID)

    # Combine the training and validation data
    data1.extend(data2)

    # Write the data to the information file
    json.dump(data1, open(info_path, 'w'))


if __name__ == "__main__":
    write_data()