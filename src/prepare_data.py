import ccpd_2019
import celeba
import wider_face
import argparse
from data_tools import *


def download_and_extract(compressed_files_directory: str or os.PathLike,
                         data_directory: str or os.PathLike = os.path.join('../data', 'raw')):
    ccpd_2019.download_and_extract(download_directory=compressed_files_directory,
                                   unzipped_directory=data_directory)
    celeba.download_and_extract(download_directory=os.path.join(compressed_files_directory, 'img_celeba.7z'),
                                unzipped_directory=CELEB_A_IMAGES_DIRECTORY)
    wider_face.download_and_extract(download_directory=os.path.join(compressed_files_directory, 'WIDER_FACE'),
                                    unzipped_directory=os.path.join(data_directory, 'WIDER_FACE'))


def main(compressed_files_directory: str or os.PathLike,
         data_directory: str or os.PathLike = os.path.join('../data', 'raw'),
         download: int = 0,
         combine: int = 1,
         pre_process: int = 0):
    """
    Encapsulation of the data retrieval process (Downloads, extracts, and then generates the final dataset to be used.
    :param compressed_files_directory: str, the base directory to which downloaded files will be saved.
    :param data_directory: str, the base directory to which extracted images will be saved
    :param download: int, whither to download the datasets as zip files.
    :param combine: int, to what extent to merge the dataset files:
                        0: don't merge anything
                        1: merge test files only
                        2: merge all files
    :param pre_process: int, whither to pre-process the dataset before writing to disk.
    :return:
    """
    if download > 0:
        download_and_extract(compressed_files_directory=compressed_files_directory,
                             data_directory=data_directory)
    pre_processing_function = pre_process_data if pre_process > 0 else None
    ccpd_2019.generate_dataset_registration_info(data_directory=CCPD_IMAGES_DIRECTORY,
                                                 info_path=CCPD_INFORMATION_FILE,
                                                 create_record=create_record,
                                                 pre_process=pre_processing_function)
    celeba.generate_dataset_registration_info(data_directory=CELEB_A_DATASET_DIRECTORY,
                                              annotations_file=CELEB_A_ANNOTATIONS_FILE,
                                              info_path=CELEB_A_INFORMATION_FILE,
                                              create_record=create_record,
                                              pre_process=pre_processing_function)

    wider_face.write_data(data_directory_train=WIDER_FACE_IMAGES_DIRECTORY_TRAIN,
                          data_directory_valid=WIDER_FACE_IMAGES_DIRECTORY_VALID,
                          annotation_file_train=WIDER_FACE_ANNOTATIONS_FILE_TRAIN,
                          annotation_file_valid=WIDER_FACE_ANNOTATIONS_FILE_VALID,
                          create_record=create_record,
                          info_path=WIDER_FACE_INFORMATION_FILE,
                          pre_process=pre_processing_function)
    generate_splits(directory=CCPD_IMAGES_DIRECTORY,
                    original_json=CCPD_INFORMATION_FILE,
                    num_examples=CCPD_NUM_CANDIDATES,
                    dataset_name='ccpd')
    generate_splits(directory=CELEB_A_DATASET_DIRECTORY,
                    original_json=CELEB_A_INFORMATION_FILE,
                    num_examples=CELEB_A_NUM_CANDIDATES,
                    dataset_name='celeba')
    generate_splits(directory=WIDER_FACE_IMAGES_DIRECTORY,
                    original_json=WIDER_FACE_INFORMATION_FILE,
                    num_examples=WIDER_FACE_NUM_CANDIDATES,
                    dataset_name='wider_face')
    train_files = [os.path.join(CCPD_IMAGES_DIRECTORY, 'ccpd_train.json'),
                   os.path.join(CELEB_A_DATASET_DIRECTORY, 'celeba_train.json'),
                   os.path.join(WIDER_FACE_IMAGES_DIRECTORY, 'wider_face_train.json')]
    test_files = [os.path.join(CCPD_IMAGES_DIRECTORY, 'ccpd_test.json'),
                  os.path.join(CCPD_IMAGES_DIRECTORY, 'ccpd_val.json'),
                  os.path.join(CELEB_A_DATASET_DIRECTORY, 'celeba_test.json'),
                  os.path.join(CELEB_A_DATASET_DIRECTORY, 'celeba_val.json'),
                  os.path.join(WIDER_FACE_IMAGES_DIRECTORY, 'wider_face_test.json'),
                  os.path.join(WIDER_FACE_IMAGES_DIRECTORY, 'wider_face_val.json')]
    if combine > 0:
        merge(test_files, DATASET_INFO_FILE_TEST)
    if combine > 1:
        merge(train_files, DATASET_INFO_FILE_TRAIN)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', type=int, default=0)
    parser.add_argument('--compressed_files_directory', type=str, default=os.path.join('data', 'zipped'))
    parser.add_argument('--data_directory', type=str, default=os.path.join('data', 'raw'))
    parser.add_argument('--pre_process', type=int, default=0)
    parser.add_argument('--combine', type=int, default=1)
    args = vars(parser.parse_args())
    main(**args)
