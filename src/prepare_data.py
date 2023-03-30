import ccpd_2019
import celeba
from data_tools import *


def main(compressed_files_directory: str or os.PathLike,
         data_directory: str or os.PathLike = os.path.join('../data', 'raw')):
    """
    Encapsulation of the data retrieval process (Downloads, extracts, and then generates the final dataset to be used.
    :param compressed_files_directory: str, the base directory to which downloaded files will be saved.
    :param data_directory: str, the base directory to which extracted images will be saved
    :return:
    """
    ccpd_2019.download_and_extract(download_directory=compressed_files_directory,
                                   unzipped_directory=data_directory)
    ccpd_2019.generate_dataset_registration_info(data_directory=CCPD_IMAGES_DIRECTORY,
                                                 info_path=CCPD_INFORMATION_FILE,
                                                 create_record=create_record)

    celeba.download_and_extract(download_directory=os.path.join(compressed_files_directory, 'img_celeba.7z'),
                                unzipped_directory=CELEB_A_IMAGES_DIRECTORY)
    celeba.generate_dataset_registration_info(data_directory=CELEB_A_IMAGES_DIRECTORY,
                                              annotations_file=CELEB_A_ANNOTATIONS_FILE,
                                              info_path=CELEB_A_INFORMATION_FILE,
                                              create_record=create_record)

    wider_face.download_and_extract(download_directory=os.path.join(compressed_files_directory, 'WIDER_FACE'),
                                    unzipped_directory=os.path.join(data_directory, 'WIDER_FACE'))

    wider_face.write_data(data_directory_train=WIDER_FACE_IMAGES_DIRECTORY_TRAIN,
                          data_directory_valid=WIDER_FACE_IMAGES_DIRECTORY_VALID,
                          annotation_file_train=WIDER_FACE_ANNOTATIONS_FILE_TRAIN,
                          annotation_file_valid=WIDER_FACE_ANNOTATIONS_FILE_VALID,
                          create_record=create_record,
                          info_path=WIDER_FACE_INFORMATION_FILE)

    select_candidates()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compressed_files_directory', type=str, default=os.path.join('../data', 'zipped'))
    parser.add_argument('--data_directory', type=str, default=os.path.join('../data', 'raw'))
    args = vars(parser.parse_args())
    main(**args)
