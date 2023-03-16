import os
import requests
from tqdm.auto import tqdm
import argparse


def download_files(urls: dict, directory: str = 'models'):
    """
    Downloads files from the given URLs and saves them to the given directory.

    Args:
        urls (dict): A dictionary of URLs to download.
        directory (str): The directory where the files should be saved.
    """
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Loop through each URL and download the file
    for key, url in urls.items():
        # Get the filename from the URL
        filename = os.path.join(directory, url.split('/')[-1])

        # Download the file
        response = requests.get(url, stream=True)

        # Get the size of the file and set the block size for the progress bar
        file_size = int(response.headers.get('Content-Length', 0))
        block_size = 1024

        # Create a progress bar for the download
        progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True, desc=f'Downloading {key.capitalize()}')

        # Loop through the response data and write it to a file while updating the progress bar
        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        # Close the progress bar
        progress_bar.close()

        # Print an error message if the file was not downloaded successfully
        if file_size != 0 and progress_bar.n != file_size:
            print(f"ERROR: Failed to download {filename}")


def get_info():
    faste_r_cnn_variations = [
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_1x/137847829/model_final_51d356.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/model_final_68d202.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/model_final_3e0943.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl',
    ]

    retina_net_variations = [
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_1x/190397773/model_final_bfca0b.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl',
    ]

    rpn_and_fast_r_cnn_variations = [
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_C4_1x/137258005/model_final_450694.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/rpn_R_50_FPN_1x/137258492/model_final_02ce48.pkl',
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/fast_rcnn_R_50_FPN_1x/137635226/model_final_e5f7ce.pkl',
    ]

    faste_r_cnn_variations_dictionary = {url.split('/')[-3]: url for url in faste_r_cnn_variations}
    retina_net_variations_dictionary = {url.split('/')[-3]: url for url in retina_net_variations}
    rpn_and_fast_r_cnn_variations_dictionary = {url.split('/')[-3]: url for url in rpn_and_fast_r_cnn_variations}

    return [(faste_r_cnn_variations_dictionary, os.path.join('models', 'faste_r_cnn_variations')),
            (retina_net_variations_dictionary, os.path.join('models', 'retina_net_variations')),
            (rpn_and_fast_r_cnn_variations_dictionary, os.path.join('models', 'rpn_and_fast_r_cnn_variations'))]


if __name__ == '__main__':
    [download_files(urls_dict, directory) for urls_dict, directory in get_info()]




