import os
from data_tools import download_files


def get_info():
    faste_r_cnn_variations = ['/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl',
                              '/faster_rcnn_R_50_DC5_1x/137847829/model_final_51d356.pkl',
                              '/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl',
                              '/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl',
                              '/faster_rcnn_R_50_DC5_3x/137849425/model_final_68d202.pkl',
                              '/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl',
                              '/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl',
                              '/faster_rcnn_R_101_DC5_3x/138204841/model_final_3e0943.pkl',
                              '/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl',
                              '/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl']
    faste_r_cnn_variations = [
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection' + url for url in faste_r_cnn_variations
    ]

    retina_net_variations = ['/retinanet_R_50_FPN_1x/190397773/model_final_bfca0b.pkl',
                             '/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl',
                             '/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl']
    retina_net_variations = [
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection' + url for url in retina_net_variations
    ]

    rpn_and_fast_r_cnn_variations = ['/rpn_R_50_C4_1x/137258005/model_final_450694.pkl',
                                     '/rpn_R_50_FPN_1x/137258492/model_final_02ce48.pkl',
                                     '/fast_rcnn_R_50_FPN_1x/137635226/model_final_e5f7ce.pkl']

    rpn_and_fast_r_cnn_variations = [
        'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection' + url for url in rpn_and_fast_r_cnn_variations
    ]

    faste_r_cnn_variations_dictionary = {url.split('/')[-3]: url for url in faste_r_cnn_variations}
    retina_net_variations_dictionary = {url.split('/')[-3]: url for url in retina_net_variations}
    rpn_and_fast_r_cnn_variations_dictionary = {url.split('/')[-3]: url for url in rpn_and_fast_r_cnn_variations}

    return [(faste_r_cnn_variations_dictionary, os.path.join('models', 'faste_r_cnn_variations')),
            (retina_net_variations_dictionary, os.path.join('models', 'retina_net_variations')),
            (rpn_and_fast_r_cnn_variations_dictionary, os.path.join('models', 'rpn_and_fast_r_cnn_variations'))]


if __name__ == '__main__':
    [download_files(urls_dict, directory) for urls_dict, directory in get_info()]
