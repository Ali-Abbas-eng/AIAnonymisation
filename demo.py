from detectron2.utils.logger import setup_logger
import os
import numpy as np
import cv2
import pandas as pd
from detectron2 import model_zoo
import data_prep
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from data_prep import get_data_records
import torch, torchvision
import logging
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo

setup_logger()
logging.getLogger('detectron2').setLevel(logging.WARNING)
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 4, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
train_data = get_data_records(split='train')
test_data = get_data_records(split='test')
val_data = get_data_records(split='val')
train_data_df = pd.DataFrame(train_data)
test_data_df = pd.DataFrame(test_data)
val_data_df = pd.DataFrame(val_data)


def annotate_images(data_point: pd.Series):
    image = plt.imread(data_point.loc['file_name'])
    for anno in data_point['annotations']:
        upper_left_corner = tuple(anno['bbox'][:2])
        bottom_right_corner = tuple(anno['bbox'][2:])
        cv2.rectangle(image, upper_left_corner, bottom_right_corner, (0, 255, 0), 2)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    return image

indexes = np.random.randint(low=0, high=train_data_df.shape[0], size=(8,)).astype(int)
sample_images = [annotate_images(data_point) for _, data_point in train_data_df.iloc[indexes].iterrows()]
sample_images = torch.as_tensor(sample_images)
sample_images = sample_images.permute(0, 3, 1, 2)
plt.figure(figsize=(24, 12))
grid_img = torchvision.utils.make_grid(sample_images, nrow=4)
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis('off')


class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name: str, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, False, output_folder)]
        return DatasetEvaluators(evaluators)


def train_and_evaluate_model(model_name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.DATASETS.TRAIN = ("FDDB_train",)
    cfg.DATASETS.TEST = ("FDDB_val",)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 10
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.EVAL_PERIOD = 5

    # Train the model.
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    networks = ['COCO-Detection/retinanet_R_50_FPN_1x.yaml',
                'COCO-Detection/retinanet_R_50_FPN_3x.yaml',
                'COCO-Detection/retinanet_R_101_FPN_3x.yaml',
                'COCO-Detection/rpn_R_50_C4_1x.yaml',
                'COCO-Detection/rpn_R_50_FPN_1x.yaml', ]
    try:
        data_prep.register()
    except AssertionError:
        pass
    for network in networks:
        train_and_evaluate_model(model_name=network)

