import warnings
from utils import register_dataset, get_cfg
import argparse
import os
from detectron2.engine.defaults import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from dataset import custom_data_mapper
import torch
from detectron2.data import transforms
from utils import path_fixer


class Trainer(DefaultTrainer):
    """
    A custom trainer class that inherits the DefaultTrainer class from Detectron2.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder: str = None):
        """
        A class method that builds a COCOEvaluator object for evaluation.

        Parameters:
            cfg (CfgNode): A configuration object.
            dataset_name (str): The name of the dataset to be evaluated.
            output_folder (str): The directory to save evaluation results. If None, it is saved in cfg.OUTPUT_DIR/eval.

        Returns:
            evaluator (COCOEvaluator): A COCOEvaluator object for evaluation.
        """

        # If output folder is not provided, create a new one in cfg.OUTPUT_DIR/eval
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'eval')
        os.makedirs(output_folder, exist_ok=True)

        # Create a COCOEvaluator object for evaluation
        evaluator = COCOEvaluator(dataset_name, cfg, False, output_folder)

        return evaluator

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )
        return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    @classmethod
    def build_train_loader(cls, cfg):

        return build_detection_train_loader(cfg, mapper=custom_data_mapper)


def train(network_base_name: str,
          train_files: list,
          valid_files: list,
          initial_learning_rate: float,
          train_steps: int,
          eval_steps: int,
          batch_size: int,
          output_directory: str,
          decay_freq: int,
          decay_gamma: float,
          min_learning_rate: float,
          freeze_at: int,
          roi_heads: int):
    """
    Trains a model using the specified configurations.
    :param network_base_name: The URL to the YAML configuration file.
    :param train_files: list, list of json files containing train dataset catalogues to be registered.
    :param valid_files: list, list of json files containing train dataset catalogues to be registered.
    :param initial_learning_rate: The initial learning rate to use for training. Default is 0.00025.
    :param train_steps: The number of training steps to perform. Default is 120_000.
    :param eval_steps: The number of evaluation steps to perform. Default is 50_000.
    :param batch_size: The batch size to use for training. Default is 2.
    :param output_directory: str, the directory to which training results will be saved.
    :param decay_freq: int, the frequency of decaying the learning rate.
    :param decay_gamma: float, decay step for the learning rate scheduler
    :param min_learning_rate: float, the minimum value for the learning rate.
    :param freeze_at: int, index of the last layer to be frozen in the sequence of frozen layer (0 means freeze all but the
    output layer, -1 means train all).
    :param roi_heads: int, number of Region Of Interest Heads in the output layer of the model.
    """
    # register the datasets
    train_sets = [file.split('/')[-1].replace('.json', '') for file in train_files]
    valid_sets = [file.split('/')[-1].replace('.json', '') for file in valid_files]

    [register_dataset(file, file.split('/')[-1].replace('.json', '')) for file in [*train_files, *valid_files]]
    yaml_url = f'COCO-Detection/{network_base_name}.yaml'
    # Get configurations from specified parameters
    configurations = get_cfg(network_base_name=network_base_name,
                             yaml_url=yaml_url,
                             train_datasets=tuple(train_sets),
                             test_datasets=tuple(valid_sets),
                             decay_gamma=decay_gamma,
                             decay_freq=decay_freq,
                             initial_learning_rate=initial_learning_rate,
                             train_steps=train_steps,
                             eval_freq=eval_steps,
                             batch_size=batch_size,
                             freeze_at=freeze_at,
                             output_directory=output_directory,
                             min_learning_rate=min_learning_rate,
                             roi_heads=roi_heads)
    trial_information = f'Network: {network_base_name}\n' \
                        f'Train Dataset(s): {train_files}\n' \
                        f'Test Dataset(s):{valid_files}\n' \
                        f'Decay Gamma: {decay_gamma}\n' \
                        f'Decay Frequency: {decay_freq}\n' \
                        f'Initial Learning Rate: {initial_learning_rate}\n' \
                        f'Train Steps: {train_steps}\n' \
                        f'Evaluation Frequency: {eval_steps}\n' \
                        f'Batch Size: {batch_size}\n' \
                        f'Freeze At: {freeze_at}\n' \
                        f'Minimum Learning Rate: {min_learning_rate}\n' \
                        f'Region of Interest (ROI) Heads: {roi_heads}\n'
    with open(os.path.join(path_fixer(configurations.OUTPUT_DIR), 'Trial Details.txt'), 'w') as file_handle:
        file_handle.write(trial_information)
        file_handle.close()

    # Create trainer object with configurations
    trainer = Trainer(configurations)

    trainer.resume_or_load(True)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_base_name', type=str, required=True)
    parser.add_argument('--train_files', nargs='+', default=['data/raw/CelebA/celeba_train_001.json',
                                                             'data/raw/CCPD2019/ccpd_train_001.json'])
    parser.add_argument('--valid_files', nargs='+', default=['data/val_001.json'])
    parser.add_argument('--decay_freq', type=int, default=10_000)
    parser.add_argument('--decay_gamma', type=float, default=0.5)
    parser.add_argument('--output_directory', type=str, default='output_trial_005')
    parser.add_argument('--initial_learning_rate', type=float, default=0.00025)
    parser.add_argument('--train_steps', type=int, default=50_000)
    parser.add_argument('--eval_steps', type=int, default=10_000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--min_learning_rate', type=float, default=1e-5)
    parser.add_argument('--freeze_at', type=int, default=0)
    parser.add_argument('--roi_heads', type=int, default=256)

    args = vars(parser.parse_args())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train(**args)

