from data_tools import register_dataset, visualize_sample, merge, DATASET_INFO_FILE_TEST, DATASET_INFO_FILE_TRAIN
from training_utils import get_cfg, Trainer
import argparse
from time import sleep
from tqdm.auto import tqdm
from detectron2.solver.build import defaultdict


def train(yaml_url: str,
          train_files: list,
          test_files: list,
          initial_learning_rate: float = 0.00025,
          train_steps: int = 120_000,
          eval_steps: int = 50_000,
          batch_size: int = 2,
          output_directory: str = 'output',
          decay_gamma: float = 0.7):
    """
    Trains a model using the specified configurations.
    :param yaml_url: The URL to the YAML configuration file.
    :param combine: int, can be one of [0, 1, 2], 0 if each data file represents one split of one dataset,
                                                  1 if the user wants to combine test files (recommended),
                                                  2 if the user wants to combine train and test files (not
    :param train_files: list, list of json files containing train dataset catalogues to be registered.
    :param test_files: list, list of json files containing train dataset catalogues to be registered.
    :param initial_learning_rate: The initial learning rate to use for training. Default is 0.00025.
    :param train_steps: The number of training steps to perform. Default is 120_000.
    :param eval_steps: The number of evaluation steps to perform. Default is 50_000.
    :param batch_size: The batch size to use for training. Default is 2.
    :param output_directory: str, the directory to which training results will be saved
    :param decay_gamma: float, decay step for the learning rate scheduler
    after one is already done training)
    """
    # register the datasets
    train_datasets = [file.split('/')[-1].replace('.json', '') for file in train_files]
    test_datasets = [file.split('/')[-1].replace('.json', '') for file in test_files]

    [register_dataset(file, file.split('/')[-1].replace('.json', '')) for file in [*train_files, *test_files]]
    [visualize_sample(file) for file in [*train_files, *test_files]]

    # Get configurations from specified parameters
    configurations = get_cfg(network_base_name=yaml_url.split('/')[-1].replace('.yaml', ''),
                             yaml_url=yaml_url,
                             train_datasets=tuple(train_datasets),
                             test_datasets=tuple(test_datasets),
                             decay_gamma=decay_gamma,
                             initial_learning_rate=initial_learning_rate,
                             train_steps=train_steps,
                             eval_freq=eval_steps,
                             batch_size=batch_size,
                             output_directory=output_directory)
    # Create trainer object with configurations
    trainer = Trainer(configurations)

    # Resume training if possible
    trainer.resume_or_load(True)

    # Train model
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_url', type=str, required=True)
    parser.add_argument('--train_files', nargs='+', default=['data/raw/CelebA/celeba_train.json',
                                                             'data/raw/CCPD2019/ccpd_train.json'])
    parser.add_argument('--test_files', nargs='+', default=['data/test.json'])
    parser.add_argument('--decay_gamma', type=float, default=0.7)
    parser.add_argument('--output_directory', type=str, default='output')
    parser.add_argument('--initial_learning_rate', type=float, default=1e-6)
    parser.add_argument('--train_steps', type=int, default=30_000)
    parser.add_argument('--eval_steps', type=int, default=10_000)
    parser.add_argument('--batch_size', type=int, default=2)

    args = vars(parser.parse_args())

    train(**args)
