import warnings

from data_tools import register_dataset, visualize_sample
from training_utils import get_cfg, Trainer
import os
import argparse
from evaluation import evaluate


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
          eval_device: str,
          min_learning_rate: float,
          freeze_at: int):
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
    :param eval_device: str, the device on which evaluation of the model will be done, default 'cuda' (recommended).
    :param min_learning_rate: float, the minimum value for the learning rate.
    :param freeze_at: int, index of the last layer to be frozen in the sequence of frozen layer (0 means freeze all but the
    output layer, -1 means train all).
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
                             output_directory=output_directory,
                             min_learning_rate=min_learning_rate,
                             freeze_at=freeze_at)

#     [visualize_sample(file, show=False, save_path=file.replace('json', 'png'))
#      for file in [*train_files, *valid_files]]
    # Create trainer object with configurations
    trainer = Trainer(configurations)

    # Resume training if possible
    trainer.resume_or_load(True)

    # Train model
    trainer.train()

    evaluate(network=network_base_name,
             model_weights=os.path.join(output_directory, network_base_name, 'model_final.pth'),
             test_data_file=os.path.join('data', 'test.json'),
             output_dir=os.path.join(output_directory, network_base_name, 'test'),
             device=eval_device)


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
    parser.add_argument('--eval_device', type=str, default='cuda')
    parser.add_argument('--min_learning_rate', type=float, default=1e-5)
    parser.add_argument('--freeze_at', type=int, default=0)

    args = vars(parser.parse_args())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train(**args)

