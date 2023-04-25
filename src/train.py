from data_tools import register_dataset, visualize_sample
from training_utils import get_cfg, Trainer
import os
import argparse
from evaluation import evaluate


def train(yaml_url: str,
          train_files: list,
          test_files: list,
          initial_learning_rate: float,
          train_steps: int,
          eval_steps: int,
          batch_size: int,
          output_directory: str,
          decay_gamma: float,
          eval_device: str):
    """
    Trains a model using the specified configurations.
    :param yaml_url: The URL to the YAML configuration file.
    :param train_files: list, list of json files containing train dataset catalogues to be registered.
    :param test_files: list, list of json files containing train dataset catalogues to be registered.
    :param initial_learning_rate: The initial learning rate to use for training. Default is 0.00025.
    :param train_steps: The number of training steps to perform. Default is 120_000.
    :param eval_steps: The number of evaluation steps to perform. Default is 50_000.
    :param batch_size: The batch size to use for training. Default is 2.
    :param output_directory: str, the directory to which training results will be saved
    :param decay_gamma: float, decay step for the learning rate scheduler
    :param eval_device: str, the device on which evaluation of the model will be done, default 'cuda' (recommended).
    """
    # register the datasets
    train_datasets = [file.split('/')[-1].replace('.json', '') for file in train_files]
    test_datasets = [file.split('/')[-1].replace('.json', '') for file in test_files]

    [register_dataset(file, file.split('/')[-1].replace('.json', '')) for file in [*train_files, *test_files]]
    [visualize_sample(file) for file in [*train_files, *test_files]]
    network_base_name = yaml_url.split('/')[-1].replace('.yaml', '')
    # Get configurations from specified parameters
    configurations = get_cfg(network_base_name=network_base_name,
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

    evaluate(yaml_url=yaml_url,
             model_weights=os.path.join(output_directory, network_base_name, 'model_final.pth'),
             test_data_file=os.path.join('data', 'test.json'),
             output_dir=os.path.join(output_directory, network_base_name, 'test'),
             device=eval_device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_url', type=str, required=True)
    parser.add_argument('--train_files', nargs='+', default=['data/raw/CelebA/celeba_train.json',
                                                             'data/raw/CCPD2019/ccpd_train.json'])
    parser.add_argument('--test_files', nargs='+', default=['data/val.json'])
    parser.add_argument('--decay_gamma', type=float, default=0.1)
    parser.add_argument('--output_directory', type=str, default='output')
    parser.add_argument('--initial_learning_rate', type=float, default=0.00025)
    parser.add_argument('--train_steps', type=int, default=320_000)
    parser.add_argument('--eval_steps', type=int, default=50_000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_device', type=str, default='cuda')

    args = vars(parser.parse_args())

    train(**args)

