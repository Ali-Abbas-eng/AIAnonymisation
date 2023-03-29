from data_tools import register_datasets
from training_utils import get_cfg, Trainer, EarlyStoppingHook
import argparse


def train(network_base_name: str,
          weights_path: str,
          yaml_url: str,
          thing_classes: list,
          data_directory: str = 'data',
          initial_learning_rate: float = 0.00025,
          train_steps: int = 120_000,
          eval_steps: int = 50_000,
          checkpoints_freq: int = 40_000,
          log_freq: int = 5000,
          batch_size: int = 2,
          output_directory: str = 'output'):
    """
    Trains a model using the specified configurations.

    :param network_base_name: The name of the network base to use.
    :param weights_path: The path to the weights file.
    :param yaml_url: The URL to the YAML configuration file.
    :param thing_classes: list, a list of strings representing the classes in the dataset
    :param data_directory: str, the directory that holds the datasets
    :param initial_learning_rate: The initial learning rate to use for training. Default is 0.00025.
    :param train_steps: The number of training steps to perform. Default is 120_000.
    :param eval_steps: The number of evaluation steps to perform. Default is 50_000.
    :param checkpoints_freq: int, the frequency at which to make a model checkpoint. Default is 40_000
    :param log_freq: int, the frequency at which to log training details. Default is 5000
    :param batch_size: The batch size to use for training. Default is 2.
    :param output_directory: str, the directory to which training results will be saved
    """

    # register the datasets
    register_datasets(data_directory=data_directory, thing_classes=thing_classes)

    # Get configurations from specified parameters
    configurations = get_cfg(network_base_name=network_base_name,
                             weights_path=weights_path,
                             yaml_url=yaml_url,
                             initial_learning_rate=initial_learning_rate,
                             train_steps=train_steps,
                             eval_freq=eval_steps,
                             checkpoints_freq=checkpoints_freq,
                             log_freq=log_freq,
                             batch_size=batch_size,
                             output_directory=output_directory)

    # Create trainer object with configurations
    trainer = Trainer(configurations)

    # register early stopping hook
    trainer.register_hooks([EarlyStoppingHook(patience=10)])

    # Train model
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_base_name', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--yaml_url', type=str, required=True)
    parser.add_argument('--data_directory', type=str, default='data')
    parser.add_argument('--output_directory', type=str, default='output')
    parser.add_argument('--thing_classes', type=list, default=['FACE', 'LP'])
    parser.add_argument('--initial_learning_rate', type=float, default=0.00025)
    parser.add_argument('--train_steps', type=int, default=160_000)
    parser.add_argument('--eval_steps', type=int, default=50_000)
    parser.add_argument('--log_freq', type=int, default=500)
    parser.add_argument('--checkpoint_freq', type=int, default=40_000)
    parser.add_argument('--batch_size', type=int, default=4)

    args = vars(parser.parse_args())

    train(**args)
