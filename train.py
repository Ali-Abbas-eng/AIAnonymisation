from training_utils import register_datasets, get_cfg, Trainer
import argparse


def train(network_base_name: str,
          weights_path: str,
          yaml_url: str,
          initial_learning_rate: float = 0.00025,
          train_steps: int = 5000,
          eval_steps: int = 5000,
          batch_size: int = 2,
          max_patience: int = 50,
          learning_rate_decay_factor: float = .9):
    """
    Trains a model using the specified configurations.

    :param network_base_name: The name of the network base to use.
    :param weights_path: The path to the weights file.
    :param yaml_url: The URL to the YAML configuration file.
    :param initial_learning_rate: The initial learning rate to use for training. Default is 0.00025.
    :param train_steps: The number of training steps to perform. Default is 5000.
    :param eval_steps: The number of evaluation steps to perform. Default is 5000.
    :param batch_size: The batch size to use for training. Default is 2.
    :param max_patience: The maximum number of steps without improvement before stopping training. Default is 50.
    :param learning_rate_decay_factor: The factor by which to decay the learning rate. Default is .9.

    """

    # Get configurations from specified parameters
    configurations = get_cfg(network_base_name=network_base_name,
                             weights_path=weights_path,
                             yaml_url=yaml_url,
                             initial_learning_rate=initial_learning_rate,
                             train_steps=train_steps,
                             eval_freq=eval_steps,
                             batch_size=batch_size,
                             max_patience=max_patience,
                             learning_rate_decay_factor=learning_rate_decay_factor)

    # Create trainer object with configurations
    trainer = Trainer(configurations)

    # Train model
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_base_name', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--yaml_url', type=str, required=True)
    parser.add_argument('--initial_learning_rate', type=float, default=0.00025)
    parser.add_argument('--train_steps', type=int, default=5000)
    parser.add_argument('--eval_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_patience', type=int, default=50)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=.9)

    args = vars(parser.parse_args())

    train(**args)
