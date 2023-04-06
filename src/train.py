from data_tools import register_dataset, visualize_sample, merge, DATASET_INFO_FILE_TEST, DATASET_INFO_FILE_TRAIN
from training_utils import get_cfg, Trainer
import argparse
from time import sleep
from tqdm.auto import tqdm


def train(weights_path: str,
          yaml_url: str,
          train_files: list,
          test_files: list,
          combine: int,
          initial_learning_rate: float = 0.00025,
          train_steps: int = 120_000,
          eval_steps: int = 50_000,
          checkpoints_freq: int = 40_000,
          log_freq: int = 5000,
          batch_size: int = 2,
          output_directory: str = 'output',
          delay: int = 6 * 3600):
    """
    Trains a model using the specified configurations.

    :param network_base_name: The name of the network base to use.
    :param weights_path: The path to the weights file.
    :param yaml_url: The URL to the YAML configuration file.
    :param combine: int, can be one of [0, 1, 2], 0 if each data file represents one split of one dataset,
                                                  1 if the user wants to combine test files (recommended),
                                                  2 if the user wants to combine train and test files (not
    :param train_files: list, list of json files containing train dataset catalogues to be registered.
    :param test_files: list, list of json files containing train dataset catalogues to be registered.
    :param initial_learning_rate: The initial learning rate to use for training. Default is 0.00025.
    :param train_steps: The number of training steps to perform. Default is 120_000.
    :param eval_steps: The number of evaluation steps to perform. Default is 50_000.
    :param checkpoints_freq: int, the frequency at which to make a model checkpoint. Default is 40_000
    :param log_freq: int, the frequency at which to log training details. Default is 5000
    :param batch_size: The batch size to use for training. Default is 2.
    :param output_directory: str, the directory to which training results will be saved
    :param delay: int, number of seconds to wait before starting execution (in case one wants to start training a model
    after one is already done training)
    """
    for _ in tqdm(range(delay), desc='Sleeping'):
        sleep(1.)
    if combine > 0:
        test_files = [merge(test_files, DATASET_INFO_FILE_TEST)]
    if combine > 1:
        train_files = [merge(train_files, DATASET_INFO_FILE_TRAIN)]
    # register the datasets
    train_datasets = [file.split('/')[-1].replace('.json', '') for file in train_files]
    test_datasets = [file.split('/')[-1].replace('.json', '') for file in test_files]

    [register_dataset(file, file.split('/')[-1].replace('.json', '')) for file in [*train_files, *test_files]]
    [visualize_sample(file) for file in [*train_files, *test_files]]

    # Get configurations from specified parameters
    configurations = get_cfg(network_base_name=yaml_url.split('/')[-1].replace('json', ''),
                             weights_path=weights_path,
                             yaml_url=yaml_url,
                             train_datasets=tuple(train_datasets),
                             test_datasets=tuple(test_datasets),
                             initial_learning_rate=initial_learning_rate,
                             train_steps=train_steps,
                             eval_freq=eval_steps,
                             checkpoints_freq=checkpoints_freq,
                             log_freq=log_freq,
                             batch_size=batch_size,
                             output_directory=output_directory)

    # Create trainer object with configurations
    trainer = Trainer(configurations)

    # Train model
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--yaml_url', type=str, required=True)
    parser.add_argument('--train_files', nargs='+', default=['data/raw/CelebA/celeba_train.json',
                                                             'data/raw/WIDER_FACE/wider_face_train.json',
                                                             'data/raw/CCPD2019/ccpd_train.json'])
    parser.add_argument('--test_files', nargs='+', default=['data/raw/CelebA/celeba_test.json',
                                                            'data/raw/WIDER_FACE/wider_face_test.json',
                                                            'data/raw/CCPD2019/ccpd_test.json'])
    parser.add_argument('--combine', type=int, default=1)
    parser.add_argument('--output_directory', type=str, default='output')
    parser.add_argument('--initial_learning_rate', type=float, default=0.00025)
    parser.add_argument('--train_steps', type=int, default=160_000)
    parser.add_argument('--eval_steps', type=int, default=50_000)
    parser.add_argument('--log_freq', type=int, default=500)
    parser.add_argument('--checkpoints_freq', type=int, default=40_000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--delay', type=int, default=6 * 3600)

    args = vars(parser.parse_args())

    train(**args)
