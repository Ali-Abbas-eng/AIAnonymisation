import json
import os
import os.path
from checkpoints_downloader import get_info
from detectron2.engine.defaults import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg as base_configurations
from detectron2.model_zoo import get_config_file
from detectron2.engine.hooks import HookBase


class TrainingSessionManagementHook(HookBase):
    """
    A custom hook for early stopping training in Detectron2.

    This hook monitors the validation loss during training and reduces the learning rate or stops training if the
    validation loss does not improve for a specified number of steps (max_patience).

    :param max_patience: The maximum number of steps to wait for an improvement in validation
    loss before reducing the learning rate or stopping training.
    :type max_patience: int
    """

    def __init__(self, max_patience, lr_factor):
        # Initialize instance variables
        self.max_patience = max_patience  # maximum patience before reducing learning rate or stopping training
        self.patience = 0  # current patience counter
        self.best_val_loss = float('inf')  # best validation loss seen so far

    def after_step(self):
        """
        Called after each training step. Monitors validation loss and reduces learning rate or stops training if
        necessary.
        """

        # Get current validation loss from trainer storage
        cur_val_loss = self.trainer.storage.latest()['validation_loss']

        # Check if validation loss has improved compared to best_val_loss
        if cur_val_loss < self.best_val_loss:
            # If validation loss has improved, update best_val_loss and reset patience counter
            self.best_val_loss = cur_val_loss
            self.patience = 0
        else:
            # If validation loss has not improved, increment patience counter
            self.patience += 1

            # Check if patience has exceeded max_patience
        if self.patience >= self.max_patience:
            # If learning rate has already been reduced once, stop training now by calling trainer.terminate()
            print(f"Early stopping at iteration {self.trainer.iter}")
            self.trainer.terminate()



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


def convert_to_detectron2_usable_info():
    """
    A function that converts a list of variations of a model to a format usable by Detectron2.

    Returns:
        records (list): A list of dictionaries containing information about the model variations in a format usable by
        Detectron2.
    """

    # Get information about the model variations
    info = get_info()

    # Create a list to store the converted information
    records = []

    # Loop through the information and convert it to a format usable by Detectron2
    for pair in info:
        variations_info = pair[0]
        base_dir = pair[1]
        for key, value in variations_info.items():
            record = {}
            path = os.path.join(base_dir, value.split(os.path.sep)[-1])
            record['name'] = key
            record['yaml'] = f'COCO-Detection/{key}.yaml'
            record['path'] = path
            records.append(record)

    return records


def create_dataset_dicts(split: str, data_directory: str = 'data'):
    """
    a functionality to merge json files that contain portions of the total dataset (each downloaded individually)
    Args:
        split: str, the set of the data which separate files are to be merged, could be ['train', 'test', 'val']
        data_directory: str, the directory to flip around looking for the data to use

    Returns: list[dict]
        a list of dictionaries containing information about the dataset as a whole

    """
    total_data = os.path.join(data_directory, f'{split}.json')
    if os.path.isfile(total_data):
        data = json.load(open(total_data, 'r'))
        return data

    else:
        # initialise an empty list to hold the data objects
        data = []

        # recursively iterate through the data directory and look for the json files
        for root, dirs, files in os.walk(data_directory):
            # iterate through all the files contained the current directory
            for file in files:
                # is the current file is a json file
                if file.endswith('.json'):
                    # if the file name contains the name of the split of interest
                    if split in file:
                        # load the content of the file (which is a list) and extend the data list we initialised earlier
                        data.extend(json.load(open(os.path.join(root, file))))
        json.dump(data, open(total_data, 'w'))
        # return the list of dictionaries that represents the datasets
        return data


def register_datasets(data_directory: str,
                      thing_classes: list):
    """
    Registers datasets for training, testing, and validation in Detectron2.

    This function registers datasets for training, testing, and validation in Detectron2 using the provided data
    directory and thing classes. The data directory should contain the data for all three splits (train, test, valid)
    in separate subdirectories. The thing classes should be a list of class names corresponding to the classes present
    in the dataset.

    :param data_directory: The path to the directory containing the data for all three splits (train, test, valid).
    :type data_directory: str
    :param thing_classes: A list of class names corresponding to the classes present in the dataset.
    :type thing_classes: list
    """

    # Define a function that returns a dataset dictionary for the current split using create_dataset_dicts()
    def data_getter(data_split: str):
        return create_dataset_dicts(split=data_split, data_directory=data_directory)

    # Loop over all three splits (train, test, valid)
    for split in ['train', 'test', 'valid']:
        # Register the current split with Detectron2 using DatasetCatalog.register()
        DatasetCatalog.register(split, lambda data_split=split: data_getter(data_split=data_split))

        # Set thing_classes for the current split using MetadataCatalog.get().set()
        MetadataCatalog.get(split).set(thing_classes=thing_classes)


def get_cfg(network_base_name: str,
            weights_path: str,
            yaml_url: str,
            initial_learning_rate: float = 0.00025,
            train_steps: int = 5000,
            eval_freq: int = 5000,
            batch_size: int = 2):
    """
    This function generates a configuration object for training a neural network.

    :param network_base_name: The base name of the network.
    :param weights_path: The path to the pre-trained weights file.
    :param yaml_url: The URL to the YAML configuration file.
    :param initial_learning_rate: The initial learning rate for training. (default=0.00025)
    :param train_steps: The number of steps to train for. (default=5000)
    :param eval_freq: The frequency of evaluation w.r.t training steps. (default=5000)
    :param batch_size: The batch size to use during training. (default=2)

    :return configurations:
        A configuration object with all specified parameters set.
    """

    # Get base configurations
    configurations = base_configurations()

    # Merge configurations from YAML file
    configurations.merge_from_file(get_config_file(yaml_url))

    # Set pre-trained weights
    configurations.MODEL.WEIGHTS = weights_path

    # Set output directory
    configurations.OUTPUT_DIR = os.path.join('output', network_base_name)

    # Set training dataset
    configurations.DATASETS.TRAIN = ('train',)

    # Set test dataset
    configurations.DATASETS.TEST = ('test',)

    # Set batch size
    configurations.SOLVER.IMS_PER_BATCH = batch_size

    # Set maximum number of iterations
    configurations.SOLVER.MAX_ITER = train_steps

    # Set the frequency of evaluation
    configurations.TEST.EVAL_PERIOD = eval_freq

    # Set initial learning rate
    configurations.SOLVER.BASE_LR = initial_learning_rate

    return configurations
