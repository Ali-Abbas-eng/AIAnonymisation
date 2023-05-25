import os
import os.path
import torch
from detectron2.engine.defaults import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg as base_configurations
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping


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


def get_cfg(network_base_name: str,
            yaml_url: str,
            train_datasets: tuple,
            test_datasets: tuple,
            output_directory: str,
            min_learning_rate: float = 1e-7,
            initial_learning_rate: float = 1e-5,
            train_steps: int = 100_000,
            eval_freq: int = 20_000,
            batch_size: int = 2,
            decay_freq: int = 1000,
            decay_gamma: float = .9,
            roi_heads: int = 256):
    """
    Generates a configuration object for a network.

    :param network_base_name: str, The base name of the network.
    :param yaml_url: str, The URL of the YAML configuration file.
    :param train_datasets: tuple, A tuple of training datasets.
    :param test_datasets: tuple, A tuple of testing datasets.
    :param min_learning_rate: float, the minimum value of the learning rate at which we stop learning rate decay.
    :param initial_learning_rate: float, The initial learning rate. Defaults to 0.00025.
    :param train_steps: int, The number of training steps. Defaults to 5000.
    :param eval_freq: int, The evaluation frequency. Defaults to 5000.
    :param batch_size: int, The batch size. Defaults to 2.
    :param output_directory: str, the directory to which training results will be saved.
    :param decay_freq: int, the interval of the learning rate decay.
    :param decay_gamma: float, decay step for the learning rate scheduler
    :param output_directory: str, The output directory. Defaults to 'output'.

    Returns:
        cfg: A configuration object for the network.
    """
    # Get the base configurations
    cfg = base_configurations()

    # Merge the configurations from the YAML file
    cfg.merge_from_file(get_config_file(yaml_url))

    # Set the output directory
    cfg.OUTPUT_DIR = os.path.join(output_directory, network_base_name)

    # Set the weights path
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
    if not os.path.exists(cfg.MODEL.WEIGHTS):
        cfg.MODEL.WEIGHTS = get_checkpoint_url(yaml_url)

    # Set the training and testing datasets
    cfg.DATASETS.TRAIN = train_datasets
    cfg.DATASETS.TEST = test_datasets

    # Set the batch size
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    # Set the checkpoint and logging frequencies
    cfg.SOLVER.CHECKPOINT_PERIOD = eval_freq
    cfg.SOLVER.LOGGER_PERIOD = eval_freq

    # Set the maximum number of training steps
    cfg.SOLVER.MAX_ITER = train_steps

    # Set the evaluation frequency
    cfg.TEST.EVAL_PERIOD = eval_freq

    # Set the initial learning rate
    cfg.SOLVER.BASE_LR = initial_learning_rate

    # Set the number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    # Set the number of Regions of Interest
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_heads

    # Create the output directory if it doesn't exist
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # set learning rate decay options
    cfg.SOLVER.GAMMA = decay_gamma

    # Calculate the number of times learning rate decay will be applied
    decay_steps = train_steps // decay_freq
    # Initialise a list which will hold the step indexes at which decay will happen
    solver_steps = []

    # For each time the learning rate will be decayed
    for i in range(decay_steps):
        # In case the decay won't decrease the learning rate to a lower value than the minimum acceptable value
        if initial_learning_rate * decay_gamma ** i > min_learning_rate:
            # Add the step index to the list of decay steps
            solver_steps.append(decay_freq * (i + 1))

    # Assign the calculated decay steps to the proper configuration node attribute
    cfg.SOLVER.STEPS = tuple(solver_steps)

    # Return the final configuration node
    return cfg
