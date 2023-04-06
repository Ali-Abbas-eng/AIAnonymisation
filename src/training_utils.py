import os
import os.path
from checkpoints_downloader import get_info
from detectron2.engine.defaults import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg as base_configurations
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.engine import hooks


class EarlyStoppingHook(hooks.HookBase):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def after_step(self):
        if self.trainer.iter == 0:
            return
        if self.best_score is None:
            self.best_score = self.trainer.storage.history("total_loss").latest()
        else:
            current_score = self.trainer.storage.history("total_loss").latest()
            if current_score > self.best_score:
                self.counter += 1
                if self.counter >= self.patience:
                    raise Exception('early stopping')
            else:
                self.best_score = current_score
                self.counter = 0


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


def get_cfg(network_base_name: str,
            weights_path: str,
            yaml_url: str,
            train_datasets: tuple,
            test_datasets: tuple,
            initial_learning_rate: float = 0.00025,
            train_steps: int = 5000,
            eval_freq: int = 5000,
            checkpoints_freq: int = 5000,
            log_freq: int = 5000,
            batch_size: int = 2,
            output_directory: str = 'output'):
    cfg = base_configurations()
    cfg.merge_from_file(get_config_file(yaml_url))
    cfg.MODEL.WEIGHTS = weights_path if weights_path is not None else get_checkpoint_url(yaml_url)
    cfg.OUTPUT_DIR = os.path.join(output_directory, network_base_name)
    cfg.DATASETS.TRAIN = train_datasets
    cfg.DATASETS.TEST = test_datasets
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoints_freq
    cfg.SOLVER.LOGGER_PERIOD = log_freq
    cfg.SOLVER.MAX_ITER = train_steps
    cfg.TEST.EVAL_PERIOD = eval_freq
    cfg.SOLVER.BASE_LR = initial_learning_rate
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg
