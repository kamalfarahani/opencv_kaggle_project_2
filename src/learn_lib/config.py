from dataclasses import dataclass


@dataclass
class TrainingConfiguration:
    """
    Describes configuration of the training process
    """

    batch_size: int = 32
    epochs_count: int = 50
    init_learning_rate: float = 0.001  # initial learning rate for lr scheduler
    log_interval: int = 5
    test_interval: int = 1
    data_root: str = "./data"
    num_workers: int = 4
    device: str = "cuda"
    early_stopping_patience: int = 5
