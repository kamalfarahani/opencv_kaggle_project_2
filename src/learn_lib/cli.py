import pandas as pd

from learn_lib.model import KenyanFood13Classifier
from learn_lib.data_loader import (
    KenyanFood13Dataset,
    KenyanFood13DataModule,
    kaggle_test_images,
)
from learn_lib.config import TrainingConfiguration
from learn_lib.train import train


def main(data_root: str) -> None:
    """Trains the model

    Args:
        data_root (str): path to the data root
    """
    mode = input("Train model? (y/n): ")
    if mode == "y":
        train_model(data_root)
    else:
        checkpoint = input("Enter checkpoint path: ")

        dataset = KenyanFood13Dataset(data_root)
        class_to_idx = dataset.class_to_idx

        model = KenyanFood13Classifier.load_from_checkpoint(checkpoint)
        model.freeze()
        model.eval()

        infer_test_images(
            model,
            data_root,
            class_to_idx,
        )


def train_model(
    data_root: str,
) -> None:
    model = KenyanFood13Classifier()
    train_config = TrainingConfiguration()
    train_config.early_stopping_patience = 10
    datamodule = KenyanFood13DataModule(
        data_root=data_root,
        use_augmentation=True,
    )

    train(
        model=model,
        training_configuration=train_config,
        data_module=datamodule,
    )


def infer_test_images(
    model: KenyanFood13Classifier,
    data_root: str,
    class_to_idx: dict[str, int] = None,
) -> None:
    device = next(model.parameters()).device
    kaggle_test_images_gen = kaggle_test_images(data_root)
    index_to_class = {v: k for k, v in class_to_idx.items()}

    result_dict = {
        "id": [],
        "class": [],
    }

    for image, image_name in kaggle_test_images_gen:
        image = image.to(device)
        output = model(image.unsqueeze(0))
        pred = output.argmax(dim=1)
        pred_class = index_to_class[pred.item()]

        print(f"Image name: {image_name}, Predicted class: {pred_class}")

        result_dict["id"].append(image_name)
        result_dict["class"].append(pred_class)

    result_df = pd.DataFrame(result_dict)

    result_df.to_csv(
        "submission.csv",
        index=False,
    )
