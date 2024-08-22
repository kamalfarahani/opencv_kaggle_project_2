import lightning as L

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from learn_lib.config import TrainingConfiguration


def train(
    model: L.LightningModule,
    data_module: L.LightningDataModule,
    training_configuration: TrainingConfiguration,
) -> None:
    early_stopping = EarlyStopping(
        monitor="val/acc",
        mode="max",
        verbose=True,
        patience=training_configuration.early_stopping_patience,
    )

    model_checkpoint = ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        dirpath="checkpoints/",
        filename="model-{epoch:02d}",
        auto_insert_metric_name=False,
        verbose=True,
    )

    trainer = L.Trainer(
        max_epochs=training_configuration.epochs_count,
        log_every_n_steps=training_configuration.log_interval,
        callbacks=[
            early_stopping,
            model_checkpoint,
        ],
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
    )
