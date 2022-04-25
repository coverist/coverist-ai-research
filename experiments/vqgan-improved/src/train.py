import argparse
import warnings
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from dataset import create_train_val_dataloaders
from lightning import VQGANTrainingModule

try:
    import apex

    amp_backend = apex.__name__
except ModuleNotFoundError:
    amp_backend = "native"

warnings.filterwarnings("ignore")


def main(
    config: DictConfig,
    resume_from: Optional[str] = None,
    resume_id: Optional[str] = None,
):
    trainer = Trainer(
        gpus=config.train.gpus,
        logger=WandbLogger(
            project="bookcover-generation-vqgan", name=config.train.name, id=resume_id
        ),
        callbacks=[ModelCheckpoint(save_last=True), LearningRateMonitor("epoch")],
        precision=config.train.precision,
        max_epochs=config.train.epochs,
        amp_backend=amp_backend,
        check_val_every_n_epoch=config.train.validation_interval,
        accumulate_grad_batches=config.train.accumulate_grads,
        log_every_n_steps=50,
    )
    trainer.fit(
        VQGANTrainingModule(config),
        *create_train_val_dataloaders(config),
        ckpt_path=resume_from,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume_from")
    parser.add_argument("--resume_id")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(config, args.resume_from, args.resume_id)
