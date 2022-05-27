import argparse
import warnings
from typing import Optional

from dataset import create_train_val_dataloaders
from lightning import VQGANTrainingModule
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

warnings.filterwarnings("ignore")


def main(
    config: DictConfig,
    resume_from: Optional[str] = None,
    resume_id: Optional[str] = None,
):
    trainer = Trainer(
        gpus=1,
        precision=16,
        amp_backend="apex",
        log_every_n_steps=config.train.log_every_n_steps,
        max_epochs=config.train.epochs,
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        val_check_interval=min(config.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(config.train.validation_interval), 1),
        callbacks=[ModelCheckpoint(save_last=True), LearningRateMonitor("step")],
        logger=WandbLogger(
            project="bookcover-generation-vit-vqgan-decoder",
            name=config.train.name,
            id=resume_id,
        ),
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
