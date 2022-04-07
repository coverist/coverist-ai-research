import argparse
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightning import VQVAEDataModule, VQVAETrainingModule


def main(
    config: DictConfig,
    resume_from: Optional[str] = None,
    resume_id: Optional[str] = None,
):
    checkpoint = ModelCheckpoint(save_last=True)
    Trainer(
        gpus=config.train.gpus,
        logger=WandbLogger(
            project="bookcover-generation", name=config.train.name, id=resume_id
        ),
        callbacks=[checkpoint],
        precision=config.train.precision,
        max_epochs=config.train.epochs,
        check_val_every_n_epoch=config.train.validation_interval,
        accumulate_grad_batches=config.train.accumulate_grads,
        log_every_n_steps=50,
    ).fit(VQVAETrainingModule(config), VQVAEDataModule(config), ckpt_path=resume_from)

    module = VQVAETrainingModule.load_from_checkpoint(
        checkpoint.last_model_path, config=config
    )
    torch.save(module.state_dict(), f"{config.train.name}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume_from")
    parser.add_argument("--resume_id")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(config, args.resume_from, args.resume_id)
