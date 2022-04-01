import argparse
import os
import warnings
from typing import Optional

import torch
from lightning import BigGANDataModule, BigGANTrainingModule
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

try:
    import apex

    amp_backend = apex.__name__
except ModuleNotFoundError:
    amp_backend = "native"

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    config: DictConfig,
    resume_from: Optional[str] = None,
    resume_id: Optional[str] = None,
):
    module = BigGANTrainingModule(config)
    datamodule = BigGANDataModule(config)
    checkpoint = ModelCheckpoint(monitor="step", mode="max", save_top_k=3)

    Trainer(
        gpus=config.train.gpus,
        logger=WandbLogger(
            project="bookcover-generation", name=config.train.name, id=resume_id
        ),
        callbacks=[checkpoint],
        precision=config.train.precision,
        max_steps=config.train.steps,
        amp_backend=amp_backend,
        check_val_every_n_epoch=config.train.validation_interval,
        accumulate_grad_batches=config.train.accumulate_grads,
        log_every_n_steps=10,
    ).fit(module, datamodule, ckpt_path=resume_from)

    module = BigGANTrainingModule.load_from_checkpoint(
        checkpoint.best_model_path, config=config
    )
    state_dict = {
        "state_dict": module.generator_ema.state_dict(),
        "labels": datamodule.labels,
    }
    torch.save(state_dict, f"{config.train.name}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume_from")
    parser.add_argument("--resume_id")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(config, args.resume_from, args.resume_id)
