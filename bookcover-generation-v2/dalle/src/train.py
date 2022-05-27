import argparse
import os
import warnings
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightning import DALLETrainingDataModule, DALLETrainingModule

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(
    config: DictConfig,
    resume_from: Optional[str] = None,
    resume_id: Optional[str] = None,
):
    checkpoint = ModelCheckpoint(save_last=True)
    Trainer(
        gpus=1,
        precision=16,
        amp_backend="apex",
        log_every_n_steps=config.train.log_every_n_steps,
        max_steps=config.optim.scheduler.num_training_steps,
        gradient_clip_val=config.train.gradient_clip_val,
        accumulate_grad_batches=config.train.accumulate_grad_batches,
        val_check_interval=min(config.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(config.train.validation_interval), 1),
        callbacks=[checkpoint],
        logger=WandbLogger(
            project="bookcover-generation-v2-dalle",
            name=config.train.name,
            id=resume_id,
        ),
    ).fit(
        DALLETrainingModule(config),
        DALLETrainingDataModule(config),
        ckpt_path=resume_from,
    )

    # Save the weights of the trained model and its tokenizer through `save_pretrained`.
    model = DALLETrainingModule.load_from_checkpoint(
        checkpoint.last_model_path, config=config
    )
    model.model.save_pretrained(config.train.name)
    model.tokenizer.save_pretrained(config.train.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume_from")
    parser.add_argument("--resume_id")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(config, args.resume_from, args.resume_id)
