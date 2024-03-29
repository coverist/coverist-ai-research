import argparse
import os
import warnings
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from lightning import DALLETrainingDataModule, DALLETrainingModule

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
    trainer = Trainer(
        gpus=config.train.gpus,
        logger=WandbLogger(
            project="bookcover-generation-dalle", name=config.train.name, id=resume_id
        ),
        callbacks=[ModelCheckpoint(save_last=True), LearningRateMonitor("step")],
        precision=config.train.precision,
        max_steps=config.optim.scheduler.num_training_steps,
        amp_backend=amp_backend,
        gradient_clip_val=config.train.max_grad_norm,
        val_check_interval=min(config.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(config.train.validation_interval), 1),
        accumulate_grad_batches=config.train.accumulate_grads,
        log_every_n_steps=config.train.logging_interval,
    )
    trainer.fit(
        DALLETrainingModule(config),
        DALLETrainingDataModule(config),
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
