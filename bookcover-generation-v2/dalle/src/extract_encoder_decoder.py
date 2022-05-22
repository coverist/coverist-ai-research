import argparse

import torch
from omegaconf import DictConfig, OmegaConf

from lightning import DALLETrainingModule


@torch.no_grad()
def main(args: argparse.Namespace, config: DictConfig):
    model = DALLETrainingModule.load_from_checkpoint(args.checkpoint, config=config)
    model.cpu().eval()

    model.model.save_pretrained(args.output)
    model.tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="dalle-encoder-decoder")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(args, config)
