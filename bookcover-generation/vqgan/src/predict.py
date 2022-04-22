import argparse
import os

import pandas as pd
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset import RecursiveImageDataset
from lightning import VQGANTrainingModule


@torch.no_grad()
def main(args: argparse.Namespace, config: DictConfig):
    model = VQGANTrainingModule.load_from_checkpoint(args.checkpoint, config=config)
    model.cuda().eval()

    dataset = RecursiveImageDataset(args.input, args.image_size)
    dataloader = DataLoader(
        dataset, args.batch_size, num_workers=os.cpu_count(), pin_memory=True
    )

    tokens_list = []
    for _, images in zip(range(100), tqdm.tqdm(dataloader)):
        _, quantized_ids, *_ = model.quantizer(model.encoder(images.cuda()))
        quantized_ids = quantized_ids.flatten(1).tolist()
        quantized_ids = [" ".join(map(str, ids)) for ids in quantized_ids]
        tokens_list.extend(quantized_ids)

    results = [
        {"id": os.path.splitext(os.path.basename(filename))[0], "tokens": tokens}
        for filename, tokens in zip(dataset.filenames, tokens_list)
    ]
    pd.DataFrame(results).to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--input", default="../../resources/kyobobook-images/**/*.jpg")
    parser.add_argument("--output", default="encoded.csv")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint", default="last.ckpt")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(args, config)
