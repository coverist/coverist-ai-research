import argparse
import os

import pandas as pd
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset import RecursiveImageDataset
from lightning import VQGANTrainingModule


def encode_to_alphabet_word(number: int) -> str:
    word = ""
    while number:
        word += chr(number % 26 + 65)
        number //= 26
    return word


def decode_alphabet_word(word: str) -> int:
    number = 0
    for i, letter in enumerate(word):
        number += (ord(letter) - 65) * 26 ** i
    return number


@torch.no_grad()
def main(args: argparse.Namespace, config: DictConfig):
    # Load the model weights from the checkpoint and move to the GPU memory and cast to
    # `float16` if `use_fp16` is enabled.
    model = VQGANTrainingModule.load_from_checkpoint(args.checkpoint, config=config)
    model.cuda().eval().type(torch.float16 if args.use_fp16 else torch.float32)

    # Create dataset and dataloader to serve batched images.
    dataset = RecursiveImageDataset(args.input, args.image_size)
    dataloader = DataLoader(
        dataset, args.batch_size, num_workers=os.cpu_count(), pin_memory=True
    )

    # Encode the images and get closest quantized embedding vectors. After that, the
    # indices of quantized vectors would be stored to the output csv file.
    tokens_list = []
    for images in tqdm.tqdm(dataloader):
        images = images.cuda().type(torch.float16 if args.use_fp16 else torch.float32)
        _, latent_ids, *_ = model.quantizer(model.encoder(images))

        latent_ids = latent_ids.flatten(1).tolist()
        latent_ids = [" ".join(map(encode_to_alphabet_word, i)) for i in latent_ids]
        tokens_list.extend(latent_ids)

    results = [
        {"id": os.path.splitext(os.path.basename(filename))[0], "tokens": tokens}
        for filename, tokens in zip(dataset.filenames, tokens_list)
    ]
    pd.DataFrame(results).to_csv(args.output, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--input", default="../../resources/kyobobook-images/**/*.jpg")
    parser.add_argument("--output", default="kyobobook-quantized.csv")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--checkpoint", default="last.ckpt")
    parser.add_argument("--use-fp16", action="store_true", default=False)
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(args, config)
