import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset import RecursiveImageDataset
from lightning import VQGANTrainingModule


def apply_augmentation(images: torch.Tensor) -> torch.Tensor:
    augmentation = random.choice(["t", "c", "tc"])

    if "t" in augmentation:
        original_size = images.size(2)
        images = F.interpolate(images, (original_size + 12, original_size + 12))

        corner = random.choice(["tl", "tr", "bl", "br"])
        if corner == "tl":
            images = images[:, :, :original_size, :original_size]
        elif corner == "tr":
            images = images[:, :, :original_size, -original_size:]
        elif corner == "bl":
            images = images[:, :, -original_size:, :original_size]
        elif corner == "br":
            images = images[:, :, -original_size:, -original_size:]

    if "c" in augmentation:
        images = images * (0.6 + 0.8 * random.random())
        images = images.clamp(-1, 1)

    return images


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
        if args.augmentation:
            images = apply_augmentation(images)

        latent_ids = model.quantizer(model.encoder(images))[1]
        tokens_list.append(latent_ids.type(torch.uint16).flatten(1).cpu().numpy())

    # Save the index of images and the quantized sequences. Note that the sequences will
    # be stored to `npy` so that other programs can use this through `np.memmap`.
    with open(f"{args.output}.index", "w") as fp:
        indices = [os.path.splitext(os.path.basename(x))[0] for x in dataset.filenames]
        fp.write("\n".join(indices))
    np.save(f"{args.output}.npy", np.concatenate(tokens_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--input", default="../../resources/kyobobook-images/**/*.jpg")
    parser.add_argument("--output", default="kyobobook-quantized")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-fp16", action="store_true", default=False)
    parser.add_argument("--augmentation", action="store_true", default=False)
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(args, config)
