# BookCover Generation Project

**DALLE and VQGAN for improved text-to-image generation**

## Introduction
This project contains both training and inference codes for VQGAN and DALLE to generate images.
Unlike the original paper of DALLE and VQGAN, we use some techniques to improve the reconstruction quality and generation diversity.


## Subdirectories
* [clip](./clip): CLIP for the metadata text of the book and its cover image.
* [vqgan](./vqgan): Encode the images to discretized latent codes and decode to the original images.
* [dalle](./dalle): Sequence-to-sequence transformer model to translate from the metadata text to quantized image tokens.
* [inference](./inference): Optimize the VQGAN and DALLE models with merging them into single graph.
