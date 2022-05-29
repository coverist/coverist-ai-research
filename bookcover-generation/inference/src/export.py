import argparse
import logging
from typing import Any, Union

import torch
import torch.nn as nn
from transformers import AutoTokenizer, EncoderDecoderModel, PreTrainedTokenizerBase

from modeling import VQGANDecoder
from wrappers import (
    DALLE,
    DALLEDecoderWrapper,
    DALLEEncoderWrapper,
    VQGANDecoderWrapper,
)


def move_tensors_to(tensors: Any, device: Union[str, torch.device] = "cpu") -> Any:
    if isinstance(tensors, tuple):
        return tuple(move_tensors_to(x, device) for x in tensors)
    elif isinstance(tensors, list):
        return [move_tensors_to(x, device) for x in tensors]
    elif isinstance(tensors, torch.Tensor):
        return tensors.to(device)


def compile_with_trace(
    module: nn.Module, use_gpu: bool = False
) -> torch.jit.ScriptModule:
    example_inputs = module.generate_example_inputs()
    example_inputs = move_tensors_to(example_inputs, "cuda" if use_gpu else "cpu")
    return torch.jit.trace(module, example_inputs)


def test_model_generation(
    dalle: Union[DALLE, torch.jit.ScriptModule],
    tokenizer: PreTrainedTokenizerBase,
    use_gpu: bool = False,
):
    prompt = f" {tokenizer.sep_token} ".join(["제목", "저자", "출판사"])
    encodings = tokenizer(prompt, return_tensors="pt").to("cuda" if use_gpu else "cpu")

    images = dalle(
        encodings.input_ids,
        encodings.attention_mask,
        num_return_sequences=4,
    )
    assert images.ndim == 4
    assert (images <= 1.0).all()
    assert (images >= -1.0).all()


@torch.no_grad()
def main(args: argparse.ArgumentParser):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    logging.debug("Load dalle-encoder-decoder and vqgan-decoder...")
    tokenizer = AutoTokenizer.from_pretrained(args.dalle_encoder_decoder)
    dalle_encoder_decoder = EncoderDecoderModel.from_pretrained(
        args.dalle_encoder_decoder
    )
    vqgan_decoder = VQGANDecoder.from_pretrained(args.vqgan_decoder)

    logging.debug("Disable gradients and training-specific layers (e.g. dropout)")
    dalle_encoder_decoder.requires_grad_(False).eval()
    vqgan_decoder.requires_grad_(False).eval()

    if args.use_gpu:
        logging.debug("Copy the weights to CUDA memory")
        dalle_encoder_decoder.cuda()
        vqgan_decoder.cuda()

    logging.debug("Wrapping the models")
    dalle_encoder = DALLEEncoderWrapper(
        dalle_encoder_decoder.encoder,
        getattr(dalle_encoder_decoder, "enc_to_dec_proj"),
    )
    dalle_decoder = DALLEDecoderWrapper(dalle_encoder_decoder.decoder)
    vqgan_decoder = VQGANDecoderWrapper(vqgan_decoder, args.sequence_length)

    if args.use_torchscript:
        logging.debug("Compile to torchscript JIT through tracing...")
        dalle_encoder = compile_with_trace(dalle_encoder, args.use_gpu)
        dalle_decoder = compile_with_trace(dalle_decoder, args.use_gpu)
        vqgan_decoder = compile_with_trace(vqgan_decoder, args.use_gpu)

    logging.debug("Create a new dalle model")
    dalle = DALLE(
        dalle_encoder,
        dalle_decoder,
        vqgan_decoder,
        args.sequence_length,
        dalle_encoder_decoder.config.decoder_start_token_id,
        dalle_encoder_decoder.decoder.config.num_attention_heads,
        dalle_encoder_decoder.decoder.config.hidden_size
        // dalle_encoder_decoder.decoder.config.num_attention_heads,
        dalle_encoder_decoder.decoder.config.num_hidden_layers,
    )
    if args.use_torchscript:
        logging.debug("Compile the dalle to torchscript JIT...")
        dalle = torch.jit.script(dalle)
        dalle = torch.jit.optimize_for_inference(dalle)

    logging.debug("Finish exporting dalle model!")
    logging.debug("Start evaluating and check if the model performs correctly")
    test_model_generation(dalle, tokenizer, args.use_gpu)

    logging.debug("Save the dalle model")
    if args.use_torchscript:
        dalle.save(args.output)
    else:
        torch.save(dalle, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dalle-encoder-decoder", default="dalle-16l-1024d")
    parser.add_argument("--vqgan-decoder", default="vqgan-f16-16384-decoder.pth")
    parser.add_argument("--sequence-length", type=int, default=576)
    parser.add_argument("--use-torchscript", action="store_true", default=False)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--output", default="dalle.pth")
    parser.add_argument("--verbose", action="store_true", default=False)
    main(parser.parse_args())
