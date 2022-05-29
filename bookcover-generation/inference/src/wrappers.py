from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from modeling import VQGANDecoder


class DALLEEncoderWrapper(nn.Module):
    def __init__(self, encoder: PreTrainedModel, linear: Optional[nn.Linear] = None):
        super().__init__()
        self.encoder = encoder
        self.linear = linear or nn.Identity()

    def generate_example_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        logits, *_ = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        return self.linear(logits)


class DALLEDecoderWrapper(nn.Module):
    def __init__(self, decoder: PreTrainedModel):
        super().__init__()
        self.decoder = decoder

    def generate_example_inputs(
        self,
    ) -> tuple[
        torch.Tensor,
        list[tuple[torch.Tensor, torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
    ]:
        num_attention_heads = self.decoder.config.num_attention_heads
        attention_head_size = self.decoder.config.hidden_size // num_attention_heads

        past_key_values = []
        for _ in range(self.decoder.config.num_hidden_layers):
            key = torch.zeros((1, num_attention_heads, 0, attention_head_size))
            value = torch.zeros((1, num_attention_heads, 0, attention_head_size))
            past_key_values.append((key, value))

        return (
            torch.zeros((1, 1), dtype=torch.long),
            past_key_values,
            torch.zeros((1, 1, self.decoder.config.hidden_size)),
            torch.zeros((1, 1)),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        return self.decoder(
            input_ids=input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )


class VQGANDecoderWrapper(nn.Module):
    def __init__(self, decoder: VQGANDecoder, sequence_length: int):
        super().__init__()
        self.decoder = decoder
        self.sequence_length = sequence_length
        self.num_rows = int(sequence_length ** 0.5)

    def generate_example_inputs(self) -> torch.Tensor:
        return torch.zeros((1, self.sequence_length), dtype=torch.long)

    def forward(self, latent_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_ids.view(-1, self.num_rows, self.num_rows))


class DALLE(nn.Module):
    def __init__(
        self,
        dalle_encoder: Union[DALLEEncoderWrapper, torch.jit.ScriptModule],
        dalle_decoder: Union[DALLEEncoderWrapper, torch.jit.ScriptModule],
        vqgan_decoder: Union[DALLEEncoderWrapper, torch.jit.ScriptModule],
        sequence_length: int,
        start_token_id: int,
        kv_num_attention_heads: int,
        kv_attention_head_size: int,
        kv_num_hidden_layers: int,
    ):
        super().__init__()
        self.dalle_encoder = dalle_encoder
        self.dalle_decoder = dalle_decoder
        self.vqgan_decoder = vqgan_decoder

        self.sequence_length = sequence_length
        self.start_token_id = start_token_id

        self.kv_num_attention_heads = kv_num_attention_heads
        self.kv_attention_head_size = kv_attention_head_size
        self.kv_num_hidden_layers = kv_num_hidden_layers

    def prepare_generation_inputs(
        self, input_ids: torch.Tensor, num_return_sequences: int
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        batch_size = input_ids.size(0) * num_return_sequences
        num_attention_heads = self.kv_num_attention_heads
        attention_head_size = self.kv_attention_head_size
        kv_shape = (batch_size, num_attention_heads, 0, attention_head_size)

        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.kv_num_hidden_layers):
            key = input_ids.new_zeros(kv_shape)
            value = input_ids.new_zeros(kv_shape)
            past_key_values.append((key, value))

        input_ids = torch.tensor(
            [[self.start_token_id]] * batch_size,
            device=input_ids.device,
            dtype=torch.int64,
        )
        return input_ids, past_key_values

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 50,
        num_return_sequences: int = 1,
    ) -> torch.Tensor:
        hidden_states = self.dalle_encoder(input_ids, attention_mask)
        hidden_states = hidden_states.repeat_interleave(num_return_sequences, dim=0)

        input_ids, past_key_values = self.prepare_generation_inputs(
            input_ids, num_return_sequences
        )
        generated_tokens = []

        for _ in range(self.sequence_length):
            logits, past_key_values = self.dalle_decoder(
                input_ids,
                past_key_values,
                hidden_states,
                attention_mask,
            )
            logits_values, logits_indices = logits.squeeze(1).topk(top_k, dim=1)

            input_ids = (logits_values / temperature).softmax(dim=1).multinomial(1)
            input_ids = logits_indices.gather(dim=1, index=input_ids)
            generated_tokens.append(input_ids)

        latent_ids = torch.cat(generated_tokens, dim=1)
        return self.vqgan_decoder(latent_ids)
