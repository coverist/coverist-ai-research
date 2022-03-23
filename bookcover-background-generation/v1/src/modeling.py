from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn as nn


@dataclass
class BigGANSelfAttentionConfig:
    hidden_size: int
    query_key_size: int
    value_size: int
    key_value_pooling: int


@dataclass
class BigGANGeneratorLayerConfig:
    input_size: int
    middle_size: int
    output_size: int
    conditional_size: int
    upsampling_factor: int


@dataclass
class BigGANDiscriminatorLayerConfig:
    input_size: int
    middle_size: int
    output_size: int
    downsampling_factor: int


@dataclass
class BigGANGeneratorConfig:
    num_classes: int
    noise_size: int
    embedding_size: int
    hidden_size_list: list[int]
    bottleneck_reduction: int = 4
    first_hidden_resolution: int = 4
    num_layers_per_block: int = 2
    attention_layer_position: int = 7
    attention_query_key_reduction: int = 8
    attention_value_reduction: int = 2
    attention_key_value_pooling: int = 2
    gradient_checkpointing: bool = False

    @property
    def first_hidden_size(self) -> int:
        return self.hidden_size_list[0]

    @property
    def last_hidden_size(self) -> int:
        return self.hidden_size_list[-1]

    @property
    def attention_config(self) -> BigGANSelfAttentionConfig:
        hidden_size = self.hidden_size_list[self.attention_layer_position]
        return BigGANSelfAttentionConfig(
            hidden_size=hidden_size,
            query_key_size=hidden_size // self.attention_query_key_reduction,
            value_size=hidden_size // self.attention_value_reduction,
            key_value_pooling=self.attention_key_value_pooling,
        )

    def __getitem__(self, idx: int) -> BigGANGeneratorLayerConfig:
        input_size = self.hidden_size_list[0 if idx == 0 else idx - 1]
        return BigGANGeneratorLayerConfig(
            input_size=input_size,
            middle_size=input_size // self.bottleneck_reduction,
            output_size=self.hidden_size_list[idx],
            conditional_size=self.noise_size + self.embedding_size,
            upsampling_factor=2 if (idx + 1) % self.num_layers_per_block == 0 else 1,
        )

    def __iter__(self) -> Iterator[BigGANGeneratorLayerConfig]:
        yield from [self[i] for i, _ in enumerate(self.hidden_size_list)]


@dataclass
class BigGANDiscriminatorConfig:
    num_classes: int
    hidden_size_list: list[int]
    bottleneck_reduction: int = 4
    num_layers_per_block: int = 2
    attention_layer_position: int = 5
    attention_query_key_reduction: int = 8
    attention_value_reduction: int = 2
    attention_key_value_pooling: int = 2
    gradient_checkpointing: bool = False

    @property
    def first_hidden_size(self) -> int:
        return self.hidden_size_list[0]

    @property
    def last_hidden_size(self) -> int:
        return self.hidden_size_list[-1]

    @property
    def attention_config(self) -> BigGANSelfAttentionConfig:
        hidden_size = self.hidden_size_list[self.attention_layer_position]
        return BigGANSelfAttentionConfig(
            hidden_size=hidden_size,
            query_key_size=hidden_size // self.attention_query_key_reduction,
            value_size=hidden_size // self.attention_value_reduction,
            key_value_pooling=self.attention_key_value_pooling,
        )

    def __getitem__(self, idx: int) -> BigGANGeneratorLayerConfig:
        input_size = self.hidden_size_list[0 if idx == 0 else idx - 1]
        return BigGANDiscriminatorLayerConfig(
            input_size=input_size,
            middle_size=input_size // self.bottleneck_reduction,
            output_size=self.hidden_size_list[idx],
            downsampling_factor=2 if idx % self.num_layers_per_block == 0 else 1,
        )

    def __iter__(self) -> Iterator[BigGANGeneratorLayerConfig]:
        yield from [self[i] for i, _ in enumerate(self.hidden_size_list)]


class CondBatchNorm2d(nn.Module):
    def __init__(self, hidden_size: int, conditional_size: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_size, affine=False)
        self.weight = nn.Linear(conditional_size, hidden_size)
        self.bias = nn.Linear(conditional_size, hidden_size)

    def forward(
        self, hidden_state: torch.Tensor, conditional_state: torch.Tensor
    ) -> torch.Tensor:
        weight, bias = self.weight(conditional_state), self.bias(conditional_state)
        hidden_state = self.batch_norm(hidden_state)
        return (1 + weight[:, :, None, None]) * hidden_state + bias[:, :, None, None]


class BigGANSelfAttention(nn.Module):
    def __init__(self, config: BigGANSelfAttentionConfig):
        super().__init__()
        self.query = nn.Conv2d(config.hidden_size, config.query_key_size, 1, bias=False)
        self.key = nn.Conv2d(config.hidden_size, config.query_key_size, 1, bias=False)
        self.value = nn.Conv2d(config.hidden_size, config.value_size, 1, bias=False)

        self.output = nn.Conv2d(config.value_size, config.hidden_size, 1, bias=False)
        self.gating = nn.Parameter(torch.zeros(()))

        self.maxpool = (
            nn.MaxPool2d(config.key_value_pooling, config.key_value_pooling)
            if config.key_value_pooling > 1
            else nn.Identity()
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        query = self.query(hidden_state)
        key = self.maxpool(self.key(hidden_state))
        value = self.maxpool(self.value(hidden_state))

        attention_probs = torch.matmul(query.flatten(2).transpose(1, 2), key.flatten(2))
        attention_probs = attention_probs.softmax(-1)
        attention_probs = attention_probs.transpose(1, 2)

        attention_state = torch.matmul(value.flatten(2), attention_probs)
        attention_state = attention_state.reshape(value.shape[:2] + query.shape[2:])

        return hidden_state + self.gating * self.output(attention_state)


class BigGANGeneratorLayer(nn.Module):
    def __init__(self, config: BigGANGeneratorLayerConfig):
        super().__init__()
        self.conv_1 = nn.Conv2d(config.input_size, config.middle_size, 1)
        self.conv_2 = nn.Conv2d(config.middle_size, config.middle_size, 3, padding=1)
        self.conv_3 = nn.Conv2d(config.middle_size, config.middle_size, 3, padding=1)
        self.conv_4 = nn.Conv2d(config.middle_size, config.output_size, 1)

        self.batch_norm_1 = CondBatchNorm2d(config.input_size, config.conditional_size)
        self.batch_norm_2 = CondBatchNorm2d(config.middle_size, config.conditional_size)
        self.batch_norm_3 = CondBatchNorm2d(config.middle_size, config.conditional_size)
        self.batch_norm_4 = CondBatchNorm2d(config.middle_size, config.conditional_size)

        self.upsampling = (
            nn.UpsamplingNearest2d(scale_factor=config.upsampling_factor)
            if config.upsampling_factor > 1
            else nn.Identity()
        )

    def forward(
        self, hidden_state: torch.Tensor, conditional_state: torch.Tensor
    ) -> torch.Tensor:
        residual_state = hidden_state

        hidden_state = self.batch_norm_1(hidden_state, conditional_state)
        hidden_state = self.conv_1(hidden_state.relu())

        hidden_state = self.batch_norm_2(hidden_state, conditional_state)
        hidden_state = self.upsampling(hidden_state.relu())
        hidden_state = self.conv_2(hidden_state)

        hidden_state = self.batch_norm_3(hidden_state, conditional_state)
        hidden_state = self.conv_3(hidden_state.relu())

        hidden_state = self.batch_norm_4(hidden_state, conditional_state)
        hidden_state = self.conv_4(hidden_state.relu())

        residual_state = residual_state[:, : hidden_state.size(1)]
        residual_state = self.upsampling(residual_state)

        return residual_state + hidden_state


class BigGANDiscriminatorLayer(nn.Module):
    def __init__(self, config: BigGANDiscriminatorLayerConfig):
        super().__init__()
        self.conv_1 = nn.Conv2d(config.input_size, config.middle_size, 1)
        self.conv_2 = nn.Conv2d(config.middle_size, config.middle_size, 3, padding=1)
        self.conv_3 = nn.Conv2d(config.middle_size, config.middle_size, 3, padding=1)
        self.conv_4 = nn.Conv2d(config.middle_size, config.output_size, 1)

        residual_size = config.output_size - config.input_size
        if residual_size > 0:
            self.conv_residual = nn.Conv2d(config.input_size, residual_size, 1)

        self.downsampling = (
            nn.AvgPool2d(config.downsampling_factor, config.downsampling_factor)
            if config.downsampling_factor > 1
            else nn.Identity()
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual_state = hidden_state

        hidden_state = self.conv_1(hidden_state.relu())
        hidden_state = self.conv_2(hidden_state.relu())
        hidden_state = self.conv_3(hidden_state.relu())
        hidden_state = self.downsampling(hidden_state.relu())
        hidden_state = self.conv_4(hidden_state)

        residual_state = self.downsampling(residual_state)
        if residual_state.size(1) != hidden_state.size(1):
            new_residual = self.conv_residual(residual_state)
            residual_state = torch.cat((residual_state, new_residual), dim=1)

        return residual_state + hidden_state


class BigGANGenerator(nn.Module):
    def __init__(self, config: BigGANGeneratorConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.num_classes, config.embedding_size)
        self.linear = nn.Linear(
            config.noise_size + config.embedding_size,
            config.first_hidden_resolution ** 2 * config.first_hidden_size,
        )
        self.layers = nn.ModuleList([BigGANGeneratorLayer(cfg) for cfg in config])
        self.attention = BigGANSelfAttention(config.attention_config)

        self.batch_norm = nn.BatchNorm2d(config.last_hidden_size)
        self.conv = nn.Conv2d(config.last_hidden_size, 3, 3, padding=1)

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                nn.init.orthogonal_(module.weight)
                nn.utils.parametrizations.spectral_norm(module)

    def forward_module(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        if self.config.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(module, *args, **kwargs)
        return module(*args, **kwargs)

    def forward(
        self, noise_vectors: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if noise_vectors.dtype != self.embeddings.weight.dtype:
            noise_vectors = noise_vectors.type_as(self.embeddings.weight)

        conditional_state = torch.cat((noise_vectors, self.embeddings(labels)), dim=1)
        hidden_state = self.linear(conditional_state).view(
            conditional_state.size(0),
            self.config.first_hidden_size,
            self.config.first_hidden_resolution,
            self.config.first_hidden_resolution,
        )

        for i, layer in enumerate(self.layers):
            hidden_state = self.forward_module(layer, hidden_state, conditional_state)
            if i == self.config.attention_layer_position:
                hidden_state = self.forward_module(self.attention, hidden_state)

        hidden_state = self.batch_norm(hidden_state).relu()
        hidden_state = self.conv(hidden_state)
        return hidden_state.tanh()


class BigGANDiscriminator(nn.Module):
    def __init__(self, config: BigGANDiscriminatorConfig):
        super().__init__()
        self.config = config

        self.conv = nn.Conv2d(3, config.first_hidden_size, 3, padding=1)
        self.layers = nn.ModuleList([BigGANDiscriminatorLayer(cfg) for cfg in config])
        self.attention = BigGANSelfAttention(config.attention_config)

        self.embeddings = nn.Embedding(config.num_classes, config.last_hidden_size)
        self.linear = nn.Linear(config.last_hidden_size, 1)

        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                nn.init.orthogonal_(module.weight)
                nn.utils.parametrizations.spectral_norm(module)

    def forward_module(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        if self.config.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(module, *args, **kwargs)
        return module(*args, **kwargs)

    def forward(self, input_images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if input_images.dtype != self.conv.weight.dtype:
            input_images = input_images.type_as(self.conv.weight.dtype)

        hidden_state = self.conv(input_images)
        for i, layer in enumerate(self.layers):
            hidden_state = self.forward_module(layer, hidden_state)
            if i == self.config.attention_layer_position:
                hidden_state = self.forward_module(self.attention, hidden_state)

        hidden_state = hidden_state.sum((2, 3))
        output_logits = (self.embeddings(labels) * hidden_state).sum(-1)
        output_logits = output_logits + self.linear(hidden_state).squeeze(-1)

        return output_logits
