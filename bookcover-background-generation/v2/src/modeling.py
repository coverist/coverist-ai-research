from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn


@dataclass
class BigGANSelfAttentionConfig:
    hidden_dim: int
    query_key_dim: int
    value_dim: int
    key_value_pooling: int


@dataclass
class BigGANGeneratorLayerConfig:
    input_dim: int
    middle_dim: int
    output_dim: int
    conditional_dim: int
    upsampling: int


@dataclass
class BigGANDiscriminatorLayerConfig:
    input_dim: int
    middle_dim: int
    output_dim: int
    downsampling: int


@dataclass
class BigGANGeneratorConfig:
    num_labels: int
    latent_dim: int
    embedding_dim: int
    hidden_dims: list[int]
    layers_in_block: int = 2
    middle_reduction: int = 4
    first_image_size: int = 4
    attn_position: int = 7
    attn_query_key_reduction: int = 8
    attn_value_reduction: int = 2
    attn_key_value_pooling: int = 2
    use_grad_ckpt: bool = False

    @property
    def first_hidden_dim(self) -> int:
        return self.hidden_dims[0]

    @property
    def last_hidden_dim(self) -> int:
        return self.hidden_dims[-1]

    @property
    def conditional_dim(self) -> int:
        return self.latent_dim + self.embedding_dim

    @property
    def latent_image_dim(self) -> int:
        return self.first_image_size ** 2 * self.first_hidden_dim

    @property
    def attention_config(self) -> BigGANSelfAttentionConfig:
        hidden_dim = self.hidden_dims[self.attn_position]
        return BigGANSelfAttentionConfig(
            hidden_dim=hidden_dim,
            query_key_dim=hidden_dim // self.attn_query_key_reduction,
            value_dim=hidden_dim // self.attn_value_reduction,
            key_value_pooling=self.attn_key_value_pooling,
        )

    def __getitem__(self, index: int) -> BigGANGeneratorLayerConfig:
        return BigGANGeneratorLayerConfig(
            input_dim=self.hidden_dims[max(0, index - 1)],
            middle_dim=self.hidden_dims[max(0, index - 1)] // self.middle_reduction,
            output_dim=self.hidden_dims[index],
            conditional_dim=self.conditional_dim,
            upsampling=2 if (index + 1) % self.layers_in_block == 0 else 1,
        )

    def __iter__(self) -> Iterator[BigGANGeneratorLayerConfig]:
        return (self[i] for i in range(len(self.hidden_dims)))


@dataclass
class BigGANDiscriminatorConfig:
    num_labels: int
    hidden_dims: list[int]
    layers_in_block: int = 2
    middle_reduction: int = 4
    attn_position: int = 1
    attn_query_key_reduction: int = 8
    attn_value_reduction: int = 2
    attn_key_value_pooling: int = 2
    use_grad_ckpt: bool = False

    @property
    def first_hidden_dim(self) -> int:
        return self.hidden_dims[0]

    @property
    def last_hidden_dim(self) -> int:
        return self.hidden_dims[-1]

    @property
    def attention_config(self) -> BigGANSelfAttentionConfig:
        hidden_dim = self.hidden_dims[self.attn_position]
        return BigGANSelfAttentionConfig(
            hidden_dim=hidden_dim,
            query_key_dim=hidden_dim // self.attn_query_key_reduction,
            value_dim=hidden_dim // self.attn_value_reduction,
            key_value_pooling=self.attn_key_value_pooling,
        )

    def __getitem__(self, index: int) -> BigGANDiscriminatorLayerConfig:
        return BigGANDiscriminatorLayerConfig(
            input_dim=self.hidden_dims[max(0, index - 1)],
            middle_dim=self.hidden_dims[max(0, index - 1)] // self.middle_reduction,
            output_dim=self.hidden_dims[index],
            upsampling=2 if (index + 1) % self.layers_in_block == 0 else 1,
        )

    def __iter__(self) -> Iterator[BigGANDiscriminatorLayerConfig]:
        return (self[i] for i in range(len(self.hidden_dims)))


class CondBatchNorm2d(nn.Module):
    def __init__(self, hidden_dim: int, conditional_dim: int):
        super().__init__()
        self.weight = nn.Linear(conditional_dim, hidden_dim)
        self.bias = nn.Linear(conditional_dim, hidden_dim)
        self.bn = nn.BatchNorm2d(hidden_dim, eps=1e-4, affine=False)

    def forward(self, hidden: torch.Tensor, conditional: torch.Tensor) -> torch.Tensor:
        weight, bias = self.weight(conditional), self.bias(conditional)
        return (1 + weight[:, :, None, None]) * self.bn(hidden) + bias[:, :, None, None]


class BigGANSelfAttention(nn.Module):
    def __init__(self, config: BigGANSelfAttentionConfig):
        super().__init__()
        self.maxpool = nn.MaxPool2d(config.key_value_pooling, config.key_value_pooling)

        self.query = nn.Conv2d(config.hidden_dim, config.query_key_dim, 1, bias=False)
        self.key = nn.Conv2d(config.hidden_dim, config.query_key_dim, 1, bias=False)
        self.value = nn.Conv2d(config.hidden_dim, config.value_dim, 1, bias=False)

        self.gating = nn.Parameter(torch.zeros(()))
        self.output = nn.Conv2d(config.value_dim, config.hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query, key, value = self.query(hidden), self.key(hidden), self.value(hidden)
        key, value = self.maxpool(key), self.maxpool(value)

        attention_probs = torch.matmul(query.flatten(2).transpose(1, 2), key.flatten(2))
        attention_probs = attention_probs.softmax(-1).transpose(1, 2)

        attention_output = torch.matmul(value.flatten(2), attention_probs)
        attention_output = attention_output.reshape(value.shape[:2] + query.shape[2:])

        return hidden + self.gating * self.output(attention_output)


class BigGANGeneratorLayer(nn.Module):
    def __init__(self, config: BigGANGeneratorLayerConfig):
        super().__init__()
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=config.upsampling)

        self.bn1 = CondBatchNorm2d(config.input_dim, config.conditional_dim)
        self.bn2 = CondBatchNorm2d(config.middle_dim, config.conditional_dim)
        self.bn3 = CondBatchNorm2d(config.middle_dim, config.conditional_dim)
        self.bn4 = CondBatchNorm2d(config.middle_dim, config.conditional_dim)

        self.conv1 = nn.Conv2d(config.input_dim, config.middle_dim, 1)
        self.conv2 = nn.Conv2d(config.middle_dim, config.middle_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(config.middle_dim, config.middle_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(config.middle_dim, config.output_dim, 1)

    def forward(self, hidden: torch.Tensor, conditional: torch.Tensor) -> torch.Tensor:
        shortcut = hidden
        hidden = self.conv1(self.bn1(hidden, conditional).relu())
        hidden = self.conv2(self.upsampling(self.bn2(hidden, conditional).relu()))
        hidden = self.conv3(self.bn3(hidden, conditional).relu())
        hidden = self.conv4(self.bn4(hidden, conditional).relu())

        shortcut = self.upsampling(shortcut[:, : hidden.size(1)])
        return hidden + shortcut


class BigGANDiscriminatorLayer(nn.Module):
    def __init__(self, config: BigGANDiscriminatorLayerConfig):
        super().__init__()
        self.downsampling = nn.AvgPool2d(config.downsampling, config.downsampling)

        self.conv1 = nn.Conv2d(config.input_dim, config.middle_dim, 1)
        self.conv2 = nn.Conv2d(config.middle_dim, config.middle_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(config.middle_dim, config.middle_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(config.middle_dim, config.output_dim, 1)

        shortcut_dim = config.output_dim - config.input_dim
        if shortcut_dim > 0:
            self.shortcut = nn.Conv2d(config.input_dim, shortcut_dim, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shortcut = hidden
        hidden = self.conv1(hidden.relu())
        hidden = self.conv2(hidden.relu())
        hidden = self.conv3(hidden.relu())
        hidden = self.conv4(self.downsampling(hidden.relu()))

        shortcut = self.downsampling(shortcut)
        if shortcut.size(1) != hidden.size(1):
            shortcut = torch.cat((shortcut, self.shortcut(shortcut)), dim=1)
        return hidden + shortcut


class BigGANGenerator(nn.Module):
    def __init__(self, config: BigGANGeneratorConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.num_labels, config.embedding_dim)
        self.linear = nn.Linear(config.conditional_dim, config.latent_image_dim)
        self.layers = nn.ModuleList(map(BigGANGeneratorLayer, config))
        self.attention = BigGANSelfAttention(config.attention_config)

        self.bn = nn.BatchNorm2d(config.last_hidden_dim, eps=1e-4)
        self.conv = nn.Conv2d(config.last_hidden_dim, 3, 3, padding=1)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                nn.init.orthogonal_(module.weight)
                nn.utils.parametrizations.spectral_norm(module, eps=1e-6)

    def forward(self, latents: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        latents = latents.type_as(self.embeddings.weight)

        conditional = torch.cat((latents, self.embeddings(labels)), dim=1)
        hidden = self.linear(conditional).view(
            conditional.size(0),
            self.config.first_hidden_dim,
            self.config.first_image_size,
            self.config.first_image_size,
        )

        for i, layer in enumerate(self.layers):
            hidden = (
                torch.utils.checkpoint.checkpoint(layer, hidden, conditional)
                if self.config.use_grad_ckpt and self.training
                else layer(hidden, conditional)
            )
            if i == self.config.attn_position:
                hidden = (
                    torch.utils.checkpoint.checkpoint(self.attention, hidden)
                    if self.config.use_grad_ckpt and self.training
                    else self.attention(hidden)
                )
        return self.conv(self.bn(hidden).relu()).tanh()


class BigGANDiscriminator(nn.Module):
    def __init__(self, config: BigGANDiscriminatorConfig):
        super().__init__()
        self.config = config

        self.conv = nn.Conv2d(3, config.first_hidden_dim, 3, padding=1)
        self.layers = nn.ModuleList(map(BigGANDiscriminatorLayer, config))
        self.attention = BigGANSelfAttention(config.attention_config)

        self.embeddings = nn.Embedding(config.num_labels, config.last_hidden_dim)
        self.linear = nn.Linear(config.last_hidden_dim, 1)

        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
                nn.init.orthogonal_(module.weight)
                nn.utils.parametrizations.spectral_norm(module, eps=1e-6)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images = images.type_as(self.conv.weight)
        hidden = self.conv(images)

        for i, layer in enumerate(self.layers):
            hidden = (
                torch.utils.checkpoint.checkpoint(layer, hidden)
                if self.config.use_grad_ckpt and self.training
                else layer(hidden)
            )
            if i == self.config.attn_position:
                hidden = (
                    torch.utils.checkpoint.checkpoint(self.attention, hidden)
                    if self.config.use_grad_ckpt and self.training
                    else self.attention(hidden)
                )

        hidden = hidden.relu().sum((2, 3))
        general_logits = self.linear(hidden).squeeze(1)
        projection_logits = (self.embeddings(labels) * hidden).sum(1)
        return general_logits + projection_logits
