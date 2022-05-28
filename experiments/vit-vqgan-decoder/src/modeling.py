import easyocr
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchToImage(nn.Module):
    def __init__(self, num_channels: int = 3):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        num_patches = int(patches.size(1) ** 0.5)
        patch_size = int((patches.size(2) // self.num_channels) ** 0.5)
        image_size = num_patches * patch_size

        patches = patches.view(-1, num_patches, num_patches, 3, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        patches = patches.view(-1, 3, image_size, image_size)
        return patches.tanh()


class SNGANDiscriminatorLayer(nn.Module):
    def __init__(
        self, input_dim: int, middle_dim: int, output_dim: int, stride: int = 1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, middle_dim, 1)
        self.conv2 = nn.Conv2d(middle_dim, middle_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(middle_dim, middle_dim, 3, stride, padding=1)
        self.conv4 = nn.Conv2d(middle_dim, output_dim, 1)

        if output_dim != input_dim or stride > 1:
            self.shortcut = nn.Conv2d(input_dim, output_dim, 1, stride)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shortcut = hidden
        hidden = self.conv1(hidden.relu())
        hidden = self.conv2(hidden.relu())
        hidden = self.conv3(hidden.relu())
        hidden = self.conv4(hidden.relu())

        if shortcut.shape != hidden.shape:
            shortcut = self.shortcut(shortcut)
        return hidden + shortcut


class SNGANDiscriminator(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        base_dim: int = 256,
        max_hidden_dim: int = 2048,
        middle_reduction: int = 4,
        num_blocks: int = 5,
        num_layers_in_block: int = 2,
    ):
        super().__init__()
        layers, last_hidden_dim = [], base_dim
        for i in range(num_blocks):
            hidden_dim = min(base_dim * 2 ** i, max_hidden_dim)
            for j in range(num_layers_in_block):
                is_last_layer = i < num_blocks - 1 and j == num_layers_in_block - 1
                layer = SNGANDiscriminatorLayer(
                    last_hidden_dim,
                    hidden_dim // middle_reduction,
                    hidden_dim,
                    stride=2 if is_last_layer else 1,
                )
                layers.append(layer)
                last_hidden_dim = hidden_dim

        self.conv = nn.Conv2d(num_channels, base_dim, 7, stride=2, padding=1)
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(min(last_hidden_dim, max_hidden_dim), 1)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(module.weight)
                nn.utils.parametrizations.spectral_norm(module, eps=1e-6)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(images.type_as(self.conv.weight))
        hidden = self.layers(hidden).relu().sum((2, 3))
        return self.linear(hidden).squeeze(1)


class OCRPerceptualLoss(nn.Module):
    def __init__(
        self, input_size: tuple[int, int] = (192, 192), language: list[str] = ["ko"]
    ):
        super().__init__()
        self.input_size = tuple(input_size)

        self.model = easyocr.Reader(language).detector.module.basenet
        self.model.requires_grad_(False)

        self.register_buffer("shift", torch.tensor([[[[0.03]], [[0.088]], [[0.188]]]]))
        self.register_buffer("scale", torch.tensor([[[[0.458]], [[0.448]], [[0.45]]]]))

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        images = (images + self.shift) / self.scale
        images = F.interpolate(images, self.input_size, mode="bilinear")

        features = self.model.eval()(images)
        features = [F.normalize(feature, dim=1, eps=1e-6) for feature in features]
        return features

    def forward(self, images: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
        loss = 0
        images_features = self.forward_features(images)
        decoded_features = self.forward_features(decoded)

        for images_feature, decoded_feature in zip(images_features, decoded_features):
            loss = loss + (images_feature - decoded_feature).square().sum(1).mean()
        return loss
