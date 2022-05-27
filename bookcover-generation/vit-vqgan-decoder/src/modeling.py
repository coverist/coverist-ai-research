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
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        patches = patches.view(-1, 3, image_size, image_size)
        return patches.tanh()


class Discriminator(nn.Sequential):
    def __init__(self, num_channels: int, hidden_dims: list[int], strides: list[int]):
        hidden_dims, strides = [num_channels] + hidden_dims, [1] + strides

        layers = []
        for in_dim, out_dim, stride in zip(hidden_dims[:-1], hidden_dims[1:], strides):
            layers.append(nn.Conv2d(in_dim, out_dim, 3, stride, padding=1))
            layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(hidden_dims[-1], 1))
        super().__init__(*layers)


class OCRPerceptualLoss(nn.Module):
    def __init__(
        self, input_size: tuple[int, int] = (192, 192), language: list[str] = ["ko"]
    ):
        super().__init__()
        self.input_size = input_size

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
