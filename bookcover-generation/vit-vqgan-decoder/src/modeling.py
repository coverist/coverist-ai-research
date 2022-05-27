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


class PatchDiscriminator(nn.Sequential):
    def __init__(self, num_channels: int = 3, base_dim: int = 64):
        super().__init__(
            nn.Conv2d(num_channels, base_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_dim, 2 * base_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * base_dim, 4 * base_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * base_dim, 8 * base_dim, kernel_size=4, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8 * base_dim, 1, kernel_size=4, padding=1),
        )


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
