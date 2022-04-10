import easyocr
import torch
import torch.nn as nn
import torch.nn.functional as F


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = easyocr.Reader(["ko"]).detector.module.basenet
        self.model.requires_grad_(False)

        self.projections = nn.ModuleList(
            nn.Conv2d(hidden_dim, 1, 1, bias=False)
            for hidden_dim in [1024, 512, 512, 256, 128]
        )
        self.register_buffer("shift", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("scale", torch.tensor([0.229, 0.224, 0.225]))

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        images = (images + 1) / 2
        images = images - self.shift[None, :, None, None]
        images = images / self.scale[None, :, None, None]

        features = self.model.eval()(images)
        features = [F.normalize(feature, dim=1, eps=1e-6) for feature in features]
        return features

    def forward(self, images: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        features_images = self.forward_features(images)
        features_recon = self.forward_features(recon)

        return sum(
            layer((features_images[i] - features_recon[i]) ** 2).mean()
            for i, layer in enumerate(self.projections)
        )
