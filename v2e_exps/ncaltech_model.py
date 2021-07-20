"""NCaltech 101 experiment models.

Author: Yuhuang Hu
Email : yuhuang.hu@ini.uzh.ch
"""

import torch
from torch import nn

from torchvision.models import resnet34


class V2ENCaltechModel(nn.Module):
    def __init__(self, pretrained=False, num_voxel_bins=3, use_dropout=False):
        super(V2ENCaltechModel, self).__init__()

        self.pretrained = pretrained
        self.use_dropout = use_dropout

        backbone = resnet34(pretrained=pretrained)
        self.num_filters = backbone.fc.in_features

        # deal with channel not 3
        if num_voxel_bins != 3:
            self.input_conv = nn.Conv2d(
                in_channels=num_voxel_bins,
                out_channels=64, kernel_size=7,
                stride=2, padding=3, bias=False)

            layers = [self.input_conv]+list(backbone.children())[1:-1]
        else:
            layers = list(backbone.children())[:-1]

        self.feature_extractor = torch.nn.Sequential(*layers)

        self.num_classes = 101
        self.classifier = nn.Linear(self.num_filters, self.num_classes)

        if use_dropout:
            self.dropout = nn.Dropout(0.85)

    def forward(self, x):
        x = self.feature_extractor(x).flatten(1)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.classifier(x)

        return x


if __name__ == "__main__":
    model = V2ENCaltechModel(False, 6)
