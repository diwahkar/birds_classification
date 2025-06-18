import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

from constants import USE_PRETRAINED

class BirdsClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        if USE_PRETRAINED:
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            for param in self.model.parameters():
                param.requires_grad = False

            num_ftrs =self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=num_ftrs, out_features=num_classes)

        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm2d(num_features=64)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm2d(num_features=128)
            self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
            self.bn3 = nn.BatchNorm2d(num_features=256)

            self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
            self._to_linear = 256

            self.fc1 = nn.Linear(in_features=self._to_linear, out_features=512)
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(in_features=512, out_features=num_classes)


    def convs(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), (2,2))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2,2))
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2,2))
        return x


    def forward(self, x):
        if USE_PRETRAINED:
            x = self.model(x)
        else:
            x = self.convs(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout1(F.relu(self.fc1(x)))
            x = self.fc2(x)
        return x
