import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet  # pip install efficientnet_pytorch

def get_model(model_name='resnet50'):
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=None)
        model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif model_name == 'efficientnet_b0':
        model = EfficientNet.from_name('efficientnet-b0')
        model._conv_stem = nn.Conv2d(1, model._conv_stem.out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        model._fc = nn.Linear(model._fc.in_features, 2)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model