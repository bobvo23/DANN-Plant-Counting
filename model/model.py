import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torch.autograd import Function


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        # code to comprehence flexible 1-3 channels
        x = x.expand(x.shape[0], 3, x.shape[-2], x.shape[-1])
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, λ):
        # Store context for backprop
        ctx.λ = λ

        # Forward pass is a no-op
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is dL/dx (since our forward's output was x)

        # Backward pass is just to apply -λ to the gradient
        # This will become the new dL/dx in the rest of the network
        output = - ctx.λ * grad_output

        # Must return number of inputs to forward()
        return output, None


class MnistAdapt(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=1, stride=1),
            # (28+2P-F)/S + 1 = 26
            nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.ReLU(True),
            # 26 / 2 = 13
            nn.Conv2d(64, 50, kernel_size=5, padding=1, stride=1),
            # (12+2P-F)/S + 1 = 10
            nn.BatchNorm2d(50), nn.MaxPool2d(2), nn.ReLU(True),    # 10 / 2 = 5
            nn.Dropout2d(),
        )

        self.num_cnn_features = 50 * 5 * 5  # Assuming 28x28 input

        self.class_classifier = nn.Sequential(
            nn.Linear(self.num_cnn_features, 100),
            nn.BatchNorm1d(100), nn.Dropout2d(), nn.ReLU(True),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100), nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(self.num_cnn_features, 100),
            nn.BatchNorm1d(100), nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x, λ=1.0):
        # Handle single-channel input by expanding (repeating) the singleton dimention
        x = x.expand(x.shape[0], 3, x.shape[-2], x.shape[-1])

        features = self.feature_extractor(x)
        features = features.view(-1, self.num_cnn_features)
        features_grl = GradientReversalFn.apply(features, λ)
        # classify on regular features
        class_pred = self.class_classifier(features)
        domain_pred = self.domain_classifier(features_grl)
        # classify on features after GRL
        return class_pred, domain_pred
