import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torch.autograd import Function
import torch


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


#####
# UDA4POC

class Downsample(nn.Module):
    def __init__(self):
        super(Downsample, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))

        x3 = self.pool1(x2)
        x4 = self.relu(self.bn3(self.conv3(x3)))
        x5 = self.relu(self.bn4(self.conv4(x4)))

        x6 = self.pool1(x5)
        x7 = self.relu(self.bn5(self.conv5(x6)))
        x8 = self.relu(self.bn6(self.conv6(x7)))

        x9 = self.pool1(x8)
        x10 = self.relu(self.bn7(self.conv7(x9)))
        x11 = self.relu(self.bn8(self.conv8(x10)))

        return x2, x5, x8, x11


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv16 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm2d(128)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv17 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn17 = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn18 = nn.BatchNorm2d(64)

        self.conv19 = nn.Conv2d(64, 1, kernel_size=1)

        self.relu = nn.ReLU()
        self.classifier = nn.Sigmoid()

    def _merge(self, layer_down, layer_up):
        # slice_f = layer_up.size()[-1]//2
        # center = layer_down.size()[-1]//2
        # s,e = center-slice_f,center+slice_f
        #x_out = torch.cat((layer_down[:,:,s:e,s:e],(layer_up)),1)
        x_out = torch.cat((layer_down, layer_up), 1)
        return x_out

    def forward(self, x, x2, x5, x8):
        x19 = self.deconv2(x)
        x20 = self._merge(x19, x8)

        x21 = self.relu(self.bn13(self.conv13(x20)))
        x22 = self.relu(self.bn14(self.conv14(x21)))

        x23 = self.deconv3(x22)
        x24 = self._merge(x23, x5)

        x25 = self.relu(self.bn15(self.conv15(x24)))
        x26 = self.relu(self.bn16(self.conv16(x25)))

        x27 = self.deconv4(x26)
        x28 = self._merge(x27, x2)

        x29 = self.relu(self.bn17(self.conv17(x28)))
        x30 = self.relu(self.bn18(self.conv18(x29)))
        x31 = self.classifier(self.conv19(x30))

        return x31


class Adaptation(nn.Module):
    def __init__(self):
        super(Adaptation, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv_ad1 = nn.Conv2d(512, 512, kernel_size=3)
        self.bn_ad1 = nn.BatchNorm2d(512)

        self.conv_ad2 = nn.Conv2d(512, 256, kernel_size=3)
        self.bn_ad2 = nn.BatchNorm2d(256)

        self.conv_ad3 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn_ad3 = nn.BatchNorm2d(256)

        self.conv_ad4 = nn.Conv2d(256, 1024, kernel_size=3, padding=1)
        self.bn_ad4 = nn.BatchNorm2d(1024)

        self.conv_ad5 = nn.Conv2d(1024, 1, kernel_size=1)
        self.bn_ad5 = nn.BatchNorm2d(1)

        self.classifier = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, grl_lambda=1):
        x_ad0 = GradientReversalFn.apply(x, grl_lambda)
        x_ad1 = self.pool1(self.relu(self.bn_ad1(self.conv_ad1(x_ad0))))
        x_ad2 = self.pool1(self.relu(self.bn_ad2(self.conv_ad2(x_ad1))))
        x_ad3 = self.pool1(self.relu(self.bn_ad3(self.conv_ad3(x_ad2))))
        x_ad4 = self.pool1(self.relu(self.bn_ad4(self.conv_ad4(x_ad3))))
        x_ad5 = self.classifier((self.conv_ad5(x_ad4)))

        return x_ad5


class CountAdapt(nn.Module):
    def __init__(self):
        super(CountAdapt, self).__init__()

        self.downsample = Downsample()
        self.upsample = Upsample()
        self.adapt = Adaptation()

    def forward(self, x, grl_lambda=1):
        # features layers from feature extraction step
        x2, x5, x8, x11 = self.downsample(x)

        density_map = self.upsample(x11, x2, x5, x8)
        class_pred = self.adapt(x11)

        return density_map, class_pred
