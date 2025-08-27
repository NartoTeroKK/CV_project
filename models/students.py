import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 64 * 64, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Residual Block
class ResidualBlock1(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock1, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x  # Skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Add skip connection
        return F.relu(out)



class StudentResNet1(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentResNet1, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial dimensions

        # Residual Block
        self.residual_block = ResidualBlock1(64)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Reduces spatial dim to 1x1

        # Fully Connected Layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = self.residual_block(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# StudentResNet2 deprecata perchè StudentResNet1 è più semplice e performante
'''
# Residual Block 2
# 
class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (se il numero di canali cambia, applico una convoluzione 1x1 per adattare i canali)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class StudentResNet2(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentResNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Strato iniziale
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Blocchi residuali
        self.layer1 = self.make_layer(64, 64, stride=1, num_blocks=2)
        self.layer2 = self.make_layer(64, 128, stride=2, num_blocks=2)
        self.layer3 = self.make_layer(128, 256, stride=2, num_blocks=2)

        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, in_channels, out_channels, stride, num_blocks):
        layers = []
        layers.append(ResidualBlock2(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
# '''


class StudentNetSeparable(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentNetSeparable, self).__init__()

        # Depthwise + Pointwise = Separable convolution
        self.depthwise = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
        self.pointwise = nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # forza output a 4x4

        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = F.relu(self.bn1(x))

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.adaptive_pool(x)  # Output shape: [batch, 64, 4, 4]

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x




class StudentNetDepthwiseSkip(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentNetDepthwiseSkip, self).__init__()

        # Depthwise + Pointwise
        self.depthwise = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3, bias=False)
        self.pointwise = nn.Conv2d(3, 32, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Seconda convoluzione
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Skip connection per conv2
        self.shortcut = nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False)

        # Pooling spaziale fisso: output [B, 64, 4, 4]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classificazione
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Depthwise + Pointwise
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = F.relu(self.bn1(x))  # [B, 32, H, W]

        # Residuo per lo skip
        residual = self.shortcut(x)  # [B, 64, H, W]

        # Conv2 + skip connection
        x = self.conv2(x)
        x = F.relu(self.bn2(x))  # [B, 64, H, W]
        x = x + residual  # somma skip

        # Adaptive pooling a [B, 64, 4, 4]
        x = self.adaptive_pool(x)

        # Flatten e classificazione
        x = x.view(x.size(0), -1)  # [B, 64*4*4]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x



class StudentNetLight(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentNetLight, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
