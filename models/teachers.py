import torch.nn as nn
import torchvision.models as models

class Teacher_resnet18(nn.Module):
    def __init__(self, num_classes=10):
        super(Teacher_resnet18, self).__init__()
        self.teacher = models.resnet18(pretrained=True)
        self.teacher.fc = nn.Linear(self.teacher.fc.in_features, num_classes)

    def forward(self, x):
        return self.teacher(x)


class Teacher_resnet50(nn.Module):
    def __init__(self, num_classes=10):
        super(Teacher_resnet50, self).__init__()
        self.teacher = models.resnet50(pretrained=True)
        self.teacher.fc = nn.Linear(self.teacher.fc.in_features, num_classes)

    def forward(self, x):
        return self.teacher(x)
    
class Teacher_efficientnet_b4(nn.Module):
    def __init__(self, num_classes=10):
        super(Teacher_efficientnet_b4, self).__init__()
        self.teacher = models.efficientnet_b4(weights= models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        # Modifica dell'ultimo layer per classificare le 10 classi di EuroSAT
        self.teacher.classifier[1] = nn.Linear(self.teacher.classifier[1].in_features, num_classes)  

    def forward(self, x):
        return self.teacher(x)
    
class Teacher_wideresnet50_2(nn.Module):
    def __init__(self, num_classes=10):
        super(Teacher_wideresnet50_2, self).__init__()
        self.teacher = models.wide_resnet50_2(pretrained=True)
        self.teacher.fc = nn.Linear(self.teacher.fc.in_features, num_classes)

    def forward(self, x):
        return self.teacher(x)
    
class Teacher_convnext_tiny(nn.Module):
    def __init__(self, num_classes=10):
        super(Teacher_convnext_tiny, self).__init__()
        self.teacher = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.teacher.classifier[2] = nn.Linear(self.teacher.classifier[2].in_features, num_classes)  

    def forward(self, x):
        return self.teacher(x)