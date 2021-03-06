import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import transforms, models, datasets

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class flowerModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()

        ### 设置哪些层需要训练
    def set_parameter_requires_grad(model, feature_extracting):
        #是否训练此层
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    """ Resnet152"""
    model_ft = models.resnet152(pretrained=True)
    set_parameter_requires_grad(model_ft, True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 10),
                                nn.LogSoftmax(dim=1))
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class catdogModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()

        ### 设置哪些层需要训练
    def set_parameter_requires_grad(model, feature_extracting):
        #是否训练此层
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    """ Resnet152"""
    model_ft = models.resnet152(pretrained=True)
    set_parameter_requires_grad(model_ft, True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 2),
                                nn.LogSoftmax(dim=1))
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
