import torch
import torch.nn as nn
import torchvision

import model.config as config


class BaseModel(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()

        self.net = nn.Sequential([
            nn.Conv2d(in_channels, hidden_size, kernel_size= 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_size, out_channels, kernel_size= 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear() # Burasini duzenleyecegim
        ])


    def forward(self, x):
        
        x = self.net(x)
        return x


def get_model(model_name, class_names, full_train = False, pretrained = False):
    
    if model_name == 'resnet50':
        if pretrained:
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT).to(config.DEVICE)
        else:
            model = torchvision.models.resnet50().to(config.DEVICE)
        
        if full_train == False:
            for parameter in model.parameters():
                parameter.requires_grad = False
            
        model.fc = nn.Linear(in_features=2048, out_features=len(class_names)).to(config.DEVICE)
        return model