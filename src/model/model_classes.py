import torch
import torch.nn as nn
import torchvision

import model.config as config

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
    elif model_name == 'densenet161':
      if pretrained:
          model = torchvision.models.densenet161(weights=torchvision.models.DenseNet161_Weights.DEFAULT).to(config.DEVICE)
      else:
          model = torchvision.models.densenet161().to(config.DEVICE)
      
      if full_train == False:
          for parameter in model.parameters():
              parameter.requires_grad = False
          
      model.classifier = nn.Linear(in_features=2208, out_features=len(class_names), bias=True).to(config.DEVICE)
      return model
    
    elif model_name == 'resnext':
      if pretrained:
          model = torchvision.models.resnext101_32x8d(weights=torchvision.models.ResNeXt101_32X8D_Weights.DEFAULT).to(config.DEVICE)
      else:
          model = torchvision.models.resnext101_32x8d().to(config.DEVICE)
      
      if full_train == False:
          for parameter in model.parameters():
              parameter.requires_grad = False
          
      model.classifier = nn.Linear(in_features=2048, out_features=len(class_names), bias=True).to(config.DEVICE)
      
      return model

    elif model_name == 'inception':
      if pretrained:
          model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT).to(config.DEVICE)
      else:
          model = torchvision.models.inception_v3().to(config.DEVICE)
      
      if full_train == False:
          for parameter in model.parameters():
              parameter.requires_grad = False
          
      model.classifier = nn.Linear(in_features=2048, out_features=len(class_names), bias=True).to(config.DEVICE)
      
      return model