import torch
import torch.nn as nn


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