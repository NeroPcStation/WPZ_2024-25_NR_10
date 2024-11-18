import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.f1 = nn.Linear(16 * 53 * 53, 256)
        self.f2 = nn.Linear(256, 128)
        self.f3 = nn.Linear(128, 5)

        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        try:
            self.load("CNN")
        except Exception:
            print("No checkpoint to load")
            self._initialize_weights()

    def forward(self, x):
        x = x.to(self.device)
        x = self.pool(F.relu(self.c1(x)))
        x = self.pool(F.relu(self.c2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        return torch.softmax(x, dim=-1)
    
    def save(self, filename="checkpoint"):
        torch.save(self.state_dict(), filename)
    
    def load(self, filename="checkpoint"):
        self.load_state_dict(torch.load(filename, weights_only=False))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
