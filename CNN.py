import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.c2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.f1 = nn.Linear(16 * 53 * 53, 22492)
        self.f2 = nn.Linear(22492, 11247)
        self.f3 = nn.Linear(11247, 5)

        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        self.load_state_dict(torch.load(filename))
