from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    # def __init__(self):
    #     super().__init__()
    #     self.fc1 = nn.Linear(784, 128)
    #     self.fc2 = nn.Linear(128, 10)
        
    #     self.flatten = nn.Flatten()
    
    # def forward(self, x):
    #     print(x.shape)
    #     x = self.fc1(x)
    #     print(x.shape)
    #     x = self.flatten(x)
    #     x = self.fc2(x)
    #     print(x.shape)
    
    #     return x
    
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(1, 32, 3, padding=1)
        self.layer2 = nn.Conv2d(32, 64, 3, padding=1)
        self.layer3 = nn.Linear(64 * 28 * 28, 10)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.layer3(x)
        
        return x
