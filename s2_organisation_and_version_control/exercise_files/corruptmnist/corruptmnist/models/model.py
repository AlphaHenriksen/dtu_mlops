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

    def __init__(self, x_dim, hidden_dim, latent_dim, output_dim, kernel_size, padding, dropout):
        super().__init__()
        self.layer1 = nn.Conv2d(1, hidden_dim, kernel_size, padding=padding)
        self.layer2 = nn.Conv2d(hidden_dim, latent_dim, kernel_size, padding=padding)
        self.layer3 = nn.Linear(latent_dim * x_dim, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.layer3(x)

        return x
