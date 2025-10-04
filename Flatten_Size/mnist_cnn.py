# mnist_cnn.py
import torch
import torch.nn as nn

def get_flatten_size(model, input_shape=(1, 28, 28)):
    dummy = torch.zeros(1, *input_shape)
    output = model(dummy)
    return output.view(1, -1).shape[1]

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flatten_dim = get_flatten_size(self.features, input_shape=(1, 28, 28))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
