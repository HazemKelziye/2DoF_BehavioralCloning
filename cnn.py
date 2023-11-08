import torch
import torch.nn as nn

class CNN2D(nn.Module):
  def __init__(self):
    super().__init__()

    # Define the convolutional block
    self.conv_block = nn.Sequential(
        nn.Conv2d(1, 10, kernel_size=(3, 3)),
        nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1)),
        nn.ReLU()
    )

    # Define the fully connected layers
    self.fc_layers = nn.Sequential(
        nn.Linear(3 * 7 * 10, 105),
        nn.ReLU(),
        nn.Linear(105, 6)
    )

  def forward(self, x):
    # apply the convolutional block
    x = self.conv_block(x)

    # reshape the output tensor for the fully connected layers
    # x = nn.Flatten()
    x = x.view(x.size(0), -1)

    # apply the fully connected layers
    x = self.fc_layers(x)

    return x