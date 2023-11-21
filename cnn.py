import torch
import torch.nn as nn


class CNN2D_2headed_3(nn.Module):
    def __init__(self):
        super(CNN2D_2headed_3, self).__init__()

        # Define the convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 5)),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=(1, 4)),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 2)),
            nn.ReLU()
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=(1, 2)),
            nn.ReLU()
        )

        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(160, 320),
            nn.ReLU(),
            nn.Linear(320, 640),
            nn.ReLU(),
            nn.Linear(640, 640),
            nn.ReLU(),
            nn.Linear(640, 320),
            nn.ReLU(),
            nn.Linear(320, 160),
            nn.ReLU()
        )

        self.fcl_output1 = nn.Linear(160, 6)
        self.fcl_output2 = nn.Linear(160, 6)

    def forward(self, x):
        # apply the convolutional block
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        # reshape the output tensor for the fully connected layers
        # x = nn.Flatten()
        x = x.view(x.size(0), -1)

        # apply the fully connected layers
        x = self.fc_layers(x)

        # calculate the outputs separately
        output1 = self.fcl_output1(x)
        output2 = self.fcl_output2(x)

        return output1, output2
