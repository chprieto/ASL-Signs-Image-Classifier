import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, 5)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 4, 5)
        self.bn2 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(4, 8, 5)
        self.bn4 = nn.BatchNorm2d(8)
        self.conv5 = nn.Conv2d(8, 8, 5)
        self.bn5 = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(8 * 2 * 2, 26)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 8 * 2 * 2)
        output = self.fc1(output)

        return output
