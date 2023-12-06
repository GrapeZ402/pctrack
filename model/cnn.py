#cnn
import torch
import torch.nn as nn

class CNNPred(nn.Module):
    def __init__(self, in_channels,output_size,hidden_size):
        super(CNNPred, self).__init__()

        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
        )
        self.fc2 = nn.Linear(hidden_size*3, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        #(10,5,3)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(batch_size,-1)
        x = self.fc2(x)
        return x.view(-1,5,4)

