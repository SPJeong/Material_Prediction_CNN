##### model.py

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels=1, descriptor_size=193):  # len(descriptor_df.columns) = 193
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2 * in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2 * in_channels, out_channels=4 * in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4 * in_channels, out_channels=8 * in_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=8 * in_channels, out_channels=16 * in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(16 * 16 * 16 + descriptor_size, 1024)  # CNN output size (16*16*16) + descriptor size
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, x, descriptors):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.maxpool(out)  # pooling (batch_size,16,16,16)
        out = self.dropout(out)

        out = out.view(out.size(0), -1)  # flattern
        out = torch.cat((out, descriptors), dim=1)  # concat descriptor layers
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)

        return out


def count_parameters(model):
    """
    Calculates the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Calculate and print the total parameters
if __name__ == '__main__':
    my_deep_model = CNN()
    total_params = count_parameters(my_deep_model)
    print(f"Total trainable parameters: {total_params:,}")