import torch.nn as nn



class SimpleCNN(nn.Module):
    def __init__(self, image_channels):
        super(SimpleCNN, self).__init__()

        # Store multiple layers in a ModuleList
        self.all_encs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(image_channels, 3, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ) for _ in range(1)] +  # First set of layers
            
            [nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ) for _ in range(8)] +  # Second set of layers

            [nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ) for _ in range(2)] +  # third set of layers

            [nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            ) for _ in range(4)] +  # third set of layers

            [nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU()
            ) for _ in range(2)] +  # third set of layers

            [nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU()
            ) for _ in range(1)]  # third set of layers

        )

    def forward(self, x):
        for encoder in self.all_encs:
            x = encoder(x)
        return x