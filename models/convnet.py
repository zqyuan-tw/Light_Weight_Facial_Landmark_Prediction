import torch
import torch.nn as nn
import torchvision.models as models

class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 68 * 2),
            nn.Tanh()
        )
            

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)

        return x.view(x.size(0), 68, 2)
        

if __name__ == "__main__":
    from torchinfo import summary
    m = ConvNet()
    print(summary(m, input_size=(2, 3, 384, 384)))