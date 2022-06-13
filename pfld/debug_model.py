import torch.nn as nn
from torch.nn import functional
import torchvision.models as models

class AuxiliaryNet(nn.Module):

    def __init__(self, input_channels, nums_class=3, activation=nn.ReLU, first_conv_stride=2):
        super(AuxiliaryNet, self).__init__()
        self.input_channels = input_channels
        # self.num_channels = [128, 128, 32, 128, 32]
        self.num_channels = [512, 512, 512, 512, 1024]
        self.conv1 = nn.Conv2d(self.input_channels, self.num_channels[0], kernel_size=3, stride=first_conv_stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels[0])

        self.conv2 = nn.Conv2d(self.num_channels[0], self.num_channels[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels[1])

        self.conv3 = nn.Conv2d(self.num_channels[1], self.num_channels[2], kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels[2])

        self.conv4 = nn.Conv2d(self.num_channels[2], self.num_channels[3], kernel_size=7, stride=1, padding=3)
        self.bn4 = nn.BatchNorm2d(self.num_channels[3])

        self.fc1 = nn.Linear(in_features=self.num_channels[3], out_features=self.num_channels[4])
        self.fc2 = nn.Linear(in_features=self.num_channels[4], out_features=nums_class)

        self.activation = activation(inplace=True)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):

        out = self.conv1(input)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.activation(out)

        out = functional.adaptive_avg_pool2d(out, 1).squeeze(-1).squeeze(-1)
        #print(out.size())
        # out = out.view(out.size(0), -1)
        out = self.fc1(out)
        euler_angles_pre = self.fc2(out)

        return euler_angles_pre


class ResNet34(nn.Module):
    
    def __init__(self, nums_class=136):
        super(ResNet34, self).__init__()

        self.resnest = models.resnet18(pretrained=False)
        self.resnest_backbone1 = nn.Sequential(*list(self.resnest.children())[:-6])
        self.resnest_backbone_end = nn.Sequential(*list(self.resnest.children())[-6:-2])
        
        self.in_features = 2048
        # ResNet50
        # 1. 3 X 256 X 256: 131072
        # 2. 3 X 128 X 128: 32768
        # ResNet34
        # 1. 3 X 256 X 256: 32768
        # 2. 3 X 128 X 128: 8192
        # RenNet18 
        # 1. 3 X 128 X 128: 8192
        # 2. 3 X 64 X 64: 2048
        self.fc = nn.Linear(in_features=self.in_features, out_features=nums_class)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        auxnet = self.resnest_backbone1(x)
        print(auxnet.size())
        out = self.resnest_backbone_end(auxnet)
        #print(out.size())
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, auxnet 


if __name__ == "__main__":
    import torch
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # input = torch.randn(2, 3, 192, 192).to(device)
    # input = torch.randn(2, 3, 384, 384).to(device)
    input = torch.randn(2, 3, 256, 256).to(device)

    #------------------------------------------------------------------
    """MobileNet Debugging Area"""
    # coefficient = 1
    # num_of_channels = [int(64 * coefficient), int(128 * coefficient), int(16 * coefficient), int(32 * coefficient), int(128 * coefficient)]
    # m = MobileNetV2(num_of_channels=num_of_channels).to(device)
    # m2 = AuxiliaryNet(input_channels=num_of_channels[0]).to(device)
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    """BlazeLandmark Debugging Area"""
    # m = BlazeLandMark(nums_class=136).to(device)
    # aux = AuxiliaryNet(input_channels=48, first_conv_stride=2).to(device)
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    """ResNet18 Debugging Area"""
    m = ResNet34(nums_class=136).to(device)
    aux = AuxiliaryNet(input_channels=64, first_conv_stride=2).to(device)
    #------------------------------------------------------------------
    
    # pre_landmarks, output = m(input)
    # m2_out = aux(output)
    # print(pre_landmarks.shape, output.shape)
    # print(m2_out.shape)
    # print(output.size())
    summary(m, (3, 64, 64))