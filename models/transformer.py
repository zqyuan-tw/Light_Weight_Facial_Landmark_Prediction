import torch
import torch.nn as nn
from vit_pytorch.mobile_vit import MobileViT

class Transformer(nn.Module):
    def __init__(self) -> None:
        super(Transformer, self).__init__()
        self.vit = MobileViT(
            image_size = (384, 384),
            dims = [120, 144, 168],
            channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
            num_classes = 136
        )
        self.tanh = nn.Tanh()            

    def forward(self, x):
        x = self.tanh(self.vit(x))

        return x.view(x.size(0), 68, 2)

if __name__ == "__main__":
    from torchinfo import summary
    m = Transformer()
    print(summary(m, input_size=(2, 3, 384, 384)))
    