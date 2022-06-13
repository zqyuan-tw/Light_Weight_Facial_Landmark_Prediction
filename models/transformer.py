import torch
import torch.nn as nn
from vit_pytorch.mobile_vit import MobileViT

class Transformer(nn.Module):
    def __init__(self) -> None:
        super(Transformer, self).__init__()
        self.vit = MobileViT(
            image_size = (384, 384),
            dims = [96, 120, 144],
            channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
            num_classes = 136
        )            

    def forward(self, x):
        x = self.vit(x)

        return x.view(x.size(0), 68, 2)

if __name__ == "__main__":
    from torchinfo import summary
    m = Transformer()
    print(summary(m, input_size=(2, 3, 384, 384)))
    