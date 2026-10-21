import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


#decoder- unet
class DecoderBlock(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

#encoder 1: vgg16 with batch normalization 
class VGG16Encoder(nn.Module):
    """
    VGG16-BN split into 5 stages at each MaxPool boundary.
    Stage 1 output is at full input resolution (256x256) — no downsampling yet.
    """

    def __init__(self):
        super().__init__()
        vgg = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)

        stages, current = [], []
        for layer in vgg.features:
            if isinstance(layer, nn.MaxPool2d):
                stages.append(nn.Sequential(*current))
                current = []
            else:
                current.append(layer)

        self.stage1 = stages[0]   # 64ch,  256x256
        self.stage2 = stages[1]   # 128ch, 128x128
        self.stage3 = stages[2]   # 256ch,  64x64
        self.stage4 = stages[3]   # 512ch,  32x32
        self.stage5 = stages[4]   # 512ch,  16x16
        self.pool   = nn.MaxPool2d(2, 2)

        self.skip_channels       = [64, 128, 256, 512, 512]
        self.bottleneck_channels = 512

    def forward(self, x):
        s1 = self.stage1(x);  x = self.pool(s1)
        s2 = self.stage2(x);  x = self.pool(s2)
        s3 = self.stage3(x);  x = self.pool(s3)
        s4 = self.stage4(x);  x = self.pool(s4)
        s5 = self.stage5(x);  x = self.pool(s5)
        return x, [s1, s2, s3, s4, s5]


#encoder 2: resnet34
class ResNet34Encoder(nn.Module):
    """
    ResNet34 split into 5 skip levels.
    Stage 0 (conv1+bn+relu) downsamples by 2, maxpool adds another /2 before layer1.
    The shallowest skip is therefore 64x64, not 256x256 like VGG16.
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        self.stage0  = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1   # 64ch,  64x64
        self.layer2  = resnet.layer2   # 128ch, 32x32
        self.layer3  = resnet.layer3   # 256ch, 16x16
        self.layer4  = resnet.layer4   # 512ch,  8x8

        self.skip_channels       = [64, 64, 128, 256, 512]
        self.bottleneck_channels = 512

    def forward(self, x):
        s1 = self.stage0(x)
        x  = self.maxpool(s1)
        s2 = self.layer1(x)
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        return s5, [s1, s2, s3, s4, s5]

class UNet(nn.Module):

    def __init__(self, encoder_name="vgg16", num_classes=11):
        super().__init__()

        if encoder_name == "vgg16":
            self.encoder = VGG16Encoder()
        elif encoder_name == "resnet34":
            self.encoder = ResNet34Encoder()
        else:
            raise ValueError(f"encoder must be 'vgg16' or 'resnet34', got '{encoder_name}'")

        sc = self.encoder.skip_channels
        bn = self.encoder.bottleneck_channels

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(bn,  sc[4], 512),
            DecoderBlock(512, sc[3], 256),
            DecoderBlock(256, sc[2], 128),
            DecoderBlock(128, sc[1],  64),
            DecoderBlock( 64, sc[0],  64),
        ])

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_decoder()

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _initialize_decoder(self):
        for block in self.decoder_blocks:
            block.apply(self._init_weights)
        self.classifier.apply(self._init_weights)

    def forward(self, x):
        input_size = x.shape[2:]
        x, skips   = self.encoder(x)
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, skips[-(i + 1)])
        x = self.classifier(x)
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

def get_model(encoder_name="vgg16", num_classes=11):
    return UNet(encoder_name, num_classes)