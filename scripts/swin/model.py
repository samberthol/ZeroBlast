import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, backbone_name='swin_tiny_patch4_window7_224', pretrained=True, drop_path_rate=0.0):
        super(SwinUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # 1. Encoder (Swin Transformer from timm)
        self.encoder = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True,
            in_chans=n_channels,
            drop_path_rate=drop_path_rate
        )
        
        feature_info = self.encoder.feature_info.channels()
        encoder_channels = feature_info
        
        # 2. Decoder
        self.up1 = Up(encoder_channels[-1], encoder_channels[-2]) 
        self.up2 = Up(encoder_channels[-2], encoder_channels[-3]) 
        self.up3 = Up(encoder_channels[-3], encoder_channels[-4]) 
        
        self.up4 = DoubleConv(encoder_channels[-4], 64) 
        
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )
        
        # Auxiliary heads for deeper supervision
        self.out_head1 = nn.Conv2d(encoder_channels[-2], n_classes, 1) # 1/16 res
        self.out_head2 = nn.Conv2d(encoder_channels[-3], n_classes, 1) # 1/8 res

    def forward(self, x):
        features = self.encoder(x)
        
        normalized_features = []
        for feat in features:
            if feat.ndim == 4:
                 feat = feat.permute(0, 3, 1, 2).contiguous()
            normalized_features.append(feat)
            
        features = normalized_features
        x0, x1, x2, x3 = features[0], features[1], features[2], features[3]
        
        x_up1 = self.up1(x3, x2)
        x_up2 = self.up2(x_up1, x1)
        x_up3 = self.up3(x_up2, x0)
        x_up4 = self.up4(x_up3) 
        
        logits = self.final_up(x_up4)
        
        aux1 = self.out_head1(x_up1)
        aux2 = self.out_head2(x_up2)
        
        return logits, aux2, aux1

    @torch.inference_mode()
    def predict(self, x):
        logits, _, _ = self.forward(x)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels + out_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
