import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class HRNetSegmentation(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, backbone_name='hrnet_w18', pretrained=True):
        super(HRNetSegmentation, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder: HRNet from timm
        # HRNet is unique because it maintains high-resolution representations throughout.
        # 1. Encoder
        self.encoder = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True,
            in_chans=n_channels
        )
        
        # 2. Dynamic Channel Detection
        # We do a dummy forward pass to see exactly what shapes the encoder returns.
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, 256, 256)
            dummy_features = self.encoder(dummy_input)
            self.encoder_channels = [f.shape[1] for f in dummy_features]
            self.total_concat_channels = sum(self.encoder_channels)
            import logging
            logging.info(f"HRNet Dynamic Channels: {self.encoder_channels} (Total: {self.total_concat_channels})")
        
        # 3. Segmentation Head
        self.head = nn.Sequential(
            nn.Conv2d(self.total_concat_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )
        
        # Auxiliary heads for deep supervision
        self.aux1 = nn.Conv2d(self.encoder_channels[min(1, len(self.encoder_channels)-1)], n_classes, 1)
        self.aux2 = nn.Conv2d(self.encoder_channels[min(2, len(self.encoder_channels)-1)], n_classes, 1)

    def forward(self, x):
        features = self.encoder(x) # [1/4, 1/8, 1/16, 1/32]
        
        # Upsample all to 1/4 resolution
        target_size = features[0].shape[2:]
        up_features = [features[0]]
        for i in range(1, len(features)):
            up_features.append(F.interpolate(features[i], size=target_size, mode='bilinear', align_corners=True))
            
        combined = torch.cat(up_features, dim=1)
        logits = self.head(combined)
        
        # Final upsample to match input 
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Auxiliary outputs for supervision
        aux1 = F.interpolate(self.aux1(features[1]), size=x.shape[2:], mode='bilinear', align_corners=True)
        aux2 = F.interpolate(self.aux2(features[2]), size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return logits, aux1, aux2

    @torch.inference_mode()
    def predict(self, x):
        logits, _, _ = self.forward(x)
        return logits
