import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerWrapper(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, model_name="nvidia/segformer-b0-finetuned-ade-512-512"):
        super(SegFormerWrapper, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # We use the config to handle a single channel input
        # Note: SegFormer usually expects 3 channels. We might need to stack or adjust the first layer.
        self.config = SegformerConfig.from_pretrained(model_name)
        self.config.num_labels = n_classes
        self.config.num_channels = n_channels
        
        # Initializing for single channel
        self.model = SegformerForSemanticSegmentation(self.config)

    def forward(self, x):
        # x: [B, 1, H, W]
        outputs = self.model(x)
        logits = outputs.logits # [B, 1, H/4, W/4]
        
        # Upsample to input resolution
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        return logits, None, None # No aux heads for now in this wrapper

    @torch.inference_mode()
    def predict(self, x):
        logits, _, _ = self.forward(x)
        return logits
