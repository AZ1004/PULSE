import torch
import torch.nn as nn
from torchvision import models

class SimCLR(nn.Module):
    def __init__(self, base_model="resnet50", out_dim=128):
        super(SimCLR, self).__init__()
        
        # 1. The Backbone (Encoder)
        if base_model == "resnet50":
            self.backbone = models.resnet50(weights=None)
            dim_mlp = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity() # Remove classification head
        elif base_model == "vit_b_16":
            self.backbone = models.vit_b_16(weights=None)
            dim_mlp = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
        else:
            raise ValueError("Model not supported yet!")

        # 2. The Projector (MLP Head)
        # Industry standard: 2-layer MLP with ReLU
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)     # Representation vector
        z = self.projector(h)    # Projection vector for loss
        return h, z
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimCLR(base_model="resnet50").to(device)
    
    # Simulate a batch of 2 augmented images (Views A and B)
    x = torch.randn(2, 3, 224, 224).to(device)
    h, z = model(x)
    
    print(f"Backbone features (h) shape: {h.shape}") # Should be [2, 2048] for ResNet50
    print(f"Projected vector (z) shape: {z.shape}")  # Should be [2, 128]
    print("Forward pass successful on Omen GPU!")