import torch
import torch.nn as nn
from transformers import ViTModel

class GestureViT(nn.Module):
    def __init__(self, num_classes=5):  # Adjust num_classes based on gestures
        super(GestureViT, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.fc = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        x = self.vit(x).last_hidden_state[:, 0, :]
        return self.fc(x)
