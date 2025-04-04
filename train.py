import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import GestureDataset
from models.vit_model import GestureViT

# Dummy data - Replace with real dataset
data = [[0.5] * 63] * 100  
labels = [0] * 50 + [1] * 50  

dataset = GestureDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureViT(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss {loss.item()}")

torch.save(model.state_dict(), "gesture_model.pth")
