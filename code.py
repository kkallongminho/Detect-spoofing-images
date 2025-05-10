import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import torch.optim as optim

# 🔧 데이터 전처리 및 로드
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ImageFolder("./train", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ✅ EfficientNet-B0 불러오기
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
model = model.to('cuda')

# 🎯 손실함수 & 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# 🔁 학습 루프
for epoch in range(5):
    model.train()
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# 💾 모델 저장
torch.save(model.state_dict(), "efficientnet_b0_final.pth")
