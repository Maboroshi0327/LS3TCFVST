import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import time

from Network import temporalBranch


class DAVIS(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.train_dir = "./DAVIS/ImageSets/2017/train.txt"
        self.test_dir = "./DAVIS/ImageSets/2017/val.txt"
        self.image_path = "./DAVIS/JPEGImages/Full-Resolution/"

        # 讀取圖片路徑
        self.image_paths = []
        if train:
            with open(self.train_dir, "r") as f:
                for line in f:
                    self.image_paths.append(self.image_path + line.strip())
        else:
            with open(self.test_dir, "r") as f:
                for line in f:
                    self.image_paths.append(self.image_path + line.strip())

        # 構建數據集
        self.each_data = []
        self.lens = 0
        for path in self.image_paths:
            images = sorted(os.listdir(path))
            lens = len(images)
            lens_8 = lens - 9 + 1
            self.lens += lens_8
            for i in range(lens_8):
                self.each_data.append(
                    [os.path.join(path, images[i + j]) for j in range(9)]
                )

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
        img_path = self.each_data[idx]
        images = [Image.open(img_path[i]).convert("RGB") for i in range(9)]

        # 定義轉換
        transform = transforms.ToTensor()

        # 調整圖像大小
        resize_transform = transforms.Resize((512, 512))

        # 將 PIL 圖像轉換為張量
        images = [transform(resize_transform(image)) for image in images]

        input = torch.stack(images[:8])
        input = input.reshape(3, 8, input.shape[2], input.shape[3])
        target = images[8]

        return input, target


# 準備數據集和數據加載器
train_dataset = DAVIS(train=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=12,
    prefetch_factor=2,
)

# 初始化模型
model = temporalBranch.Model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 10
for epoch in range(num_epochs):
    epoch_start_time = time.time()

    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向傳播和優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 統計損失
        running_loss += loss.item()

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}, Duration: {epoch_duration:.2f} s"
    )

print("訓練完成")
