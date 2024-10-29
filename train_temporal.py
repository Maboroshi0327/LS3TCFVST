import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os

from Network import temporalBranch, resnet


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

# 初始化生成器和判別器
generator = resnet.Generator()
discriminator = temporalBranch.Discriminator()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# 定義損失函數和優化器
criterion_G = temporalBranch.GeneratorLoss()
criterion_D = temporalBranch.DiscriminatorLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 訓練模型
num_epochs = 10
total_start_time = time.time()  # 記錄總運行時間的開始時間

for epoch in range(num_epochs):
    epoch_start_time = time.time()  # 記錄每個 epoch 的開始時間

    for i, (input, target) in enumerate(train_loader):
        batch_size = input.size(0)

        # 每 20 次切換訓練模型
        if (i // 20) % 2 == 0:
            # 訓練判別器
            discriminator.zero_grad()

            real_images = target.to(device)
            fake_images = generator(input.to(device))
            outputs_real = discriminator(real_images)
            outputs_fake = discriminator(fake_images)
            d_loss = criterion_D(outputs_real, outputs_fake)
            d_loss.backward()
            optimizer_D.step()

        else:
            # 訓練生成器
            generator.zero_grad()

            real_images = target.to(device)
            fake_images = generator(input.to(device))
            outputs_d = discriminator(fake_images)
            g_loss = criterion_G(fake_images, real_images, outputs_d, lambda_adv=0.1)
            g_loss.backward()
            optimizer_G.step()

        if (i + 1) % 10 == 0:
            if (i // 20) % 2 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], D Loss: {d_loss.item()}"
                )
            else:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], G Loss: {g_loss.item()}"
                )

            img = fake_images[0].detach().cpu()
            img = transforms.ToPILImage()(img)
            img.save(f"./Result_images/{epoch+1}_{i+1}_fake_image.jpg")

            img = real_images[0].detach().cpu()
            img = transforms.ToPILImage()(img)
            img.save(f"./Result_images/{epoch+1}_{i+1}_real_image.jpg")

    epoch_end_time = time.time()  # 記錄每個 epoch 的結束時間
    epoch_duration = epoch_end_time - epoch_start_time  # 計算每個 epoch 的運行時間
    print(f"Epoch [{epoch+1}/{num_epochs}], Duration: {epoch_duration:.2f} s")

total_end_time = time.time()  # 記錄總運行時間的結束時間
total_duration = total_end_time - total_start_time  # 計算總運行時間
print(f"訓練完成，總運行時間: {total_duration:.2f} seconds")
