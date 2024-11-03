import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from Network import temporalBranch, resnet
from datasets import DAVIS, ILSVRC2015_VID


# 準備數據集和數據加載器
train_dataset = ILSVRC2015_VID(mode="train", resolution=256)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=False,
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
optimizer_G = optim.Adam(
    generator.parameters(), lr=0.00001, betas=(0.5, 0.999), weight_decay=0.0002
)
optimizer_D = optim.Adam(
    discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0002
)

# 訓練模型
num_epochs = 10
total_start_time = time.time()  # 記錄總運行時間的開始時間

for epoch in range(num_epochs):
    epoch_start_time = time.time()  # 記錄每個 epoch 的開始時間

    # 初始化進度條
    batch_iterator = tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
    )

    for i, (input, target) in enumerate(batch_iterator):
        batch_size = input.size(0)

        # 訓練判別器
        discriminator.zero_grad()

        real_images = target.to(device)
        fake_images = generator(input.to(device))
        outputs_real = discriminator(real_images)
        outputs_fake = discriminator(fake_images.detach())
        d_loss = criterion_D(outputs_real, outputs_fake)
        d_loss.backward()
        optimizer_D.step()

        # 訓練生成器
        generator.zero_grad()

        real_images = target.to(device)
        fake_images = generator(input.to(device))
        outputs_d = discriminator(fake_images)
        g_loss = criterion_G(fake_images, real_images, outputs_d, lambda_adv=0.1)
        g_loss.backward()
        optimizer_G.step()

        # 更新進度條後綴資訊
        batch_iterator.set_postfix(G_Loss=g_loss.item(), D_Loss=d_loss.item())

        # 每 1000 個 batch 輸出一次訓練結果
        if (i + 1) % 1000 == 0:
            # 儲存生成的圖片
            img = fake_images[0].detach().cpu()
            img = transforms.ToPILImage()(img)
            img.save(f"./Result_images/epoch_{epoch+1}_{i+1}_fake_image.jpg")
            img = real_images[0].detach().cpu()
            img = transforms.ToPILImage()(img)
            img.save(f"./Result_images/epoch_{epoch+1}_{i+1}_real_image.jpg")

            # 儲存模型
            # torch.save(generator.state_dict(), f'./Save_models/generator_epoch_{epoch+1}_batch_{i+1}.pth')
            # torch.save(discriminator.state_dict(), f'./Save_models/discriminator_epoch_{epoch+1}_batch_{i+1}.pth')

    # 儲存生成的圖片
    img = fake_images[0].detach().cpu()
    img = transforms.ToPILImage()(img)
    img.save(f"./Result_images/epoch_{epoch+1}_end_fake_image.jpg")
    img = real_images[0].detach().cpu()
    img = transforms.ToPILImage()(img)
    img.save(f"./Result_images/epoch_{epoch+1}_end_real_image.jpg")

    # 儲存模型
    torch.save(
        generator.state_dict(), f"./Save_models/generator_epoch_{epoch+1}_end.pth"
    )
    torch.save(
        discriminator.state_dict(),
        f"./Save_models/discriminator_epoch_{epoch+1}_end.pth",
    )

    epoch_end_time = time.time()  # 記錄每個 epoch 的結束時間
    epoch_duration = epoch_end_time - epoch_start_time  # 計算每個 epoch 的運行時間
    print(f"Epoch [{epoch+1}/{num_epochs}], Duration: {epoch_duration:.2f} s")

total_end_time = time.time()  # 記錄總運行時間的結束時間
total_duration = total_end_time - total_start_time  # 計算總運行時間
print(f"訓練完成，總運行時間: {total_duration:.2f} seconds")
