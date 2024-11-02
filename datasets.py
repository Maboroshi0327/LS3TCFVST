import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class DAVIS(Dataset):
    def __init__(self, mode="train", resolution=256):
        super().__init__()
        self.resolution = resolution
        self.train_dir = "./Datasets/DAVIS/ImageSets/2017/train.txt"
        self.test_dir = "./Datasets/DAVIS/ImageSets/2017/val.txt"
        self.image_path = "./Datasets/DAVIS/JPEGImages/Full-Resolution/"

        # 讀取圖片路徑
        self.image_paths = []
        if mode == "train":
            with open(self.train_dir, "r") as f:
                for line in f:
                    self.image_paths.append(self.image_path + line.strip())
        elif mode == "test":
            with open(self.test_dir, "r") as f:
                for line in f:
                    self.image_paths.append(self.image_path + line.strip())
        else:
            raise ValueError("Mode should be either 'train' or 'test'.")

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
        resize_transform = transforms.Resize((self.resolution, self.resolution))

        # 將 PIL 圖像轉換為張量
        images = [transform(resize_transform(image)) for image in images]

        input = torch.stack(images[:8])
        input = input.reshape(3, 8, input.shape[2], input.shape[3])
        target = images[8]

        return input, target


class ILSVRC2015_VID(Dataset):
    def __init__(self, mode="train", resolution=256):
        super().__init__()
        self.resolution = resolution
        if mode == "train":
            txt_path = "./DatasetTXT/train.txt"
        elif mode == "test":
            txt_path = "./DatasetTXT/test.txt"
        elif mode == "val":
            txt_path = "./DatasetTXT/val.txt"
        else:
            raise ValueError("Mode should be 'train', 'test' or 'val'.")

        # 構建數據集
        self.each_data = []
        self.lens = 0
        with open(txt_path, "r") as f:
            for line in f:
                path = line.strip().split(" ")[0]
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
        resize_transform = transforms.Resize((self.resolution, self.resolution))

        # 將 PIL 圖像轉換為張量
        images = [transform(resize_transform(image)) for image in images]

        input = torch.stack(images[:8])
        input = input.reshape(3, 8, input.shape[2], input.shape[3])
        target = images[8]

        return input, target


def list_files(directory):
    files = [f.path for f in os.scandir(directory) if f.is_file()]
    return sorted(files)


def list_subfolders(directory):
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    return sorted(subfolders)


def make_ILSVRC2015_txt(mode="train"):
    if mode == "train":
        train_sub = list_subfolders("./Datasets/ILSVRC2015/Data/VID/train")
        with open("./DatasetTXT/train.txt", "w") as f:
            for sub1 in train_sub:
                train_sub2 = list_subfolders(sub1)
                for sub2 in train_sub2:
                    files = list_files(sub2)
                    f.write(sub2 + " " + str(len(files)) + "\n")

    elif mode == "test":
        test_sub = list_subfolders("./Datasets/ILSVRC2015/Data/VID/test")
        with open("./DatasetTXT/test.txt", "w") as f:
            for sub in test_sub:
                files = list_files(sub)
                f.write(sub + " " + str(len(files)) + "\n")

    elif mode == "val":
        val_sub = list_subfolders("./Datasets/ILSVRC2015/Data/VID/val")
        with open("./DatasetTXT/val.txt", "w") as f:
            for sub in val_sub:
                files = list_files(sub)
                f.write(sub + " " + str(len(files)) + "\n")


if __name__ == "__main__":
    import time

    start_time = time.time()

    dataset = ILSVRC2015_VID(mode="train")
    print(len(dataset))
    print("Done!")

    print(f"Total time: {time.time() - start_time} seconds")

    # make_ILSVRC2015_txt(mode="train")
    # make_ILSVRC2015_txt(mode="test")
    # make_ILSVRC2015_txt(mode="val")
