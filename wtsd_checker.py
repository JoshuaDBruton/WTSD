import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from experiments.dataset.wtsd_dataset import WTSDDataset as DataSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os


TRAIN_PATH = "/home/joshua/Desktop/data/WTSD/train"
VAL_PATH = "/home/joshua/Desktop/data/WTSD/validation"
TEST_PATH = "/home/joshua/Desktop/data/WTSD-Test"

INPUT_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TARGET_TRANSFORM = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize((128, 128), interpolation=InterpolationMode.NEAREST),
    # transforms.ToTensor(),
])

INV_NORMALISE = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)


def check_wtsd():
    train_data = DataSet(root_dir=TRAIN_PATH, transform=INPUT_TRANSFORM, target_transform=TARGET_TRANSFORM,
                         concat_coords=False)
    val_data = DataSet(root_dir=VAL_PATH, transform=INPUT_TRANSFORM, target_transform=TARGET_TRANSFORM,
                       concat_coords=False)
    test_data = DataSet(root_dir=VAL_PATH, transform=INPUT_TRANSFORM, target_transform=TARGET_TRANSFORM,
                        concat_coords=False)

    # Extracting 4 random samples from the Datasets to display
    train_data, _ = torch.utils.data.random_split(train_data, [4, len(train_data) - 4])
    val_data, _ = torch.utils.data.random_split(val_data, [4, len(val_data) - 4])
    test_data, _ = torch.utils.data.random_split(test_data, [4, len(test_data)-4])

    # Pytorch Loaders, easily usable for training, can see example at https://github.com/JoshuaDBruton/TSC
    train_loader = DataLoader(train_data, shuffle=False, num_workers=4, batch_size=1)
    val_loader = DataLoader(val_data, shuffle=False, num_workers=4, batch_size=1)
    test_loader = DataLoader(test_data, shuffle=False, num_workers=4, batch_size=1)

    _, ax = plt.subplots(4, 2)
    for a in ax:
        for x in a:
            x.axis('off')
    for i, (x, y) in enumerate(train_loader):
        ax[i][0].imshow(INV_NORMALISE(x[0][:3, :, :]).permute(1, 2, 0).detach().numpy())
        ax[i][1].imshow(y[0][0].detach().numpy())
    plt.show()


if __name__ == "__main__":
    check_wtsd()
