from torch.utils.data import Dataset
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


class WTSDDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None, concat_coords=False):
        """
        DataSet for representing pairs of query and reference images (saved as .jpg and .npy files) with
        full pixel-level labels
        Two folders, inputs, for the query images, and targets, for the reference images, are required to be present in
        the top-level root directory
        :param root_dir: the root directory, a folder containing the folders inputs/ and targets/
        :param transform: A transform to be applied to the inputs
        :param target_transform: A transform to be applied to the targets/references (perhaps down-scaling)
        :param concat_coords: Whether or not to use Optional Translational Equivariance (OTE), more details
        at https://github.com/JoshuaDBruton/TSC
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.concat_coords = concat_coords

        self.inputs_dir = os.path.join(self.root_dir, "inputs")
        self.targets_dir = os.path.join(self.root_dir, "targets")

        self.n = len(os.listdir(self.inputs_dir))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = np.array(plt.imread(os.path.join(self.inputs_dir, str(idx)+".jpg")))
        y = np.load(os.path.join(self.targets_dir, str(idx) + ".npy"))

        assert (len(x.shape) == 3), "Input x does not have 3 dimensions, shape is {}".format(x.shape)
        assert (x.shape[2] == 3), "Input x is not RGB, does not have 3 channels; x has {} channels".format(x.shape[2])
        assert (len(y.shape) == 2), "Shape of y is {}, not 2-dim grayscale".format(y.shape)
        assert (x.dtype == np.uint8), "Type of x is not uint8, it is {}".format(x.dtype)
        assert (y.dtype == np.uint8), "Type of y is not uint8, it is {}".format(y.dtype)

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = torch.as_tensor(np.array(y), dtype=torch.int64).unsqueeze(0)
            y = self.target_transform(y)

        if self.concat_coords:
            gx, gy = torch.meshgrid(torch.arange(0, x.shape[1]), torch.arange(0, x.shape[2]))
            gx = gx/torch.max(gx)
            gy = gy/torch.max(gy)
            gx = gx.unsqueeze(0)
            gy = gy.unsqueeze(0)
            x = torch.cat((x, gx, gy), 0)

        sample = (x, y)

        return sample
