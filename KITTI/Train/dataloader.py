import time
from KITTI.Train.dataset import KITTIDataset
import torch


def get_dataloader(root, split, batch_size=1, num_workers=0, shuffle=True, drop_last=True):
    dataset = KITTIDataset(
        root=root,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )
    dataset.initial()
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=drop_last
    )

    return dataloader
