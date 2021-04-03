import time
from ThreeDMatch.Train.dataset import ThreeDMatchDataset
import torch


def get_dataloader(root, split, batch_size=1, num_workers=4, shuffle=True, drop_last=True):
    dataset = ThreeDMatchDataset(
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
        num_workers=num_workers,
        drop_last=drop_last
    )

    return dataloader


if __name__ == '__main__':
    dataset = 'sun3d'
    dataroot = "/data/3DMatch/whole"
    trainloader = get_dataloader(dataroot, split='test', batch_size=32)
    start_time = time.time()
    print(f"Totally {len(trainloader)} iter.")
    for iter, (patches, ids) in enumerate(trainloader):
        if iter % 100 == 0:
            print(f"Iter {iter}: {time.time() - start_time} s")
    print(f"On the fly: {time.time() - start_time}")
