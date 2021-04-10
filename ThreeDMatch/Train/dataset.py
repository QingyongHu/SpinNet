import torch.utils.data as Data
import os
import random
import glob
import pickle
import open3d as o3d
import numpy as np


class ThreeDMatchDataset(Data.Dataset):
    def __init__(self, root, split, batch_size, shuffle, drop_last):
        """
        Create ThreeDMatchDataset to read multiple training files
        Args:
            root: the path to the dataset file
            shuffle: whether the data need to shuffle
        """
        self.patches_path = os.path.join(root, split)
        self.split = split
        # Get name of all training pkl files
        training_data_files = glob.glob(self.patches_path + '/*.pkl')
        ids = [file.split("/")[-1] for file in training_data_files]
        ids = sorted(ids, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        ids = [file for file in ids if file.split("_")[1] == 'anc&pos']
        self.training_data_files = ids
        # Get info of training files
        self.per_num_patch = int(training_data_files[0].split("/")[-1].split("_")[2])
        self.dataset_len = int(ids[-1].split("_")[-1].split(".")[0]) * self.per_num_patch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        # Record the loaded i-th training file
        self.num_file = 0
        # load poses for each type of patches
        self.per_patch_points = int(self.training_data_files[-1].split("_")[3])
        self.num_framents = int(self.training_data_files[-1].split("_")[4].split(".")[0])
        with open(os.path.join(root,
                               f'{self.split}/{self.split}_poses_{self.per_num_patch}_{self.per_patch_points}_{self.num_framents}.pkl'),
                  'rb') as file:
            self.poses = pickle.load(file)
            print(
                f"load training poses {os.path.join(root, f'{self.split}_poses_{self.per_num_patch}_{self.per_patch_points}_{self.num_framents}.pkl')}")
        self.cur_pose_ind = 0

    def initial(self):
        with open(os.path.join(self.patches_path, self.training_data_files[self.num_file]), 'rb') as file:
            self.patches = pickle.load(file)
            print(f"load training files {os.path.join(self.patches_path, self.training_data_files[self.num_file])}")

        next_pose_ind = int(self.training_data_files[self.num_file].split(".")[0].split("_")[-1])
        poses = self.poses[self.cur_pose_ind:next_pose_ind]
        for i in range(len(self.patches)):
            ind = int(np.floor(i / self.per_num_patch))
            pose = np.concatenate([poses[ind][:3, :3].reshape(9), poses[ind][:3, 3]]).reshape(2, 6)
            self.patches[i] = np.concatenate([pose, self.patches[i]])
        self.cur_pose_ind = next_pose_ind

        self.current_patches_num = len(self.patches)
        self.index = list(range(self.current_patches_num))
        if self.shuffle:
            random.shuffle(self.patches)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        idx = self.index[0]
        patches = self.patches[idx]
        self.index = self.index[1:]
        self.current_patches_num -= 1

        if self.drop_last:
            if self.current_patches_num <= (len(self.patches) % self.batch_size):  # reach the end of training file
                self.num_file = self.num_file + 1
                if self.num_file < len(self.training_data_files):
                    remain_patches = [self.patches[i] for i in self.index]  # the remained training patches
                    with open(os.path.join(self.patches_path, self.training_data_files[self.num_file]), 'rb') as file:
                        self.patches = pickle.load(file)
                        print(
                            f"load training files {os.path.join(self.patches_path, self.training_data_files[self.num_file])}")
                    next_pose_ind = int(self.training_data_files[self.num_file].split(".")[0].split("_")[-1])
                    poses = self.poses[self.cur_pose_ind:next_pose_ind]
                    for i in range(len(self.patches)):
                        ind = int(np.floor(i / self.per_num_patch))
                        pose = np.concatenate([poses[ind][:3, :3].reshape(9), poses[ind][:3, 3]]).reshape(2, 6)
                        self.patches[i] = np.concatenate([pose, self.patches[i]])
                    self.cur_pose_ind = next_pose_ind
                    self.patches += remain_patches  # add the remained patches to compose a set of new patches
                    self.current_patches_num = len(self.patches)
                    self.index = list(range(self.current_patches_num))
                    if self.shuffle:
                        random.shuffle(self.patches)
                else:
                    self.num_file = 0
                    self.cur_pose_ind = 0
                    self.initial()
        else:
            if self.current_patches_num <= 0:
                self.num_file = self.num_file + 1
                if self.num_file < len(self.training_data_files):
                    with open(os.path.join(self.patches_path, self.training_data_files[self.num_file]), 'rb') as file:
                        self.patches = pickle.load(file)
                        print(
                            f"load training files {os.path.join(self.patches_path, self.training_data_files[self.num_file])}")
                    next_pose_ind = int(self.training_data_files[self.num_file].split(".")[0].split("_")[-1])
                    poses = self.poses[self.cur_pose_ind:next_pose_ind]
                    for i in range(len(self.patches)):
                        ind = int(np.floor(i / self.per_num_patch))
                        pose = np.concatenate([poses[ind][:3, :3].reshape(9), poses[ind][:3, 3]]).reshape(2, 6)
                        self.patches[i] = np.concatenate([pose, self.patches[i]])
                    self.cur_pose_ind = next_pose_ind
                    self.current_patches_num = len(self.patches)
                    self.index = list(range(self.current_patches_num))
                    if self.shuffle:
                        random.shuffle(self.patches)
                else:
                    self.num_file = 0
                    self.cur_pose_ind = 0
                    self.initial()

        anc_local_patch = patches[2:, :3]
        pos_local_patch = patches[2:, 3:]
        rotate = patches[:2, :].reshape(12)[:9].reshape(3, 3)
        shift = patches[:2, :].reshape(12)[9:]

        # np.random.shuffle(anc_local_patch)
        # np.random.shuffle(pos_local_patch)

        return anc_local_patch, pos_local_patch, rotate, shift


if __name__ == "__main__":
    data_root = "../data/3DMatch_patches/"
    batch_size = 48
    epoch = 1
    train_dataset = ThreeDMatchDataset(root=data_root, split='train', batch_size=batch_size, shuffle=True,
                                       drop_last=True)
    train_dataset.initial()
    for _ in range(epoch):
        train_iter = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=True)
        for iter, (anc_local_patch, pos_local_patch, rotate, shift) in enumerate(train_iter):
            B = anc_local_patch.shape[0]
