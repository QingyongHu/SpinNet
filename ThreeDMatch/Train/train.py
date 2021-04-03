import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import shutil
import sys

sys.path.append('../../')
from ThreeDMatch.Train.dataloader import get_dataloader
from ThreeDMatch.Train.trainer import Trainer
from network.SpinNet import Descriptor_Net
from torch import optim


class Args(object):
    def __init__(self):
        self.experiment_id = "Proposal" + time.strftime('%m%d%H%M')
        snapshot_root = 'snapshot/%s' % self.experiment_id
        tensorboard_root = 'tensorboard/%s' % self.experiment_id
        os.makedirs(snapshot_root, exist_ok=True)
        os.makedirs(tensorboard_root, exist_ok=True)
        shutil.copy2(os.path.join('', 'train.py'), os.path.join(snapshot_root, 'train.py'))
        shutil.copy2(os.path.join('', 'trainer.py'), os.path.join(snapshot_root, 'trainer.py'))
        shutil.copy2(os.path.join('', '../../network/SpinNet.py'), os.path.join(snapshot_root, 'SpinNet.py'))
        shutil.copy2(os.path.join('', '../../network/ThreeDCCN.py'), os.path.join(snapshot_root, 'ThreeDCCN.py'))
        shutil.copy2(os.path.join('', '../../loss/desc_loss.py'), os.path.join(snapshot_root, 'loss.py'))
        self.epoch = 20
        self.batch_size = 76
        self.rad_n = 9
        self.azi_n = 80
        self.ele_n = 40
        self.des_r = 0.30
        self.voxel_r = 0.04
        self.voxel_sample = 30

        self.dataset = '3DMatch'
        self.data_train_dir = '../../data/3DMatch/patches'
        self.data_val_dir = '../../data/3DMatch/patches'

        self.gpu_mode = True
        self.verbose = True
        self.freeze_epoch = 5

        # model & optimizer
        self.model = Descriptor_Net(self.des_r, self.rad_n, self.azi_n, self.ele_n,
                                    self.voxel_r, self.voxel_sample, self.dataset)
        self.pretrain = ''
        self.parameter = self.model.get_parameter()
        self.optimizer = optim.Adam(self.parameter, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        self.scheduler_interval = 5

        # dataloader
        self.train_loader = get_dataloader(root=self.data_train_dir,
                                           batch_size=self.batch_size,
                                           split='train',
                                           shuffle=True,
                                           num_workers=0,  # if the dataset is offline generated, must 0
                                           )
        self.val_loader = get_dataloader(root=self.data_val_dir,
                                         batch_size=self.batch_size,
                                         split='val',
                                         shuffle=False,
                                         num_workers=0,  # if the dataset is offline generated, must 0
                                         )

        print("Training set size:", self.train_loader.dataset.__len__())
        print("Validate set size:", self.val_loader.dataset.__len__())

        # snapshot
        self.snapshot_interval = int(self.train_loader.dataset.__len__() / self.batch_size / 2)
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.result_dir = os.path.join(snapshot_root, 'results/')
        self.tboard_dir = tensorboard_root

        # evaluate
        self.evaluate_interval = 1

        self.check_args()

    def check_args(self):
        """checking arguments"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        if not os.path.exists(self.tboard_dir):
            os.makedirs(self.tboard_dir)
        return self


if __name__ == '__main__':
    args = Args()
    trainer = Trainer(args)
    trainer.train()
