import torch
import torch.nn.functional as F
import time, os
import numpy as np
from loss.desc_loss import ContrastiveLoss
from tensorboardX import SummaryWriter


class Trainer(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.num_points_per_patch = args.num_points_per_patch
        self.batch_size = args.batch_size
        self.dataset = args.dataset
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.gpu_mode = args.gpu_mode
        self.verbose = args.verbose
        self.freeze_epoch = args.freeze_epoch

        self.rad_n = args.rad_n
        self.azi_n = args.azi_n
        self.ele_n = args.ele_n
        self.des_r = args.des_r
        self.voxel_r = args.voxel_r
        self.voxel_sample = args.voxel_sample

        self.model = args.model
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.evaluate_interval = args.evaluate_interval
        self.writer = SummaryWriter(log_dir=args.tboard_dir)

        self.train_loader = args.train_loader
        self.val_loader = args.val_loader

        self.desc_loss = ContrastiveLoss()

        if self.gpu_mode:
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=[0])

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

    def train(self):
        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }
        best_loss = 1000000000
        print('training start!!')
        start_time = time.time()

        self.model.train()
        freeze_sign = 1
        for epoch in range(self.epoch):

            self.train_epoch(epoch)

            if epoch % self.evaluate_interval == 0 or epoch == 0:
                res = self.evaluate(epoch + 1)
                print(f'Evaluation: Epoch {epoch}: Loss {res["loss"]}')

                if res['loss'] < best_loss:
                    best_loss = res['loss']
                    self._snapshot('best')
                if self.writer:
                    self.writer.add_scalar('Loss', res['loss'], epoch)

            if epoch % self.scheduler_interval == 0:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                print('update detector learning rate: %f -> %f' % (old_lr, new_lr))

            if self.writer:
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)
                self.writer.add_scalar('Train Loss', self.train_hist['loss'][-1], epoch)

                # finish all epoch
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        loss_buf = []
        num_batch = int(len(self.train_loader.dataset) / self.batch_size)
        for iter, (anc_local_patch, pos_local_patch, rotate, shift) in enumerate(self.train_loader):

            B = anc_local_patch.shape[0]
            anc_local_patch = anc_local_patch.float()
            pos_local_patch = pos_local_patch.float()
            rotate = rotate.float()
            shift = shift.float()

            if self.gpu_mode:
                anc_local_patch = anc_local_patch.cuda()
                pos_local_patch = pos_local_patch.cuda()

            # forward
            self.optimizer.zero_grad()
            a_desc = self.model(anc_local_patch)
            p_desc = self.model(pos_local_patch)
            anc_desc = F.normalize(a_desc.view(B, -1), p=2, dim=1)
            pos_desc = F.normalize(p_desc.view(B, -1), p=2, dim=1)

            # calculate the contrastive loss
            des_loss, accuracy = self.desc_loss(anc_desc, pos_desc)
            loss = des_loss

            # backward
            loss.backward()
            self.optimizer.step()
            loss_buf.append(float(loss))

            if iter % self.snapshot_interval == 0:
                self._snapshot(f'{epoch}_{iter + 1}')

            if iter % 200 == 0 and self.verbose:
                iter_time = time.time() - epoch_start_time
                print(f"Epoch: {epoch} [{iter:4d}/{num_batch}] loss: {loss:.2f} time: {iter_time:.2f}s")
                print(f"Epoch: {epoch} [{iter:4d}/{num_batch}] des loss: {des_loss:.2f} time: {iter_time:.2f}s")
                print(f"Accuracy: {accuracy.item():.4f}\n")
            del loss
            del anc_local_patch
            del pos_local_patch
        # finish one epoch
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch}: Loss {np.mean(loss_buf)}, time {epoch_time:.4f}s')

        del loss_buf

    def evaluate(self):
        self.model.eval()
        loss_buf = []
        with torch.no_grad():
            for iter, (anc_local_patch, pos_local_patch, rotate, shift) in enumerate(self.val_loader):

                B = anc_local_patch.shape[0]
                anc_local_patch = anc_local_patch.float()
                pos_local_patch = pos_local_patch.float()
                rotate = rotate.float()
                shift = shift.float()

                if self.gpu_mode:
                    anc_local_patch = anc_local_patch.cuda()
                    pos_local_patch = pos_local_patch.cuda()

                # forward
                a_des = self.model(anc_local_patch)
                p_des = self.model(pos_local_patch)
                anc_des = F.normalize(a_des.view(B, -1), p=2, dim=1)
                pos_des = F.normalize(p_des.view(B, -1), p=2, dim=1)

                # calculate the contrastive loss
                des_loss, accuracy = self.desc_loss(anc_des, pos_des)
                loss = des_loss
                loss_buf.append(float(loss))

                del loss
                del anc_local_patch
                del pos_local_patch

        self.model.train()

        res = {
            'loss': np.mean(loss_buf)
        }
        del loss_buf
        return res

    def _snapshot(self, epoch):
        save_dir = os.path.join(self.save_dir, self.dataset)
        torch.save(self.model.state_dict(), save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")

    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"Load model from {pretrain}.pkl")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
