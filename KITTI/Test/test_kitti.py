import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import glob
import time
import gc
import shutil
import pointnet2_ops.pointnet2_utils as pnt2
import copy
import importlib
import sys

sys.path.append('../../')
import script.common as cm

kitti_icp_cache = {}
kitti_cache = {}


class Timer(object):
    """A simple timer."""

    def __init__(self, binary_fn=None, init_val=0):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.binary_fn = binary_fn
        self.tmp = init_val

    def reset(self):
        self.total_time = 0
        self.calls = 0
        self.start_time = 0
        self.diff = 0

    @property
    def avg(self):
        return self.total_time / self.calls

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        if self.binary_fn:
            self.tmp = self.binary_fn(self.tmp, self.diff)
        if average:
            return self.avg
        else:
            return self.diff


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.sq_sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sq_sum += val ** 2 * n
        self.var = self.sq_sum / self.count - self.avg ** 2


def get_desc(descpath, filename):
    desc = np.load(os.path.join(descpath, filename + '.npy'))
    return desc


def get_keypts(keypts_path, filename):
    keypts = np.load(os.path.join(keypts_path, filename + '.npy'))
    return keypts


def make_open3d_feature(data, dim, npts):
    feature = o3d.pipelines.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.astype('d').transpose()
    return feature


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


class KITTI(object):
    DATA_FILES = {
        'train': 'train_kitti.txt',
        'val': 'val_kitti.txt',
        'test': 'test_kitti.txt'
    }
    """
    Given point cloud fragments and corresponding pose in '{root}'.
        1. Save the aligned point cloud pts in '{savepath}/3DMatch_{downsample}_points.pkl'
        2. Calculate the overlap ratio and save in '{savepath}/3DMatch_{downsample}_overlap.pkl'
        3. Save the ids of anchor keypoints and positive keypoints in '{savepath}/3DMatch_{downsample}_keypts.pkl'
    """

    def __init__(self, root, descpath, icp_path, split, model, num_points_per_patch, use_random_points):
        self.root = root
        self.descpath = descpath
        self.split = split
        self.num_points_per_patch = num_points_per_patch
        self.icp_path = icp_path
        self.use_random_points = use_random_points
        self.model = model
        if not os.path.exists(self.icp_path):
            os.makedirs(self.icp_path)

        # list: anc & pos
        self.patches = []
        self.pose = []
        # Initiate containers
        self.files = {'train': [], 'val': [], 'test': []}

        self.prepare_kitti_ply(split=self.split)

    def prepare_kitti_ply(self, split='train'):
        subset_names = open(self.DATA_FILES[split]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1))
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files[split].append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1
        # Remove problematic sequence
        for item in [
            (8, 15, 58),
        ]:
            if item in self.files[split]:
                self.files[split].pop(self.files[split].index(item))

        if split == 'train':
            self.num_train = len(self.files[split])
            print("Num_train", self.num_train)
        elif split == 'val':
            self.num_val = len(self.files[split])
            print("Num_val", self.num_val)
        elif split == 'test':
            self.num_test = len(self.files[split])
            print("Num_test", self.num_test)

        for idx in range(len(self.files[split])):
            drive = self.files[split][idx][0]
            t0, t1 = self.files[split][idx][1], self.files[split][idx][2]
            all_odometry = self.get_video_odometry(drive, [t0, t1])
            positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
            fname0 = self._get_velodyne_fn(drive, t0)
            fname1 = self._get_velodyne_fn(drive, t1)

            # XYZ and reflectance
            xyzr0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)
            xyzr1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)

            xyz0 = xyzr0[:, :3]
            xyz1 = xyzr1[:, :3]

            key = '%d_%d_%d' % (drive, t0, t1)
            filename = self.icp_path + '/' + key + '.npy'
            if key not in kitti_icp_cache:
                if not os.path.exists(filename):
                    M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                         @ np.linalg.inv(self.velo2cam)).T
                    xyz0_t = self.apply_transform(xyz0, M)
                    pcd0 = make_open3d_point_cloud(xyz0_t, [0.5, 0.5, 0.5])
                    pcd1 = make_open3d_point_cloud(xyz1, [0, 1, 0])
                    reg = o3d.pipelines.registration.registration_icp(pcd0, pcd1, 0.10, np.eye(4),
                                                            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                max_iteration=400))
                    pcd0.transform(reg.transformation)
                    M2 = M @ reg.transformation
                    # write to a file
                    np.save(filename, M2)
                else:
                    M2 = np.load(filename)
                kitti_icp_cache[key] = M2
            else:
                M2 = kitti_icp_cache[key]
            trans = M2
            # extract patches for anc&pos
            np.random.shuffle(xyz0)
            np.random.shuffle(xyz1)

            if is_rotate_dataset:
                # Add arbitrary rotation
                # rotate terminal frament with an arbitrary angle around the z-axis
                angles_3d = np.random.rand(3) * np.pi * 2
                R = cm.angles2rotation_matrix(angles_3d)
                T = np.identity(4)
                T[:3, :3] = R
                pcd1 = make_open3d_point_cloud(xyz1)
                pcd1.transform(T)
                xyz1 = np.array(pcd1.points)
                all_trans_matrix[key] = T

            if not os.path.exists(self.descpath + str(drive)):
                os.makedirs(self.descpath + str(drive))
            if self.use_random_points:
                num_keypts = 5000
                step_size = 50
                desc_len = 32
                model = self.model.cuda()
                # calc t0 descriptors
                desc_t0_path = os.path.join(self.descpath + str(drive), f"cloud_bin_" + str(t0) + f".desc.bin.npy")
                keypts_t0_path = os.path.join(self.descpath + str(drive), f"cloud_bin_" + str(t0) + f".keypts.npy")
                if not os.path.exists(desc_t0_path):
                    keypoints_id = np.random.choice(xyz0.shape[0], num_keypts)
                    keypts = xyz0[keypoints_id]
                    np.save(keypts_t0_path, keypts.astype(np.float32))
                    local_patches = self.select_patches(xyz0, keypts, vicinity=vicinity,
                                                        num_points_per_patch=self.num_points_per_patch)
                    B = local_patches.shape[0]
                    # cuda out of memry
                    desc_list = []
                    start_time = time.time()
                    iter_num = np.int(np.ceil(B / step_size))
                    for k in range(iter_num):
                        if k == iter_num - 1:
                            desc = model(local_patches[k * step_size:, :, :])
                        else:
                            desc = model(local_patches[k * step_size: (k + 1) * step_size, :, :])
                        desc_list.append(desc.view(desc.shape[0], desc_len).detach().cpu().numpy())
                        del desc
                    step_time = time.time() - start_time
                    print(f'Finish {B} descriptors spend {step_time:.4f}s')
                    desc = np.concatenate(desc_list, 0).reshape([B, desc_len])
                    np.save(desc_t0_path, desc.astype(np.float32))
                else:
                    print(f"{desc_t0_path} already exists.")

                # calc t1 descriptors
                desc_t1_path = os.path.join(self.descpath + str(drive), f"cloud_bin_" + str(t1) + f".desc.bin.npy")
                keypts_t1_path = os.path.join(self.descpath + str(drive), f"cloud_bin_" + str(t1) + f".keypts.npy")
                if not os.path.exists(desc_t1_path):
                    keypoints_id = np.random.choice(xyz1.shape[0], num_keypts)
                    keypts = xyz1[keypoints_id]
                    np.save(keypts_t1_path, keypts.astype(np.float32))
                    local_patches = self.select_patches(xyz1, keypts, vicinity=vicinity,
                                                        num_points_per_patch=self.num_points_per_patch)
                    B = local_patches.shape[0]
                    # calculate descriptors
                    desc_list = []
                    start_time = time.time()
                    iter_num = np.int(np.ceil(B / step_size))
                    for k in range(iter_num):
                        if k == iter_num - 1:
                            desc = model(local_patches[k * step_size:, :, :])
                        else:
                            desc = model(local_patches[k * step_size: (k + 1) * step_size, :, :])
                        desc_list.append(desc.view(desc.shape[0], desc_len).detach().cpu().numpy())
                        del desc
                    step_time = time.time() - start_time
                    print(f'Finish {B} descriptors spend {step_time:.4f}s')
                    desc = np.concatenate(desc_list, 0).reshape([B, desc_len])
                    np.save(desc_t1_path, desc.astype(np.float32))
                else:
                    print(f"{desc_t1_path} already exists.")
            else:
                num_keypts = 512

    def select_patches(self, pts, refer_pts, vicinity, num_points_per_patch=1024):
        gc.collect()
        pts = torch.FloatTensor(pts).cuda().unsqueeze(0)
        refer_pts = torch.FloatTensor(refer_pts).cuda().unsqueeze(0)
        group_idx = pnt2.ball_query(vicinity, num_points_per_patch, pts, refer_pts)
        pts_trans = pts.transpose(1, 2).contiguous()
        new_points = pnt2.grouping_operation(
            pts_trans, group_idx
        )
        new_points = new_points.permute([0, 2, 3, 1])
        mask = group_idx[:, :, 0].unsqueeze(2).repeat(1, 1, num_points_per_patch)
        mask = (group_idx == mask).float()
        mask[:, :, 0] = 0
        mask[:, :, num_points_per_patch - 1] = 1
        mask = mask.unsqueeze(3).repeat([1, 1, 1, 3])
        new_pts = refer_pts.unsqueeze(2).repeat([1, 1, num_points_per_patch, 1])
        local_patches = new_points * (1 - mask).float() + new_pts * mask.float()
        local_patches = local_patches.squeeze(0)
        del mask
        del new_points
        del group_idx
        del new_pts
        del pts
        del pts_trans

        return local_patches

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        data_path = self.root + '/poses/%02d.txt' % drive
        if data_path not in kitti_cache:
            kitti_cache[data_path] = np.genfromtxt(data_path)
        if return_all:
            return kitti_cache[data_path]
        else:
            return kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        T_w_cam0 = odometry.reshape(3, 4)
        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
        return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname


if __name__ == '__main__':
    is_rotate_dataset = False
    all_trans_matrix = {}
    experiment_id = time.strftime('%m%d%H%M')  # '11210201'#
    model_str = experiment_id
    reg_timer = Timer()
    success_meter, rte_meter, rre_meter = AverageMeter(), AverageMeter(), AverageMeter()
    ch = logging.StreamHandler(sys.stdout)
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

    # dynamically load the model from snapshot
    module_file_path = '../model.py'
    shutil.copy2(os.path.join('.', '../../network/SpinNet.py'), module_file_path)
    module_name = ''
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    vicinity = 2.0
    model = module.Descriptor_Net(vicinity, 9, 60, 30, 0.3, 30, 'KITTI')
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load('../../pre-trained_models/KITTI_best.pkl'))

    test_data = KITTI(root='../../data/KITTI/dataset',
                      descpath=f'SpinNet_desc_{model_str}/',
                      icp_path='../../data/KITTI/icp',
                      split='test',
                      model=model,
                      num_points_per_patch=2048,
                      use_random_points=True
                      )

    files = test_data.files[test_data.split]
    for idx in range(len(files)):
        drive = files[idx][0]
        t0, t1 = files[idx][1], files[idx][2]
        key = '%d_%d_%d' % (drive, t0, t1)
        filename = test_data.icp_path + '/' + key + '.npy'
        T_gth = kitti_icp_cache[key]
        if is_rotate_dataset:
            T_gth = np.matmul(all_trans_matrix[key], T_gth)

        descpath = os.path.join(test_data.descpath, str(drive))
        fname0 = test_data._get_velodyne_fn(drive, t0)
        fname1 = test_data._get_velodyne_fn(drive, t1)
        # XYZ and reflectance
        xyz0 = get_keypts(descpath, f"cloud_bin_" + str(t0) + f".keypts")
        xyz1 = get_keypts(descpath, f"cloud_bin_" + str(t1) + f".keypts")
        pcd0 = make_open3d_point_cloud(xyz0)
        pcd1 = make_open3d_point_cloud(xyz1)

        source_desc = get_desc(descpath, f"cloud_bin_" + str(t0) + f".desc.bin")
        target_desc = get_desc(descpath, f"cloud_bin_" + str(t1) + f".desc.bin")
        feat0 = make_open3d_feature(source_desc, 32, source_desc.shape[0])
        feat1 = make_open3d_feature(target_desc, 32, target_desc.shape[0])

        reg_timer.tic()
        distance_threshold = 0.3
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd0, pcd1, feat0, feat1, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
        T_ransac = torch.from_numpy(ransac_result.transformation.astype(np.float32))
        reg_timer.toc()

        # Translation error
        rte = np.linalg.norm(T_ransac[:3, 3] - T_gth[:3, 3])
        rre = np.arccos((np.trace(T_ransac[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

        if rte < 2:
            rte_meter.update(rte)

        if not np.isnan(rre) and rre < np.pi / 180 * 5:
            rre_meter.update(rre * 180 / np.pi)

        if rte < 2 and not np.isnan(rre) and rre < np.pi / 180 * 5:
            success_meter.update(1)
        else:
            success_meter.update(0)
            logging.info(f"Failed with RTE: {rte}, RRE: {rre}")

        if (idx + 1) % 10 == 0:
            logging.info(
                f" RRE: {rre_meter.avg}, Success: {success_meter.sum} / {success_meter.count}" +
                f" ({success_meter.avg * 100} %)"
            )
            reg_timer.reset()

    logging.info(
        f"RTE: {rte_meter.avg}, var: {rte_meter.var}," +
        f" RRE: {rre_meter.avg}, var: {rre_meter.var}, Success: {success_meter.sum} " +
        f"/ {success_meter.count} ({success_meter.avg * 100} %)"
    )
