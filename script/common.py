import open3d
import numpy as np
import os
import time
import torch
from sklearn.neighbors import KDTree
import pointnet2_ops.pointnet2_utils as pnt2
import torch.nn.functional as F
from torch.autograd import Variable


class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False


def select_patches(pts, ind, num_patches=1024, vicinity=0.15, num_points_per_patch=1024, is_rand=True):
    # A point sampling algorithm for 3d matching of irregular geometries.
    tree = KDTree(pts[:, 0:3])
    num_points = pts.shape[0]
    if is_rand:
        out_inds = np.random.choice(range(ind.shape[0]), num_patches, replace=False)
        inds = ind[out_inds]
    else:
        inds = ind
    refer_pts = pts[inds]

    ind_local = tree.query_radius(refer_pts[:, 0:3], r=vicinity)
    local_patches = []
    for i in range(np.size(ind_local)):
        local_neighbors = pts[ind_local[i], :]
        if local_neighbors.shape[0] >= num_points_per_patch:
            temp = np.random.choice(range(local_neighbors.shape[0]), num_points_per_patch, replace=False)
            local_neighbors = local_neighbors[temp]
            local_neighbors[-1, :] = refer_pts[i, :]
        else:
            fix_idx = np.asarray(range(local_neighbors.shape[0]))
            while local_neighbors.shape[0] + fix_idx.shape[0] < num_points_per_patch:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(local_neighbors.shape[0]))), axis=0)
            random_idx = np.random.choice(local_neighbors.shape[0], num_points_per_patch - fix_idx.shape[0],
                                          replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
            local_neighbors = local_neighbors[choice_idx]
            local_neighbors[-1, :] = refer_pts[i, :]

            # fill_num = num_points_per_patch-local_neighbors.shape[0]
            # local_neighbors = np.concatenate((local_neighbors, np.tile(refer_pts[i,:],(fill_num,1))), axis=0)
        local_patches.append(local_neighbors)
    if is_rand:
        return local_patches, out_inds
    else:
        return local_patches


def transform_pc_pytorch(pc, sn):
    '''

    :param pc: 3xN tensor
    :param sn: 5xN tensor / 4xN tensor
    :param node: 3xM tensor
    :return: pc, sn, node of the same shape, detach
    '''
    angles_3d = np.random.rand(3) * np.pi * 2
    shift = np.random.uniform(-1, 1, (1, 3))

    sigma, clip = 0.010, 0.02
    N, C = pc.shape
    jitter_pc = np.clip(sigma * np.random.randn(N, 3), -1 * clip, clip)
    sigma, clip = 0.010, 0.02
    jitter_sn = np.clip(sigma * np.random.randn(N, 4), -1 * clip, clip)
    pc += jitter_pc
    sn += jitter_sn

    pc = pc_rotate_translate(pc, angles_3d, shift)
    sn[:, 0:3] = vec_rotate(sn[:, 0:3], angles_3d)  # 3x3 * 3xN -> 3xN

    return pc, sn, \
           angles_3d, shift


def l2_norm(input, axis=1):
    norm = torch.norm(input, p=2, dim=axis, keepdim=True)
    output = torch.div(input, norm)
    return output


def angles2rotation_matrix(angles):
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def pc_rotate_translate(data, angles, translates):
    '''
    :param data: numpy array of Nx3 array
    :param angles: numpy array / list of 3
    :param translates: numpy array / list of 3
    :return: rotated_data: numpy array of Nx3
    '''
    R = angles2rotation_matrix(angles)
    rotated_data = np.dot(data, np.transpose(R)) + translates

    return rotated_data


def pc_rotate_translate_torch(data, angles, translates):
    '''
    :param data: Tensor of BxNx3 array
    :param angles: Tensor of Bx3
    :param translates: Tensor of Bx3
    :return: rotated_data: Tensor of Nx3
    '''
    device = data.device
    B, N, _ = data.shape

    R = np.zeros([B, 3, 3])
    for i in range(B):
        R[i] = angles2rotation_matrix(angles[i])  # 3x3
    R = torch.FloatTensor(R).to(device)

    rotated_data = torch.matmul(data, R.transpose(-1, -2)) + torch.FloatTensor(translates).unsqueeze(1).to(device)

    return rotated_data


def _pc_rotate_translate_torch(data, R, translates):
    '''
    :param data: Tensor of BxNx3 array
    :param angles: Tensor of Bx3
    :param translates: Tensor of Bx3
    :return: rotated_data: Tensor of Nx3
    '''
    device = data.device
    B, N, _ = data.shape

    rotated_data = torch.matmul(data, R.to(device).transpose(-1, -2)) + torch.FloatTensor(translates).unsqueeze(1).to(
        device)

    return rotated_data


def max_ind(data):
    B, C, row, col = data.shape
    inds = np.zeros([B, 2])
    for i in range(B):
        ind = torch.argmax(data[i])
        r = int(ind // col)
        c = ind % col
        inds[i, 0] = r
        inds[i, 1] = c
    return inds


def vec_rotate(data, angles):
    '''
    :param data: numpy array of Nx3 array
    :param angles: numpy array / list of 3
    :return: rotated_data: numpy array of Nx3
    '''
    R = angles2rotation_matrix(angles)
    rotated_data = np.dot(data, R)

    return rotated_data


def vec_rotate_torch(data, angles):
    '''
    :param data: BxNx3 tensor
    :param angles: Bx3 numpy array
    :return:
    '''
    device = data.device
    B, N, _ = data.shape

    R = np.zeros([B, 3, 3])
    for i in range(B):
        R[i] = angles2rotation_matrix(angles[i])  # 3x3
    R = torch.FloatTensor(R).to(device)

    rotated_data = torch.matmul(data, R.transpose(-1, -2))  # BxNx3 * Bx3x3 -> BxNx3
    return rotated_data


def rotate_perturbation_point_cloud(data, angle_sigma=0.01, angle_clip=0.05):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    # truncated Gaussian sampling
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    rotated_data = vec_rotate(data, angles)

    return rotated_data


def jitter_point_cloud(data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original point clouds
        Return:
          BxNx3 array, jittered point clouds
    """
    B, N, C = data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += data
    return jittered_data


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def cdist(a, b):
    '''
    :param a:
    :param b:
    :return:
    '''
    diff = a.unsqueeze(0) - b.unsqueeze(1)
    dis_matrix = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-12)
    return dis_matrix


def s2_grid(n_alpha, n_beta):
    '''
    :return: rings around the equator
    size of the kernel = n_alpha * n_beta
    '''
    beta = np.linspace(start=0, stop=np.pi, num=n_beta, endpoint=False) + np.pi / n_beta / 2
    # ele = np.arcsin(np.linspace(start=0, stop=1, num=n_beta / 2, endpoint=False) + 1 / n_beta / 4)
    # beta = np.concatenate([np.sort(-ele), ele])
    alpha = np.linspace(start=0, stop=2 * np.pi, num=n_alpha, endpoint=False) + np.pi / n_alpha
    B, A = np.meshgrid(beta, alpha, indexing='ij')
    B = B.flatten()
    A = A.flatten()
    grid = np.stack((B, A), axis=1)
    return grid


def pad_image(input, kernel_size):
    """
    Circularly padding image for convolution
    :param input: [B, C, H, W]
    :param kernel_size:
    :return:
    """
    device = input.device
    if kernel_size % 2 == 0:
        pad_size = kernel_size // 2
        output = torch.cat([input, input[:, :, :, 0:pad_size]], dim=3)
        zeros_pad = torch.zeros([output.shape[0], output.shape[1], pad_size, output.shape[3]]).to(device)
        output = torch.cat([output, zeros_pad], dim=2)
    else:
        pad_size = (kernel_size - 1) // 2
        output = torch.cat([input, input[:, :, :, 0:pad_size]], dim=3)
        output = torch.cat([input[:, :, :, -pad_size:], output], dim=3)
        zeros_pad = torch.zeros([output.shape[0], output.shape[1], pad_size, output.shape[3]]).to(device)
        output = torch.cat([output, zeros_pad], dim=2)
        output = torch.cat([zeros_pad, output], dim=2)
    return output


def pad_image_3d(input, kernel_size):
    """
    Circularly padding image for convolution
    :param input: [B, C, D, H, W]
    :param kernel_size:
    :return:
    """
    device = input.device
    if kernel_size % 2 == 0:
        pad_size = kernel_size // 2
        output = torch.cat([input, input[:, :, :, :, 0:pad_size]], dim=4)
        zeros_pad = torch.zeros([output.shape[0], output.shape[1], output.shape[2], pad_size, output.shape[4]]).to(
            device)
        output = torch.cat([output, zeros_pad], dim=3)
    else:
        pad_size = (kernel_size - 1) // 2
        output = torch.cat([input, input[:, :, :, :, 0:pad_size]], dim=4)
        output = torch.cat([input[:, :, :, :, -pad_size:], output], dim=4)
        zeros_pad = torch.zeros([output.shape[0], output.shape[1], output.shape[2], pad_size, output.shape[4]]).to(
            device)
        output = torch.cat([output, zeros_pad], dim=3)
        output = torch.cat([zeros_pad, output], dim=3)
    return output


def pad_image_on_azi(input, kernel_size):
    """
    Circularly padding image for convolution
    :param input: [B, C, H, W]
    :param kernel_size:
    :return:
    """
    device = input.device
    pad_size = (kernel_size - 1) // 2
    output = torch.cat([input, input[:, :, :, 0:pad_size]], dim=3)
    output = torch.cat([input[:, :, :, -pad_size:], output], dim=3)
    return output


def kmax_pooling(x, dim, k):
    kmax = x.topk(k, dim=dim)[0]
    return kmax


def change_coordinates(coords, radius, p_from='C', p_to='S'):
    """
    Change Spherical to Cartesian coordinates and vice versa, for points x in S^2.

    In the spherical system, we have coordinates beta and alpha,
    where beta in [0, pi] and alpha in [0, 2pi]

    We use the names beta and alpha for compatibility with the SO(3) code (S^2 being a quotient SO(3)/SO(2)).
    Many sources, like wikipedia use theta=beta and phi=alpha.

    :param coords: coordinate array
    :param p_from: 'C' for Cartesian or 'S' for spherical coordinates
    :param p_to: 'C' for Cartesian or 'S' for spherical coordinates
    :return: new coordinates
    """
    if p_from == p_to:
        return coords
    elif p_from == 'S' and p_to == 'C':

        beta = coords[..., 0]
        alpha = coords[..., 1]
        r = radius

        out = np.empty(beta.shape + (3,))

        ct = np.cos(beta)
        cp = np.cos(alpha)
        st = np.sin(beta)
        sp = np.sin(alpha)
        out[..., 0] = r * st * cp  # x
        out[..., 1] = r * st * sp  # y
        out[..., 2] = r * ct  # z
        return out

    elif p_from == 'C' and p_to == 'S':

        x = coords[..., 0]
        y = coords[..., 1]
        z = coords[..., 2]

        out = np.empty(x.shape + (2,))
        out[..., 0] = np.arccos(z)  # beta
        out[..., 1] = np.arctan2(y, x)  # alpha
        return out

    else:
        raise ValueError('Unknown conversion:' + str(p_from) + ' to ' + str(p_to))


def get_voxel_coordinate(radius, rad_n, azi_n, ele_n):
    grid = s2_grid(n_alpha=azi_n, n_beta=ele_n)
    pts_xyz_on_S2 = change_coordinates(grid, radius, 'S', 'C')
    pts_xyz_on_S2 = np.expand_dims(pts_xyz_on_S2, axis=0).repeat(rad_n, axis=0)
    scale = np.reshape(np.arange(rad_n) / rad_n + 1 / (2 * rad_n), [rad_n, 1, 1])
    pts_xyz = scale * pts_xyz_on_S2
    return pts_xyz


def knn_query(pts, new_pts, knn):
    """
    :param pts: all points, [B. N. 3]
    :param new_pts: query points, [B, S. 3]
    :param knn: the number of queried points
    :return:
    """
    device = pts.device
    B, N, C = pts.shape
    _, S, _ = new_pts.shape
    group_idx = torch.arange(N).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_pts, pts)


def sphere_query(pts, new_pts, radius, nsample):
    """
    :param pts: all points, [B. N. 3]
    :param new_pts: query points, [B, S. 3]
    :param radius: local sperical radius
    :param nsample: max sample number in local sphere
    :return:
    """

    device = pts.device
    B, N, C = pts.shape
    _, S, _ = new_pts.shape

    pts = pts.contiguous()
    new_pts = new_pts.contiguous()
    group_idx = pnt2.ball_query(radius, nsample, pts, new_pts)
    mask = group_idx[:, :, 0].unsqueeze(2).repeat(1, 1, nsample)
    mask = (group_idx == mask).float()
    mask[:, :, 0] = 0

    # C implementation
    pts_trans = pts.transpose(1, 2).contiguous()
    new_points = pnt2.grouping_operation(
        pts_trans, group_idx
    )  # (B, 3, npoint, nsample)
    new_points = new_points.permute([0, 2, 3, 1])

    # replace the wrong points using new_pts
    mask = mask.unsqueeze(3).repeat([1, 1, 1, 3])
    # new_pts = new_pts.unsqueeze(2).repeat([1, 1, nsample + 1, 1])
    new_pts = new_pts.unsqueeze(2).repeat([1, 1, nsample, 1])
    n_points = new_points * (1 - mask).float() + new_pts * mask.float()

    del mask
    del new_points
    del group_idx
    del new_pts
    del pts
    del pts_trans

    return n_points


def sphere_query_new(pts, new_pts, radius, nsample):
    """
    :param pts: all points, [B. N. 3]
    :param new_pts: query points, [B, S. 3]
    :param radius: local sperical radius
    :param nsample: max sample number in local sphere
    :return:
    """

    device = pts.device
    B, N, C = pts.shape
    _, S, _ = new_pts.shape

    pts = pts.contiguous()
    new_pts = new_pts.contiguous()
    group_idx = pnt2.ball_query(radius, nsample, pts, new_pts)
    mask = group_idx[:, :, 0].unsqueeze(2).repeat(1, 1, nsample)
    mask = (group_idx == mask).float()
    mask[:, :, 0] = 0

    mask1 = (group_idx[:, :, 0] == 0).unsqueeze(2).float()
    mask1 = torch.cat([mask1, torch.zeros_like(mask)[:, :, :-1]], dim=2)
    mask = mask + mask1

    # C implementation
    pts_trans = pts.transpose(1, 2).contiguous()
    new_points = pnt2.grouping_operation(
        pts_trans, group_idx
    )  # (B, 3, npoint, nsample)
    new_points = new_points.permute([0, 2, 3, 1])

    # replace the wrong points using new_pts
    mask = mask.unsqueeze(3).repeat([1, 1, 1, 3])
    n_points = new_points * (1 - mask).float()

    del mask
    del new_points
    del group_idx
    del new_pts
    del pts
    del pts_trans

    return n_points


def var_to_invar(pts, rad_n, azi_n, ele_n):
    """
    :param pts: input points data, [B, N, nsample, 3]
    :param rad_n: radial number
    :param azi_n: azimuth number
    :param ele_n: elevator number
    :return:
    """
    device = pts.device
    B, N, nsample, C = pts.shape
    assert N == rad_n * azi_n * ele_n
    angle_step = np.array([0, 0, 2 * np.pi / azi_n])
    pts = pts.view(B, rad_n, ele_n, azi_n, nsample, C)

    R = np.zeros([azi_n, 3, 3])
    for i in range(azi_n):
        angle = -1 * i * angle_step
        r = angles2rotation_matrix(angle)
        R[i] = r
    R = torch.FloatTensor(R).to(device)
    R = R.view(1, 1, 1, azi_n, 3, 3).repeat(B, rad_n, ele_n, 1, 1, 1)
    new_pts = torch.matmul(pts, R.transpose(-1, -2))

    del R
    del pts

    return new_pts.view(B, -1, nsample, C)


def cal_Z_axis(local_cor, local_weight=None, ref_point=None):
    device = local_cor.device
    B, N, _ = local_cor.shape
    cov_matrix = torch.matmul(local_cor.transpose(-1, -2), local_cor) if local_weight is None \
        else Variable(torch.matmul(local_cor.transpose(-1, -2), local_cor * local_weight), requires_grad=True)
    Z_axis = torch.symeig(cov_matrix, eigenvectors=True)[1][:, :, 0]
    mask = (torch.sum(-Z_axis * ref_point, dim=1) < 0).float().unsqueeze(1)
    Z_axis = Z_axis * (1 - mask) - Z_axis * mask

    return Z_axis


def RodsRotatFormula(a, b):
    B, _ = a.shape
    device = a.device
    b = b.to(device)
    c = torch.cross(a, b)
    theta = torch.acos(F.cosine_similarity(a, b)).unsqueeze(1).unsqueeze(2)

    c = F.normalize(c, p=2, dim=1)
    one = torch.ones(B, 1, 1).to(device)
    zero = torch.zeros(B, 1, 1).to(device)
    a11 = zero
    a12 = -c[:, 2].unsqueeze(1).unsqueeze(2)
    a13 = c[:, 1].unsqueeze(1).unsqueeze(2)
    a21 = c[:, 2].unsqueeze(1).unsqueeze(2)
    a22 = zero
    a23 = -c[:, 0].unsqueeze(1).unsqueeze(2)
    a31 = -c[:, 1].unsqueeze(1).unsqueeze(2)
    a32 = c[:, 0].unsqueeze(1).unsqueeze(2)
    a33 = zero
    Rx = torch.cat(
        (torch.cat((a11, a12, a13), dim=2), torch.cat((a21, a22, a23), dim=2), torch.cat((a31, a32, a33), dim=2)),
        dim=1)
    I = torch.eye(3).to(device)
    R = I.unsqueeze(0).repeat(B, 1, 1) + torch.sin(theta) * Rx + (1 - torch.cos(theta)) * torch.matmul(Rx, Rx)
    return R.transpose(-1, -2)


def rgbd_to_point_cloud(data_dir, ind, downsample=0.03, aligned=True):
    pcd = open3d.read_point_cloud(os.path.join(data_dir, f'{ind}.ply'))
    # downsample the point cloud
    if downsample != 0:
        pcd = open3d.voxel_down_sample(pcd, voxel_size=downsample)
    # align the point cloud
    if aligned is True:
        matrix = np.load(os.path.join(data_dir, f'{ind}.pose.npy'))
        pcd.transform(matrix)

    return pcd


def cal_local_normal(pcd):
    if open3d.geometry.estimate_normals(pcd, open3d.KDTreeSearchParamKNN(knn=17)):
        return True
    else:
        print("Calculate Normal Error")
        return False


def select_referenced_point(pcd, num_patches=2048):
    # A point sampling algorithm for 3d matching of irregular geometries.
    pts = np.asarray(pcd.points)
    num_points = pts.shape[0]
    inds = np.random.choice(range(num_points), num_patches, replace=False)
    return open3d.geometry.select_down_sample(pcd, inds)


def collect_local_neighbor(ref_pcd, pcd, vicinity=0.3, num_points_per_patch=1024, random_state=None):
    # collect local neighbor within vicinity for each interest point.
    # each local patch is downsampled to 1024 (setting of PPFNet p5.)
    kdtree = open3d.geometry.KDTreeFlann(pcd)
    dict = []
    for point in ref_pcd.points:
        # Bug fix: here the first returned result will be itself. So the calculated ppf will be nan.
        [k, idx, variant] = kdtree.search_radius_vector_3d(point, vicinity)
        # random select fix number [num_points] of points to form the local patch.
        if random_state is not None:
            if k > num_points_per_patch:
                idx = random_state.choice(idx[1:], num_points_per_patch, replace=False)
            else:
                idx = random_state.choice(idx[1:], num_points_per_patch)
        else:
            if k > num_points_per_patch:
                idx = np.random.choice(idx[1:], num_points_per_patch, replace=False)
            else:
                idx = np.random.choice(idx[1:], num_points_per_patch)
        dict.append(idx)
    return dict
