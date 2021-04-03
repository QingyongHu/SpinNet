import os
import open3d
import numpy as np


def get_pcd(pcdpath, filename):
    return open3d.io.read_point_cloud(os.path.join(pcdpath, filename + '.ply'))


def get_keypts(keyptspath, filename):
    keypts = np.fromfile(os.path.join(keyptspath, filename + '.keypts.bin'), dtype=np.float32)
    num_keypts = int(keypts[0])
    keypts = keypts[1:].reshape([num_keypts, 3])
    return keypts


def get_ETH_keypts(pcd, keyptspath, filename):
    pts = np.array(pcd.points)
    key_ind = np.loadtxt(os.path.join(keyptspath, filename + '_Keypoints.txt'), dtype=np.int)
    keypts = pts[key_ind]
    return keypts


def get_keypts_(keyptspath, filename):
    keypts = np.load(os.path.join(keyptspath, filename + f'.keypts.bin.npy'))
    return keypts


def get_desc(descpath, filename, desc_name):
    if desc_name == '3dmatch':
        desc = np.fromfile(os.path.join(descpath, filename + '.desc.3dmatch.bin'), dtype=np.float32)
        num_desc = int(desc[0])
        desc_size = int(desc[1])
        desc = desc[2:].reshape([num_desc, desc_size])
    elif desc_name == 'SpinNet':
        desc = np.load(os.path.join(descpath, filename + '.desc.SpinNet.bin.npy'))
    else:
        print("No such descriptor")
        exit(-1)
    return desc


def loadlog(gtpath):
    with open(os.path.join(gtpath, 'gt.log')) as f:
        content = f.readlines()
    result = {}
    i = 0
    while i < len(content):
        line = content[i].replace("\n", "").split("\t")[0:3]
        trans = np.zeros([4, 4])
        trans[0] = [float(x) for x in content[i + 1].replace("\n", "").split("\t")[0:4]]
        trans[1] = [float(x) for x in content[i + 2].replace("\n", "").split("\t")[0:4]]
        trans[2] = [float(x) for x in content[i + 3].replace("\n", "").split("\t")[0:4]]
        trans[3] = [float(x) for x in content[i + 4].replace("\n", "").split("\t")[0:4]]
        i = i + 5
        result[f'{int(line[0])}_{int(line[1])}'] = trans

    return result
