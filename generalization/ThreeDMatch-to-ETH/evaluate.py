import sys

sys.path.append('../../')
import open3d
import numpy as np
import time
import os
from ThreeDMatch.Test.tools import get_pcd, get_ETH_keypts, get_desc, loadlog
from sklearn.neighbors import KDTree
import glob


def calculate_M(source_desc, target_desc):
    """
    Find the mutually closest point pairs in feature space.
    source and target are descriptor for 2 point cloud key points. [5000, 512]
    """

    kdtree_s = KDTree(target_desc)
    sourceNNdis, sourceNNidx = kdtree_s.query(source_desc, 1)
    kdtree_t = KDTree(source_desc)
    targetNNdis, targetNNidx = kdtree_t.query(target_desc, 1)
    result = []
    for i in range(len(sourceNNidx)):
        if targetNNidx[sourceNNidx[i]] == i:
            result.append([i, sourceNNidx[i][0]])
    return np.array(result)


def register2Fragments(id1, id2, keyptspath, descpath, resultpath, desc_name='ppf'):
    cloud_bin_s = f'Hokuyo_{id1}'
    cloud_bin_t = f'Hokuyo_{id2}'
    write_file = f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'
    if os.path.exists(os.path.join(resultpath, write_file)):
        #      print(f"{write_file} already exists.")
        return 0, 0, 0
    pcd_s = get_pcd(pcdpath, cloud_bin_s)
    source_keypts = get_ETH_keypts(pcd_s, keyptspath, cloud_bin_s)
    pcd_t = get_pcd(pcdpath, cloud_bin_t)
    target_keypts = get_ETH_keypts(pcd_t, keyptspath, cloud_bin_t)
    # print(source_keypts.shape)
    source_desc = get_desc(descpath, cloud_bin_s, desc_name=desc_name)
    target_desc = get_desc(descpath, cloud_bin_t, desc_name=desc_name)
    source_desc = np.nan_to_num(source_desc)
    target_desc = np.nan_to_num(target_desc)

    key = f'{cloud_bin_s.split("_")[-1]}_{cloud_bin_t.split("_")[-1]}'
    if key not in gtLog.keys():
        num_inliers = 0
        inlier_ratio = 0
        gt_flag = 0
    else:
        # find mutually cloest point.
        corr = calculate_M(source_desc, target_desc)

        gtTrans = gtLog[key]
        frag1 = source_keypts[corr[:, 0]]
        frag2_pc = open3d.geometry.PointCloud()
        frag2_pc.points = open3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
        frag2_pc.transform(gtTrans)
        frag2 = np.asarray(frag2_pc.points)
        distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
        num_inliers = np.sum(distance < 0.1)
        inlier_ratio = num_inliers / len(distance)
        gt_flag = 1

        # calculate the transformation matrix using RANSAC, this is for Registration Recall.
        source_pcd = open3d.geometry.PointCloud()
        source_pcd.points = open3d.utility.Vector3dVector(source_keypts)
        target_pcd = open3d.geometry.PointCloud()
        target_pcd.points = open3d.utility.Vector3dVector(target_keypts)
        s_desc = open3d.pipelines.registration.Feature()
        s_desc.data = source_desc.T
        t_desc = open3d.pipelines.registration.Feature()
        t_desc.data = target_desc.T
        result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd, target_pcd, s_desc, t_desc,
            0.05,
            open3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
            open3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000))
        # write the transformation matrix into .log file for evaluation.
        with open(os.path.join(logpath, f'{desc_name}_{timestr}.log'), 'a+') as f:
            trans = result.transformation
            trans = np.linalg.inv(trans)
            s1 = f'{id1}\t {id2}\t  37\n'
            f.write(s1)
            f.write(f"{trans[0, 0]}\t {trans[0, 1]}\t {trans[0, 2]}\t {trans[0, 3]}\t \n")
            f.write(f"{trans[1, 0]}\t {trans[1, 1]}\t {trans[1, 2]}\t {trans[1, 3]}\t \n")
            f.write(f"{trans[2, 0]}\t {trans[2, 1]}\t {trans[2, 2]}\t {trans[2, 3]}\t \n")
            f.write(f"{trans[3, 0]}\t {trans[3, 1]}\t {trans[3, 2]}\t {trans[3, 3]}\t \n")

    s = f"{cloud_bin_s}\t{cloud_bin_t}\t{num_inliers}\t{inlier_ratio:.8f}\t{gt_flag}"
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'w+') as f:
        f.write(s)
    return num_inliers, inlier_ratio, gt_flag


def read_register_result(id1, id2):
    cloud_bin_s = f'Hokuyo_{id1}'
    cloud_bin_t = f'Hokuyo_{id2}'
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'r') as f:
        content = f.readlines()
    nums = content[0].replace("\n", "").split("\t")[2:5]
    return nums


if __name__ == '__main__':
    scene_list = [
        'gazebo_summer',
        'gazebo_winter',
        'wood_autmn',
        'wood_summer',
    ]
    desc_name = 'SpinNet'
    timestr = sys.argv[1]
    inliers_list = []
    recall_list = []
    for scene in scene_list:
        pcdpath = f"../../data/ETH/{scene}/"
        interpath = f"../../data/ETH/{scene}/01_Keypoints/"
        gtpath = f'../../data/ETH/{scene}/'
        keyptspath = interpath  # os.path.join(interpath, "keypoints/")
        descpath = os.path.join(".", f"{desc_name}_desc_{timestr}/{scene}")
        logpath = f"log_result/{scene}-evaluation"
        gtLog = loadlog(gtpath)
        resultpath = os.path.join(".", f"pred_result/{scene}/{desc_name}_result_{timestr}")
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)
        if not os.path.exists(logpath):
            os.makedirs(logpath)

        # register each pair
        fragments = glob.glob(pcdpath + '*.ply')
        num_frag = len(fragments)
        print(f"Start Evaluate Descriptor {desc_name} for {scene}")
        start_time = time.time()
        for id1 in range(num_frag):
            for id2 in range(id1 + 1, num_frag):
                num_inliers, inlier_ratio, gt_flag = register2Fragments(id1, id2, keyptspath, descpath, resultpath,
                                                                        desc_name)
        print(f"Finish Evaluation, time: {time.time() - start_time:.2f}s")

        # evaluate
        result = []
        for id1 in range(num_frag):
            for id2 in range(id1 + 1, num_frag):
                line = read_register_result(id1, id2)
                result.append([int(line[0]), float(line[1]), int(line[2])])
        result = np.array(result)
        indices_results = np.sum(result[:, 2] == 1)
        correct_match = np.sum(result[:, 1] > 0.05)
        recall = float(correct_match / indices_results) * 100
        print(f"Correct Match {correct_match}, ground truth Match {indices_results}")
        print(f"Recall {recall}%")
        ave_num_inliers = np.sum(np.where(result[:, 1] > 0.05, result[:, 0], np.zeros(result.shape[0]))) / correct_match
        print(f"Average Num Inliners: {ave_num_inliers}")
        recall_list.append(recall)
        inliers_list.append(ave_num_inliers)
    print(recall_list)
    average_recall = sum(recall_list) / len(recall_list)
    print(f"All 8 scene, average recall: {average_recall}%")
    average_inliers = sum(inliers_list) / len(inliers_list)
    print(f"All 8 scene, average num inliers: {average_inliers}")
