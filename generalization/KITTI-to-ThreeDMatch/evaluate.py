import sys

sys.path.append('../../')
import open3d
import numpy as np
import time
import os
from ThreeDMatch.Test.tools import get_pcd, get_keypts, get_desc, loadlog
from sklearn.neighbors import KDTree


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


def register2Fragments(id1, id2, keyptspath, descpath, resultpath, desc_name='SpinNet'):
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    write_file = f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'
    if os.path.exists(os.path.join(resultpath, write_file)):
        print(f"{write_file} already exists.")
        return 0, 0, 0

    if is_D3Feat_keypts:
        keypts_path = './D3Feat_contralo-54-pred/keypoints/' + keyptspath.split('/')[-2] + '/' + cloud_bin_s + '.npy'
        source_keypts = np.load(keypts_path)
        source_keypts = source_keypts[-num_keypoints:, :]
        keypts_path = './D3Feat_contralo-54-pred/keypoints/' + keyptspath.split('/')[-2] + '/' + cloud_bin_t + '.npy'
        target_keypts = np.load(keypts_path)
        target_keypts = target_keypts[-num_keypoints:, :]
        source_desc = get_desc(descpath, cloud_bin_s, desc_name=desc_name)
        target_desc = get_desc(descpath, cloud_bin_t, desc_name=desc_name)
        source_desc = np.nan_to_num(source_desc)
        target_desc = np.nan_to_num(target_desc)
        source_desc = source_desc[-num_keypoints:, :]
        target_desc = target_desc[-num_keypoints:, :]
    else:
        source_keypts = get_keypts(keyptspath, cloud_bin_s)
        target_keypts = get_keypts(keyptspath, cloud_bin_t)
        # print(source_keypts.shape)
        source_desc = get_desc(descpath, cloud_bin_s, desc_name=desc_name)
        target_desc = get_desc(descpath, cloud_bin_t, desc_name=desc_name)
        source_desc = np.nan_to_num(source_desc)
        target_desc = np.nan_to_num(target_desc)
        if source_desc.shape[0] > num_keypoints:
            rand_ind = np.random.choice(source_desc.shape[0], num_keypoints, replace=False)
            source_keypts = source_keypts[rand_ind]
            target_keypts = target_keypts[rand_ind]
            source_desc = source_desc[rand_ind]
            target_desc = target_desc[rand_ind]

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
        num_inliers = np.sum(distance < 0.10)
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

        # Another registration method
        corr_v = open3d.utility.Vector2iVector(corr)
        result = open3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source_pcd, target_pcd, corr_v,
            0.05,
            open3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
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
    cloud_bin_s = f'cloud_bin_{id1}'
    cloud_bin_t = f'cloud_bin_{id2}'
    with open(os.path.join(resultpath, f'{cloud_bin_s}_{cloud_bin_t}.rt.txt'), 'r') as f:
        content = f.readlines()
    nums = content[0].replace("\n", "").split("\t")[2:5]
    return nums


if __name__ == '__main__':
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    desc_name = 'SpinNet'
    timestr = sys.argv[1]
    inliers_list = []
    recall_list = []
    inliers_ratio_list = []
    num_keypoints = 5000
    is_D3Feat_keypts = False
    for scene in scene_list:
        pcdpath = f"../../data/3DMatch/fragments/{scene}/"
        interpath = f"../../data/3DMatch/intermediate-files-real/{scene}/"
        gtpath = f'../../data/3DMatch/fragments/{scene}-evaluation/'
        keyptspath = interpath  # os.path.join(interpath, "keypoints/")
        descpath = os.path.join(".", f"{desc_name}_desc_{timestr}/{scene}")
        gtLog = loadlog(gtpath)
        logpath = f"log_result/{scene}-evaluation"
        resultpath = os.path.join(".", f"pred_result/{scene}/{desc_name}_result_{timestr}")
        if not os.path.exists(resultpath):
            os.makedirs(resultpath)
        if not os.path.exists(logpath):
            os.makedirs(logpath)

        # register each pair
        num_frag = len(os.listdir(pcdpath))
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
        ave_inlier_ratio = np.sum(
            np.where(result[:, 1] > 0.05, result[:, 1], np.zeros(result.shape[0]))) / correct_match
        print(f"Average Num Inliner Ratio: {ave_inlier_ratio}")
        recall_list.append(recall)
        inliers_list.append(ave_num_inliers)
        inliers_ratio_list.append(ave_inlier_ratio)
    print(recall_list)
    average_recall = sum(recall_list) / len(recall_list)
    print(f"All 8 scene, average recall: {average_recall}%")
    average_inliers = sum(inliers_list) / len(inliers_list)
    print(f"All 8 scene, average num inliers: {average_inliers}")
    average_inliers_ratio = sum(inliers_ratio_list) / len(inliers_list)
    print(f"All 8 scene, average num inliers ratio: {average_inliers_ratio}")
