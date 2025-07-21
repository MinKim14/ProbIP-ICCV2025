import os
import sys

sys.path.append(os.path.abspath("./"))

import glob

import articulate as art
import numpy as np
import torch

from tqdm import tqdm
from utils.config import paths
import pickle


def _glb_mat_xsens_to_glb_mat_smpl(glb_full_pose_xsens):
    glb_full_pose_smpl = torch.eye(3).repeat(glb_full_pose_xsens.shape[0], 24, 1, 1)
    indices = [
        0,
        19,
        15,
        1,
        20,
        16,
        3,
        21,
        17,
        4,
        22,
        18,
        5,
        11,
        7,
        6,
        12,
        8,
        13,
        9,
        13,
        9,
        13,
        9,
    ]
    for idx, i in enumerate(indices):
        glb_full_pose_smpl[:, idx, :] = glb_full_pose_xsens[:, i, :]
    return glb_full_pose_smpl


def process_xsens():
    r"""
    imu_mask [LeftElbow, RightElbow, LeftShoulder, RightShoulder, Head, Pelvis]

    joint_mask: ['LeftUpperLeg', 'RightUpperLeg', 'L5', 'LeftLowerLeg', 'RightLowerLeg', 'L3',
                 'T12', 'T8', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head', 'LeftUpperArm',
                 'RightUpperArm', 'LeftForeArm', 'RightForeArm']
    """

    imu_mask = torch.tensor([9, 5, 15, 12, 2, 0])
    joint_mask = torch.tensor([19, 15, 1, 20, 16, 2, 3, 4, 5, 11, 7, 6, 12, 8, 13, 9])

    print("\n")
    infos = os.listdir(os.path.join(paths.split_dir, "xsens"))
    for info_name in infos:

        dataset, phase = info_name.split(".")[0].split("_")
        with open(os.path.join(paths.split_dir, "xsens", info_name), "r") as file:
            l = file.read().splitlines()
        print("processing {}_{}...".format(dataset, phase))

        out_vrot, out_vacc, out_transl = [], [], []

        out_global_xsens_pose = []
        out_global_smpl_pose = []
        out_local_smpl_pose = []

        print("processing {}...".format(info_name))

        body_model = art.ParametricModel(paths.smpl_m, device="cpu", video_path="./")

        for motion_name in tqdm(l):
            # print(motion_name)
            temp_data = torch.load(
                os.path.join(paths.extract_dir, dataset, motion_name)
            )

            glb_pose = art.math.quaternion_to_rotation_matrix(
                temp_data["joint"]["orientation"]
            ).view(-1, 23, 3, 3)
            out_global_xsens_pose.append(
                art.math.rotation_matrix_to_axis_angle(glb_pose).view(-1, 23, 3)
            )

            glb_gt_smpl = _glb_mat_xsens_to_glb_mat_smpl(glb_pose)
            out_global_smpl_pose.append(
                art.math.rotation_matrix_to_axis_angle(glb_gt_smpl).view(-1, 24, 3)
            )

            local_gt_smpl = body_model.inverse_kinematics_R(glb_gt_smpl).view(
                glb_gt_smpl.shape[0], 24, 3, 3
            )
            out_local_smpl_pose.append(
                art.math.rotation_matrix_to_axis_angle(local_gt_smpl).view(-1, 24, 3)
            )

            acc = temp_data["imu"]["free acceleration"][:, imu_mask].view(-1, 6, 3)
            ori = art.math.quaternion_to_rotation_matrix(
                temp_data["imu"]["calibrated orientation"][:, imu_mask]
            ).view(-1, 6, 9)

            out_vacc.append(acc)
            out_vrot.append(ori)

            transl = temp_data["joint"]["position"][:, 0]
            out_transl.append(transl)

        out_dir = os.path.join(paths.work_dir, phase, dataset)
        os.makedirs(os.path.join(paths.work_dir, phase, dataset), exist_ok=True)
        print("Saving")

        torch.save(out_global_xsens_pose, os.path.join(out_dir, "global_xsens_pose.pt"))
        torch.save(out_global_smpl_pose, os.path.join(out_dir, "global_smpl_pose.pt"))
        torch.save(out_local_smpl_pose, os.path.join(out_dir, "local_smpl_pose.pt"))

        torch.save(out_transl, os.path.join(out_dir, "tran.pt"))
        torch.save(out_vrot, os.path.join(out_dir, "vrot.pt"))
        torch.save(out_vacc, os.path.join(out_dir, "vacc.pt"))


def fill_dip_nan(tensor):
    nan_indices = torch.isnan(tensor)
    filled_tensor = tensor.clone()
    for t in range(tensor.size(0)):
        for i in range(tensor.size(1)):
            for j in range(tensor.size(2)):
                if nan_indices[t, i, j]:
                    left_idx = t - 1
                    while left_idx >= 0 and torch.isnan(tensor[left_idx, i, j]):
                        left_idx -= 1
                    left_neighbor_value = tensor[left_idx, i, j] if left_idx >= 0 else 0

                    right_idx = t + 1
                    while right_idx < tensor.size(0) and torch.isnan(
                        tensor[right_idx, i, j]
                    ):
                        right_idx += 1
                    right_neighbor_value = (
                        tensor[right_idx, i, j] if right_idx < tensor.size(0) else 0
                    )

                    filled_tensor[t, i, j] = (
                        left_neighbor_value + right_neighbor_value
                    ) / 2
    return filled_tensor


def process_dipimu():
    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ["s_09", "s_10"]

    train_split = [
        "s_01",
        "s_02",
        "s_03",
        "s_04",
        "s_05",
        "s_06",
        "s_07",
        "s_08",
    ]

    accs, oris, poses, trans = [], [], [], []

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(
                "data/dip_trans",
                subject_name + "_" + motion_name.replace(".pkl", ".pt"),
            )
            data = pickle.load(open(path, "rb"), encoding="latin1")
            # data = torch.load(path)
            acc = data["imu_acc"][:, imu_mask].float()
            ori = data["imu_ori"][:, imu_mask].float()
            pose = data["pose"].float()

            if True in torch.isnan(acc):
                acc = fill_dip_nan(acc)
            if True in torch.isnan(ori):
                ori = fill_dip_nan(ori.view(-1, 6, 9))
            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            tran = tran[6:-6]
            if (
                torch.isnan(acc).sum() == 0
                and torch.isnan(ori).sum() == 0
                and torch.isnan(pose).sum() == 0
            ):
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                # dip-imu does not contain translations
                # trans.append(tran.clone())
                trans.append(torch.zeros(pose.shape[0], 3))
            else:
                print(
                    "DIP-IMU: %s/%s has too much nan! Discard!"
                    % (subject_name, motion_name)
                )
                print(len(acc))

    os.makedirs(paths.dipimu_dir, exist_ok=True)
    torch.save(
        {"acc": accs, "ori": oris, "pose": poses, "tran": trans},
        os.path.join(paths.dipimu_dir, "dip_test.pt"),
    )
    print("Preprocessed DIP-IMU dataset is saved at", paths.dipimu_dir)


if __name__ == "__main__":
    process_xsens()
    # process_dipimu()
