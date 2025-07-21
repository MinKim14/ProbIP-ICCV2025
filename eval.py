import numpy as np
import torch

import articulate as art
from dataset import *
from evaluator import PoseEvaluator
from model.probip import ProbIP
from utils.config import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(
    model,
    dataloader,
    full=False,
    sensor_idx=[0, 1, 2, 3, 4, 5],
    name="dip",
):
    model.eval()

    offline_errors = []
    if name == "dip":
        evaluator = PoseEvaluator(smpl_file=paths.smpl_m, xsens=False)
    else:
        evaluator = PoseEvaluator(smpl_file=paths.smpl_m, xsens=True)

    for i, s in enumerate(dataloader):
        pose, vacc, vori = s["pose"].float(), s["vacc"], s["vori"]

        vacc = (
            vacc[:, :, sensor_idx].reshape(vacc.shape[0], vacc.shape[1], -1).to(device)
        )
        vori = (
            vori[:, :, sensor_idx].reshape(vori.shape[0], vori.shape[1], -1).to(device)
        )

        input = torch.cat([vacc, vori], dim=-1)
        bs = vacc.shape[0]

        pose_t = pose.reshape(pose.shape[0], -1, 24, 3).to(model.device)
        pose_t = art.math.axis_angle_to_rotation_matrix(pose_t.reshape(-1, 3)).reshape(
            bs, -1, 24, 3, 3
        )

        (
            global_pose_t,
            _,
        ) = model.m.forward_kinematics(pose_t.reshape(-1, 24, 3, 3))

        global_pose_axis = art.math.rotation_matrix_to_axis_angle(
            global_pose_t.reshape(-1, 24, 3, 3)
        ).reshape(bs, -1, 24, 3)
        input = torch.cat([vacc, vori], dim=-1)
        init_pose = global_pose_axis[:, 0]

        if full:
            global_pose_prediction, local_pose_prediction, tran_prediction = model.forward_fullseq(input, init_pose)
        else:
            global_pose_prediction, local_pose_prediction, tran_prediction = model.forward_iter(input, init_pose)

        global_pose_prediction = global_pose_prediction.reshape(-1, 24, 3, 3)
        local_pose_prediction = local_pose_prediction.reshape(-1, 24, 3, 3)
        pose_t = pose_t.detach().cpu()

        offline_errors.append(evaluator.eval(global_pose_prediction, local_pose_prediction, pose_t))

    errors = torch.stack(offline_errors).mean(dim=0)
    evaluator.print(errors)


def main():

    modelIdx = sensor6Info
    model = ProbIP(
        sensor=modelIdx.sensor_idx,
        reduced=modelIdx.reduced_idx,
        smpl_file_path=paths.smpl_m,
    ).to(device)

    model.load_state_dict(
        torch.load(
            modelIdx.model_path,
            map_location=device,
        ),
        # strict=False,
    )

    model.eval()
    model.to(device)

    for name in ["dip",  "cip", "andy", "unipd", "virginia"]:
        if name == "dip":
            dataset = MotionDatasetDipFull()
        else:
            dataset = MotionDatasetXsensTest(name)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        print(f"===================={name}===========================")
        evaluate(model, dataloader, full=True, sensor_idx=modelIdx.input_idx, name=name)

if __name__ == "__main__":
    main()
