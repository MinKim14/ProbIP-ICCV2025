import torch
import articulate as art


class PoseEvaluator:
    def __init__(self, smpl_file, xsens=False):
        self._eval_fn = art.FullMotionEvaluator(
            smpl_file, joint_mask=torch.tensor([1, 2, 16, 17])
        )
        self.xsens = xsens

    def eval(self, global_pose_p, local_pose_p, pose_t, xsens_pose=None):
        global_pose_p = global_pose_p.clone().view(-1, 24, 3, 3)
        local_pose_p = local_pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)

        # global_pose_p[:, [7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=global_pose_p.device)
        # local_pose_p[:, [7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=local_pose_p.device)

        # local_pose_t[:, [7, 8, 10, 11, 20, 21, 22, 23]] = torch.eye(3, device=pose_t.device)

        # global_pose_t = self._eval_fn.model.forward_kinematics(pose_t, None, None, calc_mesh=False)[0]



        errs = self._eval_fn(global_pose_p, local_pose_p, pose_t, xsens_pose=xsens_pose)
        return torch.stack(
            [
                errs[9],
                errs[3],
                errs[0] * 100,
                errs[1] * 100,
                errs[4] / 100,
                errs[10],
                errs[11],
                errs[12],
                errs[5] / 100,
                errs[2],
            ]
        )

    def print(self, errors):
        names = [
            "SIP Error [°]",
            "Angular Error [°]",
            "Angular Error (w/o pelvis) [°]",
            "Positional Error [cm]",
            "Mesh Error [cm]",
            "Jitter Error [102m/s3)]",
        ]
        if self.xsens:
            idx = [0, 6, 7, 2, 3, 4]
            for i, name in enumerate(names):
                print(
                    "%s: %.2f (+/- %.2f)" % (name, errors[idx[i], 0], errors[idx[i], 1])
                )
        else:
            idx = [0, 1, 5, 2, 3, 4]
            for i, name in enumerate(names):
                print(
                    "%s: %.2f (+/- %.2f)" % (name, errors[idx[i], 0], errors[idx[i], 1])
                )
