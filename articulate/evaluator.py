r"""
    Basic evaluators, and evaluators that measure differences between poses/trans of MANO/SMPL/SMPLH model.
"""

__all__ = [
    "BinaryConfusionMatrixEvaluator",
    "BinaryClassificationErrorEvaluator",
    "PositionErrorEvaluator",
    "RotationErrorEvaluator",
    "PerJointErrorEvaluator",
    "MeanPerJointErrorEvaluator",
    "MeshErrorEvaluator",
    "FullMotionEvaluator",
]


from .model import ParametricModel
from .math import *
import torch


class BasePoseEvaluator:
    r"""
    Base class for evaluators that evaluate motions.
    """

    def __init__(
        self,
        official_model_file: str,
        rep=RotationRepresentation.ROTATION_MATRIX,
        use_pose_blendshape=False,
        device=torch.device("cpu"),
    ):
        self.model = ParametricModel(
            official_model_file, use_pose_blendshape=use_pose_blendshape, device=device
        )
        self.rep = rep
        self.device = device

    def _preprocess(self, pose, shape=None, tran=None):
        pose = to_rotation_matrix(pose.to(self.device), self.rep).view(
            pose.shape[0], -1
        )
        shape = shape.to(self.device) if shape is not None else shape
        tran = tran.to(self.device) if tran is not None else tran
        return pose, shape, tran


class BinaryConfusionMatrixEvaluator:
    r"""
    Confusion matrix for binary classification tasks.

    The (i, j) entry stands for the number of instance i that is classified as j.
    """

    def __init__(self, is_after_sigmoid=False):
        r"""
        Init a binary confusion matrix evaluator.

        :param is_after_sigmoid: Whether a sigmoid function has been applied on the predicted values or not.
        """
        self.is_after_sigmoid = is_after_sigmoid

    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        r"""
        Get the confusion matrix.

        :param p: Predicted values (0 ~ 1 if is_after_sigmoid is True) in shape [*].
        :param t: True values (0 or 1) in shape [*].
        :return: Confusion matrix in shape [2, 2].
        """
        positive, negative = 0, 1
        p = (p > 0.5).float() if self.is_after_sigmoid else (p > 0).float()
        tp = ((p == positive) & (t == positive)).sum()
        fn = ((p == negative) & (t == positive)).sum()
        fp = ((p == positive) & (t == negative)).sum()
        tn = ((p == negative) & (t == negative)).sum()
        return torch.tensor([[tp, fn], [fp, tn]])


class BinaryClassificationErrorEvaluator(BinaryConfusionMatrixEvaluator):
    r"""
    Precision, recall, and f1 score for both positive and negative samples for binary classification tasks.
    """

    def __init__(self, is_after_sigmoid=False):
        r"""
        Init a binary classification error evaluator.

        :param is_after_sigmoid: Whether a sigmoid function has been applied on the predicted values or not.
        """
        super(BinaryClassificationErrorEvaluator, self).__init__(is_after_sigmoid)

    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        r"""
        Get the precision, recall, and f1 score for both positive and negative samples.

        :param p: Predicted values (0 ~ 1 if is_after_sigmoid is True) in shape [*].
        :param t: True values (0 or 1) in shape [*].
        :return: Tensor in shape [3, 2] where column 0 and 1 are the precision, recall, and f1 score
                 for positive(0) and negative(1) samples respectively.
        """
        tp, fn, fp, tn = (
            super(BinaryClassificationErrorEvaluator, self).__call__(p, t).view(-1)
        )

        precision_positive = tp.float() / (tp + fp)
        recall_positive = tp.float() / (tp + fn)
        f1_positive = 2 / (1 / precision_positive + 1 / recall_positive)

        precision_negative = tn.float() / (tn + fn)
        recall_negative = tn.float() / (tn + fp)
        f1_negative = 2 / (1 / precision_negative + 1 / recall_negative)

        return torch.tensor(
            [
                [precision_positive, precision_negative],
                [recall_positive, recall_negative],
                [f1_positive, f1_negative],
            ]
        )


class PositionErrorEvaluator:
    r"""
    Mean distance between two sets of points. Distances are defined as vector p-norm.
    """

    def __init__(self, dimension=3, p=2):
        r"""
        Init a distance error evaluator.

        Notes
        -----
        The two tensors being evaluated will be reshape to [n, dimension] and be regarded as n points.
        Then the average of p-norms of the difference of all corresponding points will be returned.

        Args
        -----
        :param dimension: Dimension of the vector space. By default 3.
        :param p: Distance will be evaluated by vector p-norm. By default 2.
        """
        self.dimension = dimension
        self.p = p

    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        r"""
        Get the mean p-norm distance between two sets of points.

        :param p: Tensor that can reshape to [n, dimension] that stands for n points.
        :param t: Tensor that can reshape to [n, dimension] that stands for n points.
        :return: Mean p-norm distance between all corresponding points.
        """
        return (
            (p.view(-1, self.dimension) - t.view(-1, self.dimension))
            .norm(p=self.p, dim=1)
            .mean()
        )


class RotationErrorEvaluator:
    r"""
    Mean angle between two sets of rotations. Angles are in degrees.
    """

    def __init__(self, rep=RotationRepresentation.ROTATION_MATRIX):
        r"""
        Init a rotation error evaluator.

        :param rep: The rotation representation used in the input.
        """
        self.rep = rep

    def __call__(self, p: torch.Tensor, t: torch.Tensor):
        r"""
        Get the mean angle between to sets of rotations.

        :param p: Tensor that can reshape to [n, rep_dim] that stands for n rotations.
        :param t: Tensor that can reshape to [n, rep_dim] that stands for n rotations.
        :return: Mean angle in degrees between all corresponding rotations.
        """
        return radian_to_degree(angle_between(p, t, self.rep).mean())


class PerJointErrorEvaluator(BasePoseEvaluator):
    r"""
    Position and local/global rotation error of each joint.
    """

    def __init__(
        self,
        official_model_file: str,
        align_joint=None,
        rep=RotationRepresentation.ROTATION_MATRIX,
        device=torch.device("cpu"),
    ):
        r"""
        Init a PJE Evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param device: torch.device, cpu or cuda.
        """
        super().__init__(official_model_file, rep, device=device)
        self.align_joint = 0 if align_joint is None else align_joint.value

    def __call__(self, pose_p: torch.Tensor, pose_t: torch.Tensor):
        r"""
        Get position and local/global rotation errors of all joints.

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :return: Tensor in shape [3, num_joint] where the ith column is the position error,
                 local rotation error, and global rotation error (in degrees) of the ith joint.
        """
        batch_size = pose_p.shape[0]
        pose_local_p, _, _ = self._preprocess(pose_p)
        pose_local_t, _, _ = self._preprocess(pose_t)
        pose_global_p, joint_p = self.model.forward_kinematics(pose_local_p)
        pose_global_t, joint_t = self.model.forward_kinematics(pose_local_t)
        offset_from_p_to_t = (
            joint_t[:, self.align_joint] - joint_p[:, self.align_joint]
        ).unsqueeze(1)
        joint_p = joint_p + offset_from_p_to_t
        position_error_array = (joint_p - joint_t).norm(dim=2).mean(dim=0)
        local_rotation_error_array = (
            angle_between(pose_local_p, pose_local_t).view(batch_size, -1).mean(dim=0)
        )
        global_rotation_error_array = (
            angle_between(pose_global_p, pose_global_t).view(batch_size, -1).mean(dim=0)
        )
        return torch.stack(
            (
                position_error_array,
                radian_to_degree(local_rotation_error_array),
                radian_to_degree(global_rotation_error_array),
            )
        )


class MeanPerJointErrorEvaluator(PerJointErrorEvaluator):
    r"""
    Mean position and local/global rotation error of all joints.
    """

    def __init__(
        self,
        official_model_file: str,
        align_joint=None,
        rep=RotationRepresentation.ROTATION_MATRIX,
        device=torch.device("cpu"),
    ):
        r"""
        Init a MPJE Evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param device: torch.device, cpu or cuda.
        """
        super().__init__(official_model_file, align_joint, rep, device)

    def __call__(self, pose_p: torch.Tensor, pose_t: torch.Tensor):
        r"""
        Get mean position and local/global rotation errors of all joints.

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :return: Tensor in shape [3] containing the mean position error,
                 local rotation error, and global rotation error (in degrees) of all joints.
        """
        error_array = super(MeanPerJointErrorEvaluator, self).__call__(pose_p, pose_t)
        return error_array.mean(dim=1)


class MeshErrorEvaluator(BasePoseEvaluator):
    r"""
    Mean mesh vertex position error.
    """

    def __init__(
        self,
        official_model_file: str,
        align_joint=None,
        rep=RotationRepresentation.ROTATION_MATRIX,
        use_pose_blendshape=False,
        device=torch.device("cpu"),
    ):
        r"""
        Init a mesh error evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param use_pose_blendshape: Whether to use pose blendshape or not.
        :param device: torch.device, cpu or cuda.
        """
        super().__init__(official_model_file, rep, use_pose_blendshape, device=device)
        self.align_joint = 0 if align_joint is None else align_joint.value

    def __call__(
        self,
        pose_p: torch.Tensor,
        pose_t: torch.Tensor,
        shape_p: torch.Tensor = None,
        shape_t: torch.Tensor = None,
    ):
        r"""
        Get mesh vertex position error.

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param shape_p: Predicted shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param shape_t: True shape that can expand [batch_size, 10]. Use None for the mean(zero) shape.
        :return: Mean mesh vertex position error.
        """
        pose_p, shape_p, _ = self._preprocess(pose_p, shape_p)
        pose_t, shape_t, _ = self._preprocess(pose_t, shape_t)
        _, joint_p, mesh_p = self.model.forward_kinematics(
            pose_p, shape_p, calc_mesh=True
        )
        _, joint_t, mesh_t = self.model.forward_kinematics(
            pose_t, shape_t, calc_mesh=True
        )
        offset_from_p_to_t = (
            joint_t[:, self.align_joint] - joint_p[:, self.align_joint]
        ).unsqueeze(1)
        mesh_error = (mesh_p + offset_from_p_to_t - mesh_t).norm(dim=2).mean()
        return mesh_error


class FullMotionEvaluator(BasePoseEvaluator):
    r"""
    Evaluator for full motions (pose sequences with global translations). Plenty of metrics.
    """

    def __init__(
        self,
        official_model_file: str,
        align_joint=None,
        rep=RotationRepresentation.ROTATION_MATRIX,
        use_pose_blendshape=False,
        fps=60,
        joint_mask=None,
        device=torch.device("cpu"),
    ):
        r"""
        Init a full motion evaluator.

        :param official_model_file: Path to the official SMPL/MANO/SMPLH model to be loaded.
        :param align_joint: Which joint to align. (e.g. SMPLJoint.ROOT). By default the root.
        :param rep: The rotation representation used in the input poses.
        :param use_pose_blendshape: Whether to use pose blendshape or not.
        :param joint_mask: If not None, local angle error, global angle error, and joint position error
                           for these joints will be calculated additionally.
        :param fps: Motion fps, by default 60.
        :param device: torch.device, cpu or cuda.
        """
        super(FullMotionEvaluator, self).__init__(
            official_model_file, rep, use_pose_blendshape, device=device
        )
        self.align_joint = 0 if align_joint is None else align_joint.value
        self.fps = fps
        self.joint_mask = joint_mask
        self.left_idx = [13, 16, 18, 20, 22]
        self.right_idx = [14, 17, 19, 21, 23]
        self.lower_idx = [1, 2, 4, 5, 7, 8]
        self.rest_idx = [3, 6, 9, 12, 15]

    def __call__(
        self,
        pose_global_p,
        pose_local_p,
        pose_local_t,
        xsens_pose=None,
        shape_p=None,
        shape_t=None,
        tran_p=None,
        tran_t=None,
    ):
        r"""
        Get the measured errors. The returned tensor in shape [10, 2] contains mean and std of:
          0.  Joint position error (align_joint position aligned).
          1.  Vertex position error (align_joint position aligned).
          2.  Joint local angle error (in degrees).
          3.  Joint global angle error (in degrees).
          4.  Predicted motion jerk (with global translation).
          5.  True motion jerk (with global translation).
          6.  Translation error (mean root translation error per second, using a time window size of 1s).
          7.  Masked joint position error (align_joint position aligned, zero if mask is None).
          8.  Masked joint local angle error. (in degrees, zero if mask is None).
          9.  Masked joint global angle error. (in degrees, zero if mask is None).

        :param pose_p: Predicted pose or the first pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param pose_t: True pose or the second pose in shape [batch_size, *] that can
                       reshape to [batch_size, num_joint, rep_dim].
        :param shape_p: Predicted shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param shape_t: True shape that can expand to [batch_size, 10]. Use None for the mean(zero) shape.
        :param tran_p: Predicted translations in shape [batch_size, 3]. Use None for zeros.
        :param tran_t: True translations in shape [batch_size, 3]. Use None for zeros.
        :return: Tensor in shape [10, 2] for the mean and std of all errors.
        """
        f = self.fps
        bs = pose_global_p.shape[0]
        ignored_joint_non_pelvis_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])
        ignored_joint_mask = torch.tensor([7, 8, 10, 11, 20, 21, 22, 23])
        pose_local_p[:,ignored_joint_mask] = pose_local_t[:,ignored_joint_mask]


        # pose_local_p, shape_p, tran_p = self._preprocess(pose_p, shape_p, tran_p)
        # pose_local_t, shape_t, tran_t = self._preprocess(pose_t, shape_t, tran_t)
        # pose_global_p, joint_p, vertex_p = self.model.forward_kinematics(
        #     pose_local_p, shape_p, tran_p, calc_mesh=True
        # )
        _, joint_p, vertex_p = self.model.forward_kinematics(
            pose_local_p, shape_p, tran_p, calc_mesh=True
        )
        pose_global_t, joint_t, vertex_t = self.model.forward_kinematics(
            pose_local_t, shape_t, tran_t, calc_mesh=True
        )


        pose_global_p[:,ignored_joint_mask] = pose_global_t[:,ignored_joint_mask]

        pose_global_nopelvis_p = pose_global_p.clone()
        pose_global_nopelvis_t = pose_global_t.clone()
        pose_global_nopelvis_p[:, ignored_joint_non_pelvis_mask] = pose_global_nopelvis_t[
            :, ignored_joint_non_pelvis_mask
        ]

        # vertex l2 loss (without pelvis alignment)
        ve = (vertex_p - vertex_t).norm(dim=2)  # N, J
        # joint position loss (without pelvis alignment)
        je = (joint_p - joint_t).norm(dim=2)  # N, J
        # local/global angle error
        pose_local_p = pose_local_p.reshape(-1,24,3,3)
        pose_local_t = pose_local_t.reshape(-1,24,3,3)

        lae = radian_to_degree(
            angle_between(pose_local_p, pose_local_t).view(bs, -1)
        )
        gae = radian_to_degree(
            angle_between(pose_global_p, pose_global_t).view(bs, -1)
        )  # N, J

        gae_nopelvis = radian_to_degree(
            angle_between(pose_global_nopelvis_p, pose_global_nopelvis_t).view(
                bs, -1
            )
        )


        pose_global_xsens_p = glb_mat_smpl_to_glb_mat_xsens(pose_global_p)
        if xsens_pose is not None:
            pose_global_xsens_t = xsens_pose
        else:   
            pose_global_xsens_t = glb_mat_smpl_to_glb_mat_xsens(pose_global_t)

        gae_xsens = radian_to_degree(
            angle_between(pose_global_xsens_p, pose_global_xsens_t).view(
                bs,-1
            )
        )
        ignored_joint_mask_xsens = torch.tensor([0, 10, 14, 17, 18, 21, 22])
        pose_global_xsens_nopelvis_p = pose_global_xsens_p.clone()
        pose_global_xsens_nopelvis_t = pose_global_xsens_t.clone()

        pose_global_xsens_nopelvis_p[:, ignored_joint_mask_xsens] = (
            pose_global_xsens_nopelvis_t[:, ignored_joint_mask_xsens]
        )
        gae_xsens_nopelvis = radian_to_degree(
            angle_between(
                pose_global_xsens_nopelvis_p, pose_global_xsens_nopelvis_t
            ).view(bs, -1)
        )

        # prediction의 jitter
        jkp = (
            (joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3])
            * (f**3)
        ).norm(
            dim=2
        )  # N, J

        # ground truth의 jitter
        jkt = (
            (joint_t[3:] - 3 * joint_t[2:-1] + 3 * joint_t[1:-2] - joint_t[:-3])
            * (f**3)
        ).norm(
            dim=2
        )  # N, J

        te = (
            (joint_p[f:, :1] - joint_p[:-f, :1]) - (joint_t[f:, :1] - joint_t[:-f, :1])
        ).norm(
            dim=2
        )  # N, 1

        mje = (
            je[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)
        )  # N, mJ
        mlae = (
            lae[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)
        )  # N, mJ

        mgae = (
            gae[:, self.joint_mask] if self.joint_mask is not None else torch.zeros(1)
        )  # N, mJ

        return torch.tensor(
            [
                [je.mean(), je.std(dim=0).mean()],
                [ve.mean(), ve.std(dim=0).mean()],
                [lae.mean(), lae.std(dim=0).mean()],
                [gae.mean(), gae.std(dim=0).mean()],
                [jkp.mean(), jkp.std(dim=0).mean()],
                [jkt.mean(), jkt.std(dim=0).mean()],
                [te.mean(), te.std(dim=0).mean()],
                [mje.mean(), mje.std(dim=0).mean()],
                [mlae.mean(), mlae.std(dim=0).mean()],
                [mgae.mean(), mgae.std(dim=0).mean()],
                [gae_nopelvis.mean(), gae_nopelvis.std(dim=0).mean()],
                [gae_xsens.mean(), gae_xsens.std(dim=0).mean()],
                [gae_xsens_nopelvis.mean(), gae_xsens_nopelvis.std(dim=0).mean()],
            ]
        )


def glb_mat_smpl_to_glb_mat_xsens(glb_full_pose_smpl):
    glb_full_pose_xsens = torch.eye(3).repeat(glb_full_pose_smpl.shape[0], 23, 1, 1)
    indices = [
        0,
        19,  # 1
        15,  # 2
        1,  # 3
        20,
        16,
        3,  # 6
        21,
        17,
        4,  # 9
        22,
        18,
        5,  # 12
        11,  # 13
        7,  # 14
        6,
        12,  # 16
        8,  # 17
        13,
        9,
        13,
        9,
        13,
        9,
    ]
    for idx, i in enumerate(indices):
        glb_full_pose_xsens[:, i, :] = glb_full_pose_smpl[:, idx, :]
    return glb_full_pose_xsens
