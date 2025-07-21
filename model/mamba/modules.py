"""
This Python file contains functions for working with matrix-Fisher distributions and related sampling
methods on 3D rotation matrices. It includes efficient implementations for fast batch sampling,
mode calculations, and handling rotations and uncertainties.

The code is a modified implementation based on the following sources:
- https://github.com/akashsengupta1997/HierarchicalProbabilistic3DHuman
- https://github.com/tylee-fdcl/Matrix-Fisher-Distribution

References:
- For theoretical details, see the related paper: https://arxiv.org/pdf/1310.8110.pdf

Functions:
- Fast batch sampling from matrix-Fisher distributions.
- Sampling from a batchwise Bingham distribution with diagonal matrix parameters.
- Mode and sample generation from rotation and uncertainty values in the RU-Mamba block.
"""

import torch
import numpy as np


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
    - quat: (B, 4) in (w, x, y, z) representation.

    Returns:
    - (B, 3, 3), rotation matrix corresponding to the quaternion.
    """
    norm_quat = quat
    norm_quat = norm_quat / (norm_quat.norm(p=2, dim=1, keepdim=True) + 1e-8)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


def batch_bingham_sampling_torch(
    A,
    num_samples,
    Omega=None,
    Gaussian_std=None,
    b=1.5,
    M_star=None,
    oversampling_ratio=8,
):
    """
    This function samples from a batchwise Bingham distribution using a 4x4 matrix parameter A,
    with the assumption that A is a diagonal matrix (a requirement for matrix-Fisher sampling).
    If a rejection occurs during sampling, the function returns the mode of the distribution to ensure fast execution.

    The original implementation was adapted from:
    https://github.com/akashsengupta1997/HierarchicalProbabilistic3DHuman
    """

    assert A.min() >= 0
    batch_size = A.shape[0]
    Gaussian_std = Gaussian_std.unsqueeze(1).expand(
        -1, num_samples * oversampling_ratio, -1
    )

    accepted_samples = torch.zeros((batch_size, num_samples, 4), device=A.device)

    samples_obtained = False
    while not samples_obtained:
        eps = torch.randn(
            batch_size, num_samples * oversampling_ratio, 4, device=A.device
        ).float()

        y = Gaussian_std * eps

        samples = y / torch.norm(
            y, dim=2, keepdim=True
        )  # (num_samples * oversampling_ratio, 4)
        with torch.no_grad():
            p_Bing_star = torch.exp(
                -torch.einsum("mbn,mn,mbn->mb", samples, A, samples)
            )  # (num_samples * oversampling_ratio,)
            p_ACG_star = torch.einsum("mbn,mn,mbn->mb", samples, Omega, samples) ** (
                -2
            )  # (num_samples * oversampling_ratio,)

            w = torch.rand(
                batch_size, num_samples * oversampling_ratio, device=A.device
            )

            accept_vector = w < p_Bing_star / (
                M_star * p_ACG_star
            )  # (num_samples * oversampling_ratio,)

            num_accepted = accept_vector.sum(dim=-1)
            splitted_tensor = torch.split(samples[accept_vector], num_accepted.tolist())

            accepted_samples_pad = torch.nn.utils.rnn.pad_sequence(
                splitted_tensor, batch_first=True
            ).to(A.device)
            num_accepted = torch.min(
                num_accepted.min(),
                torch.ones(1, device=num_accepted.device) * num_samples,
            ).long()
        if num_accepted.item() >= num_samples:
            accepted_samples[:, : num_accepted.item()] = accepted_samples_pad[
                :, : num_accepted.item(), :
            ]
            num_accepted = num_accepted.repeat(
                batch_size,
            )
            samples_obtained = True

    return accepted_samples, num_accepted


def batch_mf_sampling_torch(
    pose_U,
    pose_S,
    pose_V,
    num_samples,
    mode,
    b=1.5,
    sample_on_cpu=False,
):
    """
    This function performs fast batch sampling from matrix-Fisher distributions defined over 3D rotation matrices.
    The original implementation is adapted from the repository:
    https://github.com/akashsengupta1997/HierarchicalProbabilistic3DHuman

    For a comprehensive theoretical background, refer to the paper:
    https://arxiv.org/pdf/1310.8110.pdf

    Another useful source for implementation details can be found at:
    https://github.com/tylee-fdcl/Matrix-Fisher-Distribution
    """
    batch_size = pose_U.shape[0]

    # Proper SVD
    with torch.no_grad():
        detU, detV = torch.det(pose_U.detach().cpu()).to(pose_U.device), torch.det(
            pose_V.detach().cpu()
        ).to(pose_V.device)
    pose_U_proper = pose_U.clone()
    pose_S_proper = pose_S.clone()
    pose_V_proper = pose_V.clone()

    pose_S_proper[:, 2] *= detU * detV  # Proper singular values: s3 = s3 * det(UV)
    pose_U_proper[:, :, 2] *= detU.unsqueeze(-1)  # Proper U = U diag(1, 1, det(U))
    pose_V_proper[:, :, 2] *= detV.unsqueeze(-1)

    sample_device = pose_S_proper.device
    bingham_A = torch.zeros(batch_size, 4, device=sample_device)
    bingham_A[:, 1] = 2 * (pose_S_proper[:, 1] + pose_S_proper[:, 2])
    bingham_A[:, 2] = 2 * (pose_S_proper[:, 0] + pose_S_proper[:, 2])
    bingham_A[:, 3] = 2 * (pose_S_proper[:, 0] + pose_S_proper[:, 1])
    bingham_A = torch.nn.ReLU()(bingham_A)
    Omega = (
        torch.ones(batch_size, 4, device=bingham_A.device) + 2 * bingham_A / b
    )  # sample from ACG(Omega) with Omega = I + 2A/b.
    Gaussian_std = Omega ** (-0.5)  # Sigma^0.5 = (Omega^-1)^0.5 = Omega^-0.5
    M_star = np.exp(-(4 - b) / 2) * (
        (4 / b) ** 2
    )  # Bound for rejection sampling: Bing(A) <= Mstar(b)ACG(I+2A/b)

    pose_quat_samples_batch = torch.zeros(
        batch_size, num_samples, 4, device=pose_U.device
    ).float()
    pose_quat_samples_batch, num_accepted = batch_bingham_sampling_torch(
        A=bingham_A,
        num_samples=num_samples,
        Omega=Omega,
        Gaussian_std=Gaussian_std,
        b=b,
        M_star=M_star,
        oversampling_ratio=8,
    )

    pose_R_samples_batch = quat_to_rotmat(
        quat=pose_quat_samples_batch.view(-1, 4)
    ).view(batch_size, num_samples, 3, 3)

    pose_R_samples_batch = torch.matmul(
        pose_U_proper[:, None, :, :],
        torch.matmul(
            pose_R_samples_batch,
            pose_V_proper[:, None, :, :].transpose(dim0=-1, dim1=-2),
        ),
    )
    pose_R_samples_batch[num_accepted < num_samples] = (
        mode[num_accepted < num_samples].unsqueeze(1).repeat(1, num_samples, 1, 1)
    )

    return pose_R_samples_batch


def get_mf_mode_RS(params, u, num_samples=0, avg_S=False):
    """
    This function calculates the mode and generates samples from the rotation and uncertainty values
    obtained from the RU-Mamba block of a matrix-Fisher distribution. The process involves computing
    the singular value decomposition (SVD) with the components: U, S, Vh such that R is derived, and
    the result scales the uncertainty vector u as S * u.
    """
    batch_size = params.shape[0]
    diag_mat = (
        1e-5
        * torch.eye(3, device=params.device)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .clone()
    )

    U, S, Vh = torch.linalg.svd(params + diag_mat, full_matrices=True)
    u = S * u

    posterior = torch.bmm(U, torch.diag_embed(u)).bmm(Vh)

    with torch.no_grad():
        dist0 = torch.norm(S[:, 0:1] - S[:, 1:2], dim=-1).unsqueeze(-1)
        dist1 = torch.norm(S[:, 1:2] - S[:, 2:3], dim=-1).unsqueeze(-1)
        dist2 = torch.norm(S[:, 2:3] - S[:, 0:1], dim=-1).unsqueeze(-1)
        dist = torch.cat([dist0, dist1, dist2], dim=-1)
        min_dist = torch.amin(dist, dim=-1)

    mask = min_dist > 0

    with torch.no_grad():
        det_u_v = torch.linalg.det(torch.bmm(U, Vh.transpose(1, 2)))

    det_modify_mat = (
        torch.eye(3, device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    )
    det_modify_mat[:, 2, 2] = det_u_v

    mode = torch.bmm(torch.bmm(U, det_modify_mat), Vh)

    if num_samples > 0:
        V = Vh.transpose(-2, -1)

        samples = batch_mf_sampling_torch(
            U,
            u,
            V,
            num_samples,
            mode,
            b=1.5,
        )
        return (
            torch.concat((mode.unsqueeze(1), samples), dim=1),
            mask,
            posterior,
        )
    return mode, mask, posterior
