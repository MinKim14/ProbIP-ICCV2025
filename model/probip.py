import torch.nn

import articulate as art
import torch.nn as nn

import lightning as L
from model.ru_mamba import *
from tqdm import tqdm
from utils.b2_loss import trans_b2_loss

smpl_file = "SMPL_male.pkl"


class ProbIP(L.LightningModule):
    r"""
    Whole pipeline for pose and translation estimation.
    """

    def __init__(
        self,
        sensor=[18, 19, 4, 5, 15, 0],
        reduced=[1, 2, 3, 6, 9, 12, 13, 14, 16, 17],
        ignored=[7, 8, 10, 11, 20, 21, 22, 23],
        smpl_file_path="smpl_model/SMPL_male.pkl",
    ):

        super().__init__()
        n_imu = len(sensor) * 3 + len(sensor) * 9
        self.sensor = sensor
        self.reduced = reduced
        self.n_reduced = len(reduced)
        self.ignored = ignored

        self.pose_s1 = RUMamba(
            n_imu,
            5 * self.n_reduced * (3 * 3),
            128,
            exp_scale=2,
            n_reduced=self.n_reduced,
        )
        self.pose_s2 = RUMamba(
            n_imu + 5 * self.n_reduced * 9,
            self.n_reduced * (3 * 3),
            128,
            exp_scale=4,
            n_reduced=self.n_reduced,
        )
        self.pose_s3 = RUMamba(
            n_imu + 5 * self.n_reduced * 9,
            self.n_reduced * (3 * 3),
            128,
            exp_scale=6,
            n_reduced=self.n_reduced,
        )
        self.pose_s4 = RUMamba(
            n_imu + 5 * self.n_reduced * 9,
            self.n_reduced * (3 * 3 + 3),
            128,
            exp_scale=8,
            n_reduced=self.n_reduced,
        )
        self.posterior_layer_norm1 = nn.LayerNorm(self.n_reduced * (3 * 3 + 3))

        self.tran_b2 = MambaSimple(n_imu + self.n_reduced * (9 + 3), 3, 128)

        m = art.ParametricModel(smpl_file_path, device="cuda")
        self.m = m

        self.global_to_local_pose = m.inverse_kinematics_R
        self.local_to_global_pose = m.forward_kinematics_R

        self.cur_loss = 0
        self.num_step = 0

    def _reduced_glb_to_full_local(self, glb_reduced_pose, sensor):

        glb_reduced_pose = glb_reduced_pose.view(-1, self.n_reduced, 3, 3)
        sensor = sensor.view(-1, len(self.sensor), 3, 3)

        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(
            glb_reduced_pose.shape[0], 24, 1, 1
        )

        global_full_pose[:, self.reduced] = glb_reduced_pose
        global_full_pose[:, self.sensor] = sensor

        local_full_pose = self.global_to_local_pose(global_full_pose).view(-1, 24, 3, 3)
        local_full_pose[:, self.ignored] = torch.eye(3, device=local_full_pose.device)
        return global_full_pose, local_full_pose

    def training_step(self, batch, batch_idx):
        s, lengths, mask = batch
        bs, ws = s["pose"].shape[:2]
        pose = s["pose"]

        vacc, vori, trans = s["vacc"], s["vori"], s["trans"].to(self.device)
        vacc = vacc.reshape(vacc.shape[0], vacc.shape[1], -1).to(self.device)
        vori = vori.reshape(vori.shape[0], vori.shape[1], -1).to(self.device)
        pose_t, tran_t = pose.reshape(pose.shape[0], -1, 24, 3).to(
            self.device
        ), trans.to(self.device)
        pose_t = art.math.axis_angle_to_rotation_matrix(pose_t.reshape(-1, 3)).reshape(
            bs, ws, 24, 3, 3
        )
        (
            global_pose_t,
            _,
        ) = self.m.forward_kinematics(pose_t.reshape(-1, 24, 3, 3))
        global_pose_t = global_pose_t.reshape(bs, ws, 24, 3, 3)

        global_pose_axis = art.math.rotation_matrix_to_axis_angle(
            global_pose_t.reshape(-1, 24, 3, 3)
        ).reshape(bs, ws, 24, 3)

        input = torch.cat([vacc, vori], dim=-1)

        init_pose = global_pose_axis[:, 0]
        (
            [mode1, mode2, mode3, mode4],
            velocity,
            [posterior1, posterior2, posterior3, posterior4],
            [samples1, samples2, samples3, samples4],
        ) = self.forward(input, init_pose)

        loss_gAR1 = (
            -torch.matmul(
                mode1.reshape(bs, ws, self.n_reduced, 3, 3)[mask[:, :] == 1].view(
                    -1, 1, 9
                ),
                global_pose_t[:, :, self.reduced][mask[:, :] == 1].view(-1, 9, 1),
            )
            .view(-1)
            .mean()
        )
        loss_gAR2 = (
            -torch.matmul(
                mode2.reshape(bs, ws, self.n_reduced, 3, 3)[mask[:, :] == 1].view(
                    -1, 1, 9
                ),
                global_pose_t[:, :, self.reduced][mask[:, :] == 1].view(-1, 9, 1),
            )
            .view(-1)
            .mean()
        )
        loss_gAR3 = (
            -torch.matmul(
                mode3.reshape(bs, ws, self.n_reduced, 3, 3)[mask[:, :] == 1].view(
                    -1, 1, 9
                ),
                global_pose_t[:, :, self.reduced][mask[:, :] == 1].view(-1, 9, 1),
            )
            .view(-1)
            .mean()
        )
        loss_gAR4 = (
            -torch.matmul(
                mode4.reshape(bs, ws, self.n_reduced, 3, 3)[mask[:, :] == 1].view(
                    -1, 1, 9
                ),
                global_pose_t[:, :, self.reduced][mask[:, :] == 1].view(-1, 9, 1),
            )
            .view(-1)
            .mean()
        )

        loss_sample1 = torch.nn.MSELoss()(
            samples1[mask[:, :] == 1],
            global_pose_t.unsqueeze(3)[:, :, self.reduced].repeat(1, 1, 1, 5, 1, 1)[
                mask[:, :] == 1
            ],
        )

        loss_sample2 = torch.nn.MSELoss()(
            samples2[mask[:, :] == 1],
            global_pose_t.unsqueeze(3)[:, :, self.reduced].repeat(1, 1, 1, 5, 1, 1)[
                mask[:, :] == 1
            ],
        )

        loss_sample3 = torch.nn.MSELoss()(
            samples3[mask[:, :] == 1],
            global_pose_t.unsqueeze(3)[:, :, self.reduced].repeat(1, 1, 1, 5, 1, 1)[
                mask[:, :] == 1
            ],
        )

        loss_sample4 = torch.nn.MSELoss()(
            samples4[mask[:, :] == 1],
            global_pose_t.unsqueeze(3)[:, :, self.reduced].repeat(1, 1, 1, 5, 1, 1)[
                mask[:, :] == 1
            ],
        )

        loss_posterior1 = vmf_loss(
            posterior1[mask[:, :] == 1].reshape(-1, 3, 3),
            global_pose_t[mask[:, :] == 1][:, self.reduced].reshape(-1, 3, 3),
            1 + (1 / (math.exp(2) * 3)),
        )
        loss_posterior2 = vmf_loss(
            posterior2[mask[:, :] == 1].reshape(-1, 3, 3),
            global_pose_t[mask[:, :] == 1][:, self.reduced].reshape(-1, 3, 3),
            1 + (1 / (math.exp(4) * 3)),
        )

        loss_posterior3 = vmf_loss(
            posterior3[mask[:, :] == 1].reshape(-1, 3, 3),
            global_pose_t[mask[:, :] == 1][:, self.reduced].reshape(-1, 3, 3),
            1 + (1 / (math.exp(6) * 3)),
        )

        loss_posterior4 = vmf_loss(
            posterior4[mask[:, :] == 1].reshape(-1, 3, 3),
            global_pose_t[mask[:, :] == 1][:, self.reduced].reshape(-1, 3, 3),
            1 + (1 / (math.exp(8) * 3)),
        )

        global_pose_loss = loss_gAR1 + loss_gAR2 + loss_gAR3 + loss_gAR4

        samples_loss = loss_sample1 + loss_sample2 + loss_sample3 + loss_sample4
        posterior_loss = 0.3 * (
            loss_posterior1 + loss_posterior2 + loss_posterior3 + loss_posterior4
        )

        tran_b2_vel = velocity
        tran_b2_vel = tran_b2_vel.view(bs, ws, 3)
        tran_b2_vel_gt = trans[:, 1:] - trans[:, :-1]
        tran_b2_vel_gt = torch.cat(
            [torch.zeros(bs, 1, 3).to(self.device), tran_b2_vel_gt], dim=1
        )
        dip_mask = s["dip_mask"].to(self.device)
        dip_mask = dip_mask[:, 0]
        trans_mask = mask.clone()
        trans_mask[dip_mask == 1] = 0
        loss_B2 = trans_b2_loss(tran_b2_vel[:, :], tran_b2_vel_gt, trans_mask)

        loss = posterior_loss + loss_B2 + samples_loss + global_pose_loss

        self.cur_loss += loss.item()
        self.num_step += 1
        return loss

    def on_train_epoch_end(self):
        print("train epoch loss : ", self.cur_loss / self.num_step)
        self.log("train_epoch_loss", self.cur_loss / self.num_step)
        self.cur_loss = 0
        self.num_step = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=4e-4)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, imu, init_joint, test=False):
        bs, ws = imu.shape[:2]

        posterior1RuExp, posterior1Ru = self.pose_s1.forward(imu, init_joint)
        posterior1R, posterior1u = (
            posterior1RuExp[:, :, : self.n_reduced * 3 * 3],
            posterior1RuExp[:, :, self.n_reduced * 3 * 3 :],
        )

        num_samples = 5
        mode1, _, posterior1 = get_mf_mode_RS(
            posterior1R.reshape(-1, 3, 3),
            posterior1u.reshape(-1, 3),
            num_samples=num_samples,
        )
        if test:
            samples1_pass = (
                mode1[:, 0].unsqueeze(1).repeat(1, 5, 1, 1).reshape(bs, ws, -1)
            )
        else:
            samples1_pass = mode1[:, 1:].reshape(bs, ws, -1)
        samples1 = mode1[:, 1:].reshape(bs, ws, self.n_reduced, num_samples, 3, 3)
        mode1 = mode1[:, 0]

        posterior2RuExp, posterior2Ru = self.pose_s2.forward(
            torch.cat([imu, samples1_pass], dim=-1),
            init_joint,
        )
        posterior2R, posterior2u = (
            posterior2RuExp[:, :, : self.n_reduced * 3 * 3],
            posterior2RuExp[:, :, self.n_reduced * 3 * 3 :],
        )

        mode2, _, posterior2 = get_mf_mode_RS(
            posterior2R.reshape(-1, 3, 3),
            posterior2u.reshape(-1, 3),
            num_samples=num_samples,
        )
        if test:
            samples2_pass = (
                mode2[:, 0].unsqueeze(1).repeat(1, 5, 1, 1).reshape(bs, ws, -1)
            )
        else:
            samples2_pass = mode2[:, 1:].reshape(bs, ws, -1)
        samples2 = mode2[:, 1:].reshape(bs, ws, self.n_reduced, num_samples, 3, 3)
        mode2 = mode2[:, 0]

        posterior1 = posterior1.reshape(bs, ws, -1, 3, 3)
        posterior2 = posterior2.reshape(bs, ws, -1, 3, 3)

        posterior3RuExp, posterior3Ru = self.pose_s3.forward(
            torch.cat([imu, samples2_pass], dim=-1),
            init_joint,
        )
        posterior3R, posterior3u = (
            posterior3RuExp[:, :, : self.n_reduced * 3 * 3],
            posterior3RuExp[:, :, self.n_reduced * 3 * 3 :],
        )

        mode3, _, posterior3 = get_mf_mode_RS(
            posterior3R.reshape(-1, 3, 3),
            posterior3u.reshape(-1, 3),
            num_samples=num_samples,
        )
        if test:
            samples3_pass = (
                mode3[:, 0].unsqueeze(1).repeat(1, 5, 1, 1).reshape(bs, ws, -1)
            )
        else:
            samples3_pass = mode3[:, 1:].reshape(bs, ws, -1)
        samples3 = mode3[:, 1:].reshape(bs, ws, self.n_reduced, num_samples, 3, 3)
        mode3 = mode3[:, 0]
        posterior3 = posterior3.reshape(bs, ws, -1, 3, 3)

        posterior4RuExp, posterior4Ru = self.pose_s4.forward(
            torch.cat([imu, samples3_pass], dim=-1),
            init_joint,
        )
        posterior4R, posterior4u = (
            posterior4RuExp[:, :, : self.n_reduced * 3 * 3],
            posterior4RuExp[:, :, self.n_reduced * 3 * 3 :],
        )

        mode4, _, posterior4 = get_mf_mode_RS(
            posterior4R.reshape(-1, 3, 3),
            posterior4u.reshape(-1, 3),
            num_samples=num_samples,
        )

        samples4 = mode4[:, 1:].reshape(bs, ws, self.n_reduced, num_samples, 3, 3)

        mode4 = mode4[:, 0]
        posterior4 = posterior4.reshape(bs, ws, -1, 3, 3)

        velocity = self.tran_b2.forward(
            torch.cat((imu, self.posterior_layer_norm1(posterior4Ru)), dim=2),
            init_joint,
        )

        return (
            [mode1, mode2, mode3, mode4],
            velocity,
            [posterior1, posterior2, posterior3, posterior4],
            [samples1, samples2, samples3, samples4],
        )

    def reset(self):
        self.pose_s1.reset_state()
        self.pose_s2.reset_state()
        self.pose_s3.reset_state()
        self.pose_s4.reset_state()
        self.tran_b2.reset_state()

    @torch.no_grad()
    def forward_fullseq(self, imu, init_joint):
        (
            [mode1, mode2, mode3, mode4],
            velocity,
            [posterior1, posterior2, posterior3, posterior4],
            [samples1, samples2, samples3, samples4],
        ) = self.forward(imu, init_joint, test=True)

        bs, ws = imu.shape[:2]

        mode4 = mode4.reshape(bs, ws, self.n_reduced, 3, 3)
        samples4 = samples4.reshape(bs, ws, self.n_reduced, 5, 3, 3)
        vori = imu[:, :, len(self.sensor) * 3 :].reshape(1, -1, len(self.sensor), 3, 3)
        global_reduced_pose = mode4.squeeze()

        velocity = velocity.squeeze().cpu()
        global_full_pose, local_full_pose = self._reduced_glb_to_full_local(global_reduced_pose.cpu(), vori.cpu())

        return global_full_pose, local_full_pose, self.velocity_to_root_position(velocity)

    @torch.no_grad()
    def forward_iter(self, imu, init_joint):
        self.reset()
        total_velocity = []
        total_global_pose_prediction = []
        total_local_pose_prediction = []

        pbar = tqdm(range(imu.shape[1]))
        import time

        times = []
        for i, s in enumerate(pbar):
            if i != 0:
                init_joint = None
            st = time.time()
            mode4, velocity = self.step(imu[:, i], init_joint)
            times.append(time.time() - st)

            total_velocity.append(velocity)
            global_pose_pose, local_pose_prediction = self._reduced_glb_to_full_local(
                mode4.cpu(), imu[:, i, len(self.sensor) * 3 :].cpu()
            )
            total_global_pose_prediction.append(global_pose_pose.detach().cpu())
            total_local_pose_prediction.append(local_pose_prediction.detach().cpu())

        print("average time : ", sum(times) / len(times))
        print("inference Hz : ", 1 / (sum(times) / len(times)))

        total_global_pose_prediction = torch.stack(total_global_pose_prediction, dim=0).squeeze()
        total_local_pose_prediction = torch.stack(total_local_pose_prediction, dim=0).squeeze()
        total_velocity = torch.stack(total_velocity, dim=0).squeeze()
        return total_global_pose_prediction, total_local_pose_prediction, total_velocity

    @torch.no_grad()
    def step(self, imu, init_joint=None):

        posterior1RuExp, _ = self.pose_s1.step(imu, init_joint)
        posterior1R, posterior1u = (
            posterior1RuExp[:, : self.n_reduced * 3 * 3],
            posterior1RuExp[:, self.n_reduced * 3 * 3 :],
        )
        diag_mat = (
            1e-5
            * torch.eye(3, device=posterior1R.device)
            .unsqueeze(0)
            .expand(1, -1, -1)
            .clone()
        )
        U, S, Vh = torch.linalg.svd(
            posterior1R.reshape(-1, 3, 3) + diag_mat, full_matrices=True
        )

        with torch.no_grad():
            det_u_v = torch.linalg.det(torch.bmm(U, Vh.transpose(1, 2)))

        det_modify_mat = (
            torch.eye(3, device=U.device)
            .unsqueeze(0)
            .expand(U.shape[0], -1, -1)
            .clone()
        )
        det_modify_mat[:, 2, 2] = det_u_v

        mode1 = torch.bmm(torch.bmm(U, det_modify_mat), Vh)
        samples1_pass = mode1.unsqueeze(1).repeat(1, 5, 1, 1).reshape(1, -1)
        posterior2RuExp, posterior2Ru = self.pose_s2.step(
            torch.cat([imu, samples1_pass], dim=-1), init_joint
        )
        posterior2R, _ = (
            posterior2RuExp[:, : self.n_reduced * 3 * 3],
            posterior2RuExp[:, self.n_reduced * 3 * 3 :],
        )
        U, S, Vh = torch.linalg.svd(
            posterior2R.reshape(-1, 3, 3) + diag_mat, full_matrices=True
        )

        with torch.no_grad():
            det_u_v = torch.linalg.det(torch.bmm(U, Vh.transpose(1, 2)))

        det_modify_mat = (
            torch.eye(3, device=U.device)
            .unsqueeze(0)
            .expand(U.shape[0], -1, -1)
            .clone()
        )
        det_modify_mat[:, 2, 2] = det_u_v

        mode2 = torch.bmm(torch.bmm(U, det_modify_mat), Vh)
        samples2_pass = mode2.unsqueeze(1).repeat(1, 5, 1, 1).reshape(1, -1)

        posterior3RuExp, _ = self.pose_s3.step(
            torch.cat([imu, samples2_pass], dim=-1), init_joint
        )
        posterior3R, _ = (
            posterior3RuExp[:, : self.n_reduced * 3 * 3],
            posterior3RuExp[:, self.n_reduced * 3 * 3 :],
        )
        U, S, Vh = torch.linalg.svd(
            posterior3R.reshape(-1, 3, 3) + diag_mat, full_matrices=True
        )

        with torch.no_grad():
            det_u_v = torch.linalg.det(torch.bmm(U, Vh.transpose(1, 2)))

        det_modify_mat = (
            torch.eye(3, device=U.device)
            .unsqueeze(0)
            .expand(U.shape[0], -1, -1)
            .clone()
        )
        det_modify_mat[:, 2, 2] = det_u_v

        mode3 = torch.bmm(torch.bmm(U, det_modify_mat), Vh)
        samples3_pass = mode3.unsqueeze(1).repeat(1, 5, 1, 1).reshape(1, -1)

        posterior4RuExp, posterior4Ru = self.pose_s4.step(
            torch.cat([imu, samples3_pass], dim=-1), init_joint
        )
        posterior4R, _ = (
            posterior4RuExp[:, : self.n_reduced * 3 * 3],
            posterior4RuExp[:, self.n_reduced * 3 * 3 :],
        )
        U, S, Vh = torch.linalg.svd(
            posterior4R.reshape(-1, 3, 3) + diag_mat, full_matrices=True
        )

        with torch.no_grad():
            det_u_v = torch.linalg.det(torch.bmm(U, Vh.transpose(1, 2)))

        det_modify_mat = (
            torch.eye(3, device=U.device)
            .unsqueeze(0)
            .expand(U.shape[0], -1, -1)
            .clone()
        )
        det_modify_mat[:, 2, 2] = det_u_v

        mode4 = torch.bmm(torch.bmm(U, det_modify_mat), Vh)
        velocity = self.tran_b2.step(
            torch.cat((imu, self.posterior_layer_norm1(posterior4Ru)), dim=-1),
            init_joint,
        )
        return mode4, velocity
        # return mode4.detach().cpu(), velocity.detach().cpu()

    @staticmethod
    def velocity_to_root_position(velocity):
        return torch.stack(
            [velocity[: i + 1].sum(dim=0) for i in range(velocity.shape[0])]
        )
