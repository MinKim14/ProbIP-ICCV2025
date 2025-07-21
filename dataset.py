import torch
import os
import os.path as osp
from utils.config import paths


class MotionTrainDataset:
    def __init__(self, window_size):
        self.window_size = window_size

        self.pose = []
        self.trans = []
        self.vacc = []
        self.vori = []
        xsens_dir = osp.join(paths.work_dir, "train")
        self.dip_mask = []
        for folder in os.listdir(xsens_dir):
            pose = torch.load(osp.join(xsens_dir, folder, "local_smpl_pose.pt"))
            tran = torch.load(osp.join(xsens_dir, folder, "tran.pt"))
            vacc = torch.load(osp.join(xsens_dir, folder, "vacc.pt"))
            vori = torch.load(osp.join(xsens_dir, folder, "vrot.pt"))

            for i in range(len(pose)):
                self.pose.append(pose[i].reshape(-1, 24, 3))
                self.trans.append(tran[i])
                self.vacc.append(vacc[i].reshape(-1, 6, 3))
                self.vori.append(vori[i].reshape(-1, 6, 3, 3))
                if folder == "dip":
                    self.dip_mask.append(1)
                else:
                    self.dip_mask.append(0)

        self.dip_mask = torch.tensor(self.dip_mask)

        self.create_window_idx()

    def create_window_idx(self):
        self.window_idx = []
        for i in range(len(self.pose)):
            if len(self.pose[i]) > self.window_size:
                for j in range(len(self.pose[i]) // self.window_size + 1):
                    ed = min((j + 1) * self.window_size, len(self.pose[i]))
                    if ed - j * self.window_size > 3:
                        self.window_idx.append((i, j * self.window_size, ed))
            else:
                self.window_idx.append((i, 0, len(self.pose[i])))

    def __len__(self):
        return len(self.window_idx)

    def __getitem__(self, idx):
        i, st, ed = self.window_idx[idx]

        item = {
            "pose": self.pose[i][st:ed],
            "trans": self.trans[i][st:ed],
            "vacc": self.vacc[i][st:ed],
            "vori": self.vori[i][st:ed],
            "dip_mask": self.dip_mask[i].reshape(-1).repeat(ed - st),
        }

        return item


class MotionDatasetXsensTest:
    def __init__(self, name):

        data_dir = osp.join(paths.work_dir, "test")
        motion_dir = osp.join(data_dir, name)

        self.pose = torch.load(osp.join(motion_dir, "local_smpl_pose.pt"), weights_only=False)
        self.xsens_pose = torch.load(osp.join(motion_dir, "global_xsens_pose.pt"), weights_only=False)
        self.vacc = torch.load(osp.join(motion_dir, "vacc.pt"), weights_only=False)
        self.vori = torch.load(osp.join(motion_dir, "vrot.pt"), weights_only=False)
        self.trans = torch.load(osp.join(motion_dir, "tran.pt"), weights_only=False)

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, i):
        item = {
            "pose": self.pose[i],
            "xsens_pose": self.xsens_pose[i],
            "trans": self.trans[i],
            "vacc": self.vacc[i],
            "vori": self.vori[i],
            "seq_idx": i,
        }
        return item


class MotionDatasetDipFull:
    def __init__(self):
        data_dict = torch.load(osp.join(paths.dipimu_dir, "dip_test.pt"), weights_only=False)

        self.pose = data_dict["pose"]
        self.trans = data_dict["tran"]
        self.vacc = data_dict["acc"]
        self.vori = data_dict["ori"]

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, i):
        item = {
            "pose": self.pose[i],
            "trans": self.trans[i],
            "vacc": self.vacc[i],
            "vori": self.vori[i],
            "seq_idx": i,
        }
        return item


class MotionDatasetTotalCaptureReal:
    def __init__(self):
        data_dict = torch.load("dataset/test.pt", weights_only=False)

        self.pose = data_dict["pose"]
        self.vacc = data_dict["acc"]
        self.vori = data_dict["ori"]

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, i):
        item = {
            "pose": self.pose[i],
            "vacc": self.vacc[i],
            "vori": self.vori[i],
            "seq_idx": i,
        }
        return item
