r"""
    Train the pose estimation.
"""

import torch
from model.probip import ProbIP
from dataset import MotionTrainDataset
from utils.config import *

import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collate_fn_padd(batch):
    lengths = torch.tensor([t["pose"].shape[0] for t in batch])  # .to(device)

    new_batch = {}
    for key in batch[0].keys():
        new_batch[key] = [t[key] for t in batch]

        new_batch[key] = torch.nn.utils.rnn.pad_sequence(
            new_batch[key], batch_first=True
        )

    mask = torch.ones(new_batch["pose"].shape[:2])
    for i, l in enumerate(lengths):
        mask[i, l:] = 0

    return new_batch, lengths, mask


def main():
    name = "sensor5_hpwwl"
    modelIdx = sensor5Infohpwwl

    model = ProbIP(
        sensor=modelIdx.sensor_idx,
        reduced=modelIdx.reduced_idx,
        smpl_file_path=paths.smpl_m,
    ).to(device)

    
    model.train()
    amass_dataset = MotionTrainDataset(
        window_size=300,
    )
    dataloader = torch.utils.data.DataLoader(
        amass_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn_padd,
    )

    loss_logger = WandbLogger(project="probip", name=name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"model_log/{name}",
        save_top_k=3,
        monitor="train_epoch_loss",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = L.Trainer(
        max_epochs=500,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=loss_logger,
        accelerator="gpu",
        # strategy="ddp_find_unused_parameters_true",
    )
    trainer.fit(
        model=model,
        train_dataloaders=dataloader,
        # ckpt_path=last_checkpoint_path if os.path.exists(last_checkpoint_path) else None,
    )


if __name__ == "__main__":
    main()
