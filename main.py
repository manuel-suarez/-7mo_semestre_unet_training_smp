import os
import time
import torch
import argparse
import pandas as pd
import lightning as L
import segmentation_models_pytorch as smp

from data import CimatDataset
from dataloaders import prepare_dataloaders
from module import CimatModule
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import CSVLogger

if __name__ == "__main__":
    # Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("logs_path")
    parser.add_argument("model_name")
    parser.add_argument("encoder_name")
    parser.add_argument("num_epochs")
    args = parser.parse_args()
    print(args)

    # Use SLURM array environment variables to determine training and cross validation set number
    slurm_array_job_id = int(os.getenv("SLURM_ARRAY_JOB_ID"))
    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    slurm_job_nodelist = os.getenv("SLURM_JOB_NODELIST")
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    slurm_job_name = os.getenv("SLURM_JOB_NAME")
    slurm_job_start_time = os.getenv("SLURM_JOB_START_TIME")
    print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
    print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
    print(f"SLURM_JOB_NODELIST: {slurm_job_nodelist}")
    print(f"SLURM_JOB_ID: {slurm_job_id}")
    print(f"SLURM_JOB_NAME: {slurm_job_name}")
    print(f"SLURM_JOB_START_TIME: {slurm_job_start_time}")

    train_file = "{:02}".format(slurm_array_task_id)
    cross_file = "{:02}".format(slurm_array_task_id)
    num_epochs = "{:02}".format(int(args.num_epochs))
    print(f"Train file: {train_file}")
    print(f"Cross file: {cross_file}")

    # Check if results path exists
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)
    # Configure directories
    home_dir = os.path.expanduser("~")
    # Features to use
    feat_channels = ["ORIGIN", "ORIGIN", "VAR"]
    # Dataloaders
    train_dataloader, valid_dataloader = prepare_dataloaders(
        home_dir, feat_channels, train_file, cross_file
    )
    smp_archs = {"unet": smp.Unet, "linknet": smp.Linknet, "fpn": smp.FPN}

    # Load and configure model (segmentation_models_pytorch)
    model_name = args.model_name
    encoder_name = args.encoder_name
    model_arch = smp_archs[model_name]
    model = model_arch(
        encoder_name=encoder_name, encoder_weights=None, classes=1, activation="sigmoid"
    )
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Check logs path
    logs_path = os.path.join(
        args.log_path,
        model_name,
        encoder_name,
        "dataset17_oov",
        f"train{train_file}_valid{cross_file}_epochs{num_epochs}",
    )
    logger = CSVLogger(logs_path)
    module = CimatModule(model, optimizer, loss_fn)
    trainer = L.Trainer(
        max_epochs=int(args.num_epochs), devices=2, accelerator="gpu", logger=logger
    )
    # Training
    print("[INFO] training the network...")
    startTime = time.time()
    trainer.fit(
        model=module,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
    # display total time
    endTime = time.time()
    print(
        "[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime
        )
    )
