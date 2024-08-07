import os
import sys
import torch
import argparse
import pandas as pd
import segmentation_models_pytorch as smp

from data import CimatDataset
from torch import nn, optim
from torch.utils.data import DataLoader
from catalyst import dl, utils

if __name__ == "__main__":
    # Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("num_epochs")
    args = parser.parse_args()
    print(args)

    # Use SLURM array environment variables to determine training and cross validation set number
    # slurm_array_job_id = int(os.getenv("SLURM_ARRAY_JOB_ID"))
    # slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
    # print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")
    slurm_array_task_id = 1

    train_file = "{:02}".format(slurm_array_task_id)
    cross_file = "{:02}".format(slurm_array_task_id)
    print(f"Train file: {train_file}")
    print(f"Cross file: {cross_file}")

    # Configure directories
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(
        home_dir,
        "data",
        "projects",
        "consorcio-ia",
        "data",
        "oil_spills_17",
        "augmented_dataset",
    )
    feat_dir = os.path.join(data_dir, "features")
    labl_dir = os.path.join(data_dir, "labels")
    train_dir = os.path.join(data_dir, "learningCSV", "trainingFiles")
    cross_dir = os.path.join(data_dir, "learningCSV", "crossFiles")

    # Check if results path exists
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path, exist_ok=True)

    # Features to use
    feat_channels = ["ORIGIN", "ORIGIN", "VAR"]

    # Load CSV key files
    train_set = pd.read_csv(os.path.join(train_dir, f"train{train_file}.csv"))
    cross_set = pd.read_csv(os.path.join(cross_dir, f"cross{cross_file}.csv"))
    print(f"Training CSV file length: {len(train_set)}")
    print(f"Cross validation CSV file length: {len(cross_set)}")

    # Load generators
    train_keys = train_set["key"]
    cross_keys = cross_set["key"]
    train_dataset = CimatDataset(
        keys=train_keys,
        features_path=feat_dir,
        features_ext=".tiff",
        features_channels=feat_channels,
        labels_path=labl_dir,
        labels_ext=".pgm",
        dimensions=[224, 224, 3],
    )

    cross_dataset = CimatDataset(
        keys=cross_keys,
        features_path=feat_dir,
        features_ext=".tiff",
        features_channels=feat_channels,
        labels_path=labl_dir,
        labels_ext=".pgm",
        dimensions=[224, 224, 3],
    )
    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Cross validation dataset length: {len(cross_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=12
    )
    cross_dataloader = DataLoader(
        cross_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    # Load and configure model (segmentation_models_pytorch)
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, classes=1, activation="sigmoid"
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loaders = {"train": train_dataloader, "valid": cross_dataloader}

    # Training
    runner = dl.SupervisedRunner(
        input_key="features", output_key="logits", target_key="targets", loss_key="loss"
    )
    # Model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        num_epochs=int(args.num_epochs),
        logdir="./logs",
        valid_loader="valid",
        valid_metric="loss",
        minimize_valid_metric=True,
        verbose=True,
    )
