import os
import sys
import torch
import argparse
import pandas as pd
import segmentation_models_pytorch as smp

from data import CimatDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Load command arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path")
    parser.add_argument("num_epochs")
    args = parser.parse_args()
    print(args)

    # Use SLURM array environment variables to determine training and cross validation set number
    slurm_array_job_id = int(os.getenv("SLURM_ARRAY_JOB_ID"))
    slurm_array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    print(f"SLURM_ARRAY_JOB_ID: {slurm_array_job_id}")
    print(f"SLURM_ARRAY_TASK_ID: {slurm_array_task_id}")

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
        labels_path=labl_dir,dims
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
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load and configure model (segmentation_models_pytorch)
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights=None,
        classes=2,
        activation='sigmoid'
    )
    loss = smp.losses.SoftCrossEntropyLoss()
    metrics = [
        smp.metrics.accuracy    
    ]
    optimizer = torch.optim.Adam()

    # Training epochs
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device='CUDA',
        verbose=True,
    )
    cross_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device='CUDA',
        verbose=True,
    )

    # Train model
    for i in range(0, args.num_epochs):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = cross_epoch.run(cross_loader)
