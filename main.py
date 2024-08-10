import os
import time
import torch
import argparse
import pandas as pd
import segmentation_models_pytorch as smp

from data import CimatDataset
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Configure device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    valid_set = pd.read_csv(os.path.join(cross_dir, f"cross{cross_file}.csv"))
    print(f"Training CSV file length: {len(train_set)}")
    print(f"Validation CSV file length: {len(valid_set)}")

    # Load generators
    train_keys = train_set["key"]
    valid_keys = valid_set["key"]
    train_dataset = CimatDataset(
        keys=train_keys,
        features_path=feat_dir,
        features_ext=".tiff",
        features_channels=feat_channels,
        labels_path=labl_dir,
        labels_ext=".pgm",
        dimensions=[224, 224, 3],
    )

    valid_dataset = CimatDataset(
        keys=valid_keys,
        features_path=feat_dir,
        features_ext=".tiff",
        features_channels=feat_channels,
        labels_path=labl_dir,
        labels_ext=".pgm",
        dimensions=[224, 224, 3],
    )
    print(f"Training dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(valid_dataset)}")

    train_dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=12
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=2, shuffle=False, num_workers=4
    )

    # Load and configure model (segmentation_models_pytorch)
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, classes=1, activation="sigmoid"
    ).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Steps per epoch
    trainSteps = len(train_dataset) // 8
    validSteps = len(valid_dataset) // 2

    # training history
    history = {"train_loss": [], "valid_loss": []}

    # Training
    print("[INFO] training the network...")
    startTime = time.time()
    for e in tqdm(range(int(args.num_epochs))):
        model.train()

        # initialize total training and validation loss
        totalTrainLoss = 0
        totalValidLoss = 0

        # loop over the training set
        for i, (x, y) in enumerate(train_dataloader):
            # send the input to the device
            (x, y) = (x.to(DEVICE), y.to(DEVICE))

            # forward pass
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add total training loss
            totalTrainLoss += loss

        # switch off autograd
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()

            # loop over the validation set
            for x, y in valid_dataloader:
                (x, y) = (x.to(DEVICE), y.to(DEVICE))

                pred = model(x)
                totalValidLoss += loss_fn(pred, y)

        # calculate avg loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValidLoss = totalValidLoss / validSteps

        history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        history["valid_loss"].append(avgValidLoss.cpu().detach().numpy())

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, int(args.num_epochs)))
        print(
            "Train loss: {:.6f}, Valid loss: {:.4f}".format(avgTrainLoss, avgValidLoss)
        )

    # display total time
    endTime = time.time()
    print(
        "[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime
        )
    )
