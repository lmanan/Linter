import os
from pprint import pprint

import torch
import yaml
from linter.criterions import get_loss
from linter.datasets import get_dataset
from linter.models import get_model
from linter.utils import get_logger
from tqdm import tqdm



def train(experiment_config):
    if not os.path.exists("models"):
        os.makedirs("models")

    dataset_config = experiment_config["dataset_config"]

    # Create train dataset
    pprint(f"Dataset Config: {dataset_config}")
    print("+"*20)

    # create train dataset
    train_dataset = get_dataset(
        data_dir=dataset_config["params"]["train"]["params"]["data_dir"],
        crop_size=int(dataset_config["params"]["train"]["params"]["crop_size"]),
    )

    # create train dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=int(dataset_config["params"]["batch_size"]),
        drop_last=True,
        num_workers=int(dataset_config["params"]["batch_size"])*2,
        pin_memory=True,
    )

    # Create model
    model_config = experiment_config["model_config"]
    print(f"Model Config: {model_config}")
    print("+" * 20)

    # set model
    model = get_model(
        dd_config=model_config["params"]["dd_config"],
        embed_dim=model_config["params"]["embed_dim"],
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in model is {pytorch_total_params}")

    # set device
    train_config = experiment_config["train_config"]
    print(f"Train Config: {train_config}")
    device = torch.device(train_config["params"]["device"])
    model = model.to(device)

    # set loss
    loss_config = experiment_config["loss_config"]
    print(f"Loss Config: {loss_config}")
    print("+" * 20)
    criterion = get_loss(
        loss_config["params"]["disc_start"],
        loss_config["params"]["kl_weight"],
        loss_config["params"]["disc_weight"],
    )

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config["params"]["initial_learning_rate"]),
    )

    # call `train_iteration`
    for iteration, sample in enumerate(tqdm(train_dataloader)):
        batch = sample["image"] # B C H W
        batch = batch.to(device)
        train_loss, prediction = train_iteration(
            batch, model=model, criterion=criterion, optimizer=optimizer, device=device
        )
        print(f"===> train loss: {train_loss:.6f}")



def train_iteration(batch, model, criterion, optimizer, device):
    model.train()
    prediction = model(batch.to(device))
    reconstruction, posterior = prediction
    (loss,) = criterion(batch, reconstruction, posterior, idx)
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), prediction


def save_model(state, iteration, is_lowest):
    pass


def save_snapshot(batch, prediction, iteration):
    pass


if __name__ == "__main__":
    with open("linter/configs/autoencoder/autoencoder_kl_4x4x4.yaml") as stream:
        try:
            experiment_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    pprint(f"Experiment Config is \n{experiment_config}")
    print("+"*20)
    train(experiment_config)
