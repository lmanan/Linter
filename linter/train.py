import os
import torch
import yaml
from tqdm import tqdm

from linter.criterions import get_loss
from linter.datasets import get_dataset
from linter.models import get_model
from linter.utils import get_logger


def train(experiment_config):
    if not os.path.exists("models"):
        os.makedirs("models")

    dataset_config = experiment_config['dataset_config']

    # Create train dataset
    print('+' * 20)
    print(f"Dataset Config: {dataset_config}")

    # create train dataset
    train_dataset = get_dataset(
        data_dir=dataset_config['data_dir'],
        patch_size=int(dataset_config['patch_size']),
        )

    # Create train dataloader
    train_config = experiment_config['train_config']
    print('+'*20)
    print(f"Train Config: {train_config}")

    # create train dataloader
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size= int(train_config['batch_size']),
        drop_last= True,
        num_workers= int(train_config['num_workers']),
        pin_memory=True,
    )

    # Create model
    model_config = experiment_config['model_config']
    print('+' * 20)
    print(f"Model Config: {model_config}")

    # set model
    model = get_model(
        dd_config = model_config['dd_config'],
        embed_dim = model_config['embed_dim'],
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters in model is {pytorch_total_params}")

    # set device
    device = torch.device(train_config['device'])
    model = model.to(device)

    # Initialize model weights
    if model_config['initialize']:
        for _name, layer in model.named_modules():
            if isinstance(layer, torch.nn.modules.conv._ConvNd):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

    # set loss
    loss_config = experiment_config['loss_config']
    print('+' * 20)
    print(f"Loss Config: {loss_config}")
    criterion = get_loss(loss_config['params']['disc_start'], loss_config['params']['kl_weight'], loss_config['params']['disc_weight'])

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config['initial_learning_rate']),
    )

    # set scheduler:
    def lambda_(iteration):
        return pow((1 - ((iteration) / int(train_config['max_iterations']))), 0.9)

    # set logger
    logger = get_logger(keys=["train"], title="loss")

    # resume training
    start_iteration = 0
    lowest_loss = 1e0

    if model_config['checkpoint'] == 'None':
        print('+' * 20)
        print(f"Initializing model from scratch ...")
    else:
        print('+' * 20)
        print(f"Resuming model from {model_config['checkpoint']}")
        state = torch.load(model_config['checkpoint'], map_location=device)
        start_iteration = state["iteration"] + 1
        lowest_loss = state["lowest_loss"]
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optim_state_dict"])
        logger.data = state["logger_data"]

    # call `train_iteration`
    for iteration, sample in enumerate(tqdm(train_dataloader)):
        batch_images = sample["image_crop"]
        batch = batch_images.to(device)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_, last_epoch=iteration - 1
        )

        train_loss, prediction = train_iteration(
            batch, model=model, criterion=criterion, optimizer=optimizer, device=device
        )
        scheduler.step()
        print(f"===> train loss: {train_loss:.6f}")
        logger.add(key="train", value=train_loss)
        logger.write()
        logger.plot()

        if iteration % train_config['save_model_every'] == 0:
            is_lowest = train_loss < lowest_loss
            lowest_loss = min(train_loss, lowest_loss)
            state = {
                "iteration": iteration,
                "lowest_loss": lowest_loss,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "logger_data": logger.data,
            }
            save_model(state, iteration, is_lowest)

        if iteration % train_config['save_snapshot_every'] == 0:
            save_snapshot(
                batch,
                prediction,
                iteration,
            )

def train_iteration(batch, model, criterion, optimizer, device):
    model.train()
    prediction = model(batch.to(device))
    loss, = criterion(prediction) # TODO
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), prediction

def save_model(state, iteration, is_lowest):
    pass

def save_snapshot(batch, prediction, iteration):
    pass

if __name__=="__main__":

    with open("configs/autoencoder/autoencoder_kl_16x16x16.yaml", "r") as stream:
        try:
            experiment_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    train(experiment_config)