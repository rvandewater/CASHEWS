import gc
import os
import time
from datetime import datetime

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchmetrics as metric
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from preprocessing.preprocess_wearable.embedding.CassandraDataset import (
    get_cassandra_dataloaders,
)
from preprocessing.preprocess_wearable.ppg_paths import CLEANED_PPG_BASE_PATH
import preprocessing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Trains embedding model with TCNAE or LAAE architecture

def train_embedding_model(cfg):
    # Setup model
    model = instantiate(cfg.model)
    model = model.to(device)
    if model.__class__.__name__ == "LAAE":
        print("Using LAAE model and MSELoss")
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
        if cfg.train.lr_schedule:
            scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                schedulers=[
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer=optimizer, T_0=int(cfg.train.epochs * 0.2)
                    ),
                    torch.optim.lr_scheduler.CyclicLR(
                        optimizer=optimizer, base_lr=cfg.train.lr, max_lr=0.01, cycle_momentum=False
                    ),
                ]
            )
        else:
            scheduler = None
    elif model.__class__.__name__ == "TCNAE":
        print("Using TCNAE model and LogCoshLoss")

        class LogCoshLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, y_t, y_prime_t):
                ey_t = y_t - y_prime_t
                return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))

        loss_fn = LogCoshLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, amsgrad=True)
        scheduler = None

    # Load data
    train_dl, test_dl = get_cassandra_dataloaders(
        ppg_path=os.path.join(CLEANED_PPG_BASE_PATH, cfg.ppg_folder),
        window_size=cfg.data.window_size,
        stride=cfg.data.stride,
        batch_size=cfg.train.batch_size,
        test_batch_size=cfg.train.test_batch_size,
        patient_subset=cfg.data.patient_subset,
        train_fraction=cfg.data.train_fraction,
        undersample_n=cfg.data.undersample_n,
    )

    # Save output dim
    example_batch = next(iter(train_dl))
    batch = example_batch.to(device)
    preds = preprocessing.embeddings_clinical_notes.clinical_notes_utils.get_embedding(batch)
    preds = preds.detach().cpu()
    # preds is tensor of shape [batchsize, embedding_size]
    print(f"Embedding length: {preds.shape[1]}")
    wandb.log(
        {"embedding/dim": preds.shape[1], "embedding/factor": preds.shape[1] / cfg.data.window_size}, step=0
    )

    assert (
        batch.shape == model(batch).shape
    ), f"Input and output shape do not match: {batch.shape} != {model(batch).shape}"

    model_save_dir = f'./models/{model.__class__.__name__}/{datetime.now().strftime("%Y-%m-%d")}/{datetime.now().strftime("%H-%M-%S")}'
    print(f"Saving models to {model_save_dir}")
    os.makedirs(model_save_dir, exist_ok=True)

    with open(f"{model_save_dir}/config.yaml", "w") as fp:
        OmegaConf.save(config=cfg, f=fp)

    # Train model
    for epoch in range(cfg.train.epochs):
        print(f"Starting epoch {epoch}")
        train_loss = train_epoch(model, train_dl, optimizer, scheduler, epoch, model_save_dir, loss_fn)
        if torch.isnan(train_loss):
            print("Loss is NaN. Stopping training.")
            break
        test_epoch(model, test_dl, epoch, loss_fn)


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim,
    scheduler: torch.optim,
    epoch: int,
    model_save_dir: str,
    loss_fn=torch.nn.MSELoss(),
):

    model.train()
    # metrics
    loss_mean = metric.MeanMetric()

    total_batches = len(train_loader)
    start_time = time.time()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        y_pred = model(data)
        loss = loss_fn(y_pred, data)

        # test if loss is nan
        if torch.isnan(loss):
            print("exiting due to nan loss")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        loss_mean(loss.cpu())

        del data
        del loss
        gc.collect()
        torch.cuda.empty_cache()

        if batch_idx % 1000 == 0:
            time_per_batch = (time.time() - start_time) / (batch_idx + 1)
            print(f"    Train Batch {batch_idx:>4} done - loss: {loss_mean.compute():.4f}")
            print(
                f"                           time: {(time.time() - start_time)/60:.2f}mins - "
                f"remaining: {(total_batches - batch_idx) * time_per_batch / 60:.2f}mins ({time_per_batch:.2f}s/batch)"
            )
            wandb.log({"train/loss": loss_mean.compute()}, step=epoch)
            if batch_idx % 5000 == 0:
                torch.save(model.state_dict(), f"{model_save_dir}/model_{epoch}.pth")
    torch.save(model.state_dict(), f"{model_save_dir}/model_{epoch}.pth")
    print(f" -- Train -- loss: {loss_mean.compute():.4f}")
    print(f"             total duration: {(time.time() - start_time)/60:.2f}mins")
    wandb.log(
        {"train/loss": loss_mean.compute(), "train/duration (mins)": (time.time() - start_time) / 60},
        step=epoch,
    )
    torch.cuda.empty_cache()

    if scheduler is not None:
        scheduler.step()
        wandb.log({"train/lr": scheduler.get_last_lr()[0]}, step=epoch)
    return loss_mean.compute()


def test_epoch(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    epoch: int,
    criterion=torch.nn.CrossEntropyLoss(),
):
    model.eval()

    # metrics
    loss_mean = metric.MeanMetric()

    img_idx = 0
    for batch_idx, data in enumerate(test_loader):
        data = data.to(device)

        y_pred = model(data)
        loss = criterion(y_pred, data)
        loss_mean(loss.cpu())

        if batch_idx % 100 == 0:
            print(f"    Test Batch {batch_idx:>4} done - loss: {loss_mean.compute():.4f}")
            wandb.log(
                {
                    "test/loss": loss_mean.compute(),
                    f"test/samples_{img_idx}": wandb.Image(
                        get_signal_sample_image(data.detach().cpu(), y_pred.detach().cpu(), n_samples=2)
                    ),
                },
                step=epoch,
            )
            plt.close()
            img_idx += 1

    print(f"\t -- Test -- loss: {loss_mean.compute():.4f}")
    wandb.log(
        {
            "test/loss": loss_mean.compute(),
            f"test/samples_{img_idx}": wandb.Image(
                get_signal_sample_image(data.detach().cpu(), y_pred.detach().cpu())
            ),
        },
        step=epoch,
    )
    plt.close()

    return loss_mean.compute()


def get_signal_sample_image(data, rec, n_samples=5):
    fig, ax = plt.subplots(figsize=(18, 6))
    data_len = data.shape[1]
    fig_data = data[:n_samples].numpy().flatten()
    rec_data = rec[:n_samples].numpy().flatten()
    plt.plot(fig_data, label="raw_data", color="black")
    plt.plot(rec_data, label="rec_data", color="red")
    plt.vlines(
        [i * data_len for i in range(1, n_samples)],
        ymin=min(fig_data.min(), rec_data.min()),
        ymax=max(fig_data.max(), rec_data.max()),
        color="gray",
        linestyles="dashed",
    )
    plt.legend()

    return fig


@hydra.main(config_path="config", config_name="embedding_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_container(cfg, resolve=True))
    # init wandb
    if not cfg.wandb.disabled:
        wandb.init(
            project="ppg_embedding",
            entity="cassandra_hpi",
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            allow_val_change=True,
            settings=wandb.Settings(start_method="fork"),
        )
    else:
        wandb.init(mode="disabled")

    if cfg.data.stride > cfg.data.window_size:
        raise ValueError("Stride must be smaller than window_size.")

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # fix config data types
    if cfg.model.get("dilations") is not None:
        if isinstance(cfg.model.dilations, int):
            # make tuple object
            cfg.model.dilations = tuple(2**i for i in range(cfg.model.dilations.bit_length()))
            print(f"set dilations to {cfg.model.dilations}")

    train_embedding_model(cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
