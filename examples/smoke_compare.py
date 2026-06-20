"""Run a tiny CASSM/GPFA smoke comparison on Lorenz data."""

from __future__ import annotations

from torch.utils.data import DataLoader

from ladys.config import load_experiment_config
from ladys.datasets import LorenzDataset, LorenzDatasetConfig
from ladys.models import CASSMConfig, GPFAConfig
from ladys.training import Trainer, TrainerConfig
from ladys.training.strategies import build_strategy


def run_one(config, train_loader, valid_loader, n_neurons: int, n_time: int):
    model = config.build(n_neurons=n_neurons, n_time=n_time)
    strategy = build_strategy(config.optimization)
    trainer = Trainer(TrainerConfig(epochs=1, device="cpu"))
    history = trainer.fit(model, strategy, train_loader, valid_loader)
    report = history[-1]
    valid_loss = None if report.valid is None else report.valid.loss
    print(
        f"{config.name}: train={report.train.loss:.4f} "
        f"valid={valid_loss:.4f} seconds={report.seconds:.2f}"
    )


def main():
    # Exercise the YAML-backed config path.
    load_experiment_config("configs/experiment/cassm_lorenz.yaml")

    dataset_config = LorenzDatasetConfig(
        neurons=8,
        num_inits=2,
        num_trials=4,
        num_steps=24,
        burn_steps=50,
        seed=1,
    )
    train_ds, valid_ds = LorenzDataset.make_splits(dataset_config)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=2, shuffle=False)
    n_time, n_neurons = train_ds.spikes.shape[1:]

    run_one(
        CASSMConfig(projection_dim=4),
        train_loader,
        valid_loader,
        n_neurons,
        n_time,
    )
    run_one(
        GPFAConfig(latent_dim=3),
        train_loader,
        valid_loader,
        n_neurons,
        n_time,
    )


if __name__ == "__main__":
    main()
