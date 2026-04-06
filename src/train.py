import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.config import load_config
from src.utils import (
    create_run_dir,
    ensure_dir,
    save_checkpoint,
    save_config_copy,
    save_latest_pointer,
    save_metadata,
)


def build_dummy_dataloader(batch_size: int, num_workers: int) -> DataLoader:
    x = torch.randn(1000, 20)
    y = torch.randint(0, 2, (1000,))
    dataset = TensorDataset(x, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu"
    )

    base_output_dir = cfg["base_output_dir"]
    ensure_dir(base_output_dir)

    run_dir = create_run_dir(base_output_dir, cfg["experiment_name"])
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    log_dir = run_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    save_metadata(run_dir, cfg)
    save_config_copy(run_dir, cfg)
    save_latest_pointer(base_output_dir, run_dir)

    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    loader = build_dummy_dataloader(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    print(f"Training on device: {device}")
    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Run directory: {run_dir}")

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{cfg['epochs']} - loss: {avg_loss:.4f}")
        writer.add_scalar("train/loss", avg_loss, epoch + 1)

        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        save_checkpoint(model, optimizer, epoch + 1, str(checkpoint_path))

    writer.close()
    print("Training complete.")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
