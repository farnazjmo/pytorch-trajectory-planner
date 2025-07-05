"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
# With help of AI

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.datasets.road_dataset import load_data
from .models import load_model, save_model
from .datasets.road_dataset import RoadDataset
from .datasets.road_transforms import EgoTrackProcessor
from homework.metrics import PlannerMetric



def masked_mse_loss(pred, target, mask):
    """
    Args:
        pred (B, n_waypoints, 2)
        target (B, n_waypoints, 2)
        mask (B, n_waypoints)
    Returns:
        scalar tensor
    """
    loss = ((pred - target) ** 2).sum(dim=-1)  # (B, n_waypoints)
    loss = loss * mask  # mask invalid waypoints
    return loss.sum() / mask.sum()


def masked_smooth_l1_loss(pred, target, mask, beta=1.0):
    """
    pred, target: (B, n, 2)
    mask: (B, n)
    """
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction='none', beta=beta)
    loss = loss.sum(dim=-1)  # (B, n)
    loss = loss * mask  # (B, n)
    return loss.sum() / mask.sum()



def get_model_inputs(model, batch):
    model_type = model.__class__.__name__.lower()
    
    if "cnn" in model_type:
        assert "image" in batch, f"'image' missing in batch. Got: {list(batch.keys())}"
        return model(image=batch["image"])
    elif "mlp" in model_type or "transformer" in model_type:
        assert "track_left" in batch and "track_right" in batch, \
            f"Missing track inputs. Got: {list(batch.keys())}"
        return model(track_left=batch["track_left"], track_right=batch["track_right"])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train(
    exp_dir="logs",
    model_name="mlp_planner",
    transform_pipeline="state_only",  
    num_workers=4,
    lr=1e-3,
    batch_size=64,
    num_epoch=40,
    seed=2024,
    **kwargs,
):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Logging
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Model
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Data
    # Create the train and val dataloaders using the built-in transform loader
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    global_step = 0
    for epoch in range(num_epoch):
        model.train()
        train_loss_total, val_loss_total = 0.0, 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            

            optimizer.zero_grad()
            preds = get_model_inputs(model, batch)

            assert batch["waypoints"] is not None, "Waypoints are None!"
            assert batch["waypoints_mask"] is not None, "Waypoints mask is None!"
            assert preds is not None, "Predictions are None!"

            loss = masked_smooth_l1_loss(preds, batch["waypoints"], batch["waypoints_mask"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()
            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        model.eval()

        val_lateral_total, val_longitudinal_total = 0.0, 0.0
        metric = PlannerMetric()
        with torch.no_grad():
            model.eval()
            for batch in val_loader:
                
                batch = {k: v.to(device) for k, v in batch.items()}  # move entire batch to device
                preds = get_model_inputs(model, batch)               # planner-agnostic

                assert batch["waypoints"] is not None, "Waypoints are None!"
                assert batch["waypoints_mask"] is not None, "Waypoints mask is None!"
                assert preds is not None, "Predictions are None!"

                loss = masked_smooth_l1_loss(preds, batch["waypoints"], batch["waypoints_mask"])
                val_loss_total += loss.item()

                # Compute lateral and longitudinal errors
                metric.add(preds, batch["waypoints"], batch["waypoints_mask"])
                

        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        results = metric.compute()
        avg_lat_err = results["lateral_error"]
        avg_long_err = results["longitudinal_error"]
        

        logger.add_scalar("val/loss", avg_val_loss, epoch)
        logger.add_scalar("val/lateral_error", avg_lat_err, epoch)
        logger.add_scalar("val/longitudinal_error", avg_long_err, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
              f"Epoch {epoch + 1:2d} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
              f"lateral_error={avg_lat_err:.4f} | longitudinal_error={avg_long_err:.4f}"
            )


    # Save model to root for grading
    save_path = save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to: {save_path}")
    print(f"Copy also saved to: {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))

