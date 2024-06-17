"""Training pipeline."""

import dataclasses
import importlib
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .augmentation import PitchReverbAugment
from .callbacks import SpikeDetection
from .config import CONFIG
from .criterion import CPCCriterion
from .data import AudioSequenceDataset, SameSpeakerBatchSampler
from .model import CPC
from .utils import Records, params_norm, previous_checkpoint

__all__ = ["train"]


@torch.inference_mode()
def evaluation(model: CPC, criterion: CPCCriterion, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss, total_accuracy = 0, 0
    for past, future, _ in loader:
        predictions, latent = model(past, future)
        loss, accuracy = criterion(predictions, latent)
        total_loss += loss.mean().float().item()
        total_accuracy += accuracy.mean().item()
    model.train()
    return total_loss / len(loader), total_accuracy / len(loader)


def train(run_name: str, workdir: str, train_manifest: str, val_manifest: str, project_name: str = "cpc") -> None:
    run_dir = Path(workdir).resolve() / project_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(project_dir=run_dir, log_with="wandb")
    accelerator.init_trackers(
        project_name,
        config=dataclasses.asdict(CONFIG) | {"train": Path(train_manifest).stem, "val": Path(val_manifest).stem},
        init_kwargs={"wandb": {"mode": CONFIG.wandb_mode, "name": run_name, "dir": Path(workdir).resolve()}},
    )
    if accelerator.is_main_process:
        wandb_run = accelerator.get_tracker("wandb", unwrap=True)
        wandb_run.log_code(Path(importlib.util.find_spec("fastcpc").origin).parent)
        accelerator.print(f"Run dir: {run_dir}.\nWandb dir: {Path(wandb_run.dir).parent}")

    seed = CONFIG.random_seed + accelerator.process_index
    model = CPC()
    criterion = CPCCriterion(torch.Generator(device=accelerator.device).manual_seed(seed))
    optimizer = Adam(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=CONFIG.scheduler_iters)
    transform = PitchReverbAugment(random_seed=seed)
    train_dataset = AudioSequenceDataset(train_manifest, transform=transform)
    train_sampler = SameSpeakerBatchSampler(train_dataset.sequences, seed)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=CONFIG.num_workers)
    val_dataset = AudioSequenceDataset(val_manifest)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=CONFIG.batch_size, num_workers=CONFIG.num_workers)

    accelerator.print(f"Train duration after splitting: {train_dataset.duration_in_seconds / 3600:.2f}h")
    accelerator.print(f"Train duration after batching: {train_sampler.duration_in_seconds / 3600:.2f}h")

    model, criterion, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, criterion, optimizer, train_loader, val_loader, scheduler
    )

    if (ckpt := previous_checkpoint(run_dir)) is not None:
        accelerator.print(f"Resuming from {ckpt}")
        accelerator.load_state(ckpt)
        initial_epoch = int(ckpt.stem.removeprefix("epoch_"))
        step = len(train_loader) * initial_epoch
        train_loader.iteration = initial_epoch
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(initial_epoch)
        else:
            train_loader.batch_sampler.batch_sampler.set_epoch(initial_epoch)
    else:
        step, initial_epoch = 0, 0
    records = Records(
        ["train/loss_mean", "train/acc_mean"]
        + ["train/batch_time", "train/data_time", "train/grad_norm", "train/params_norm"]
        + [f"train/loss_{i}" for i in range(CONFIG.num_predicts)]
        + [f"train/acc_{i}" for i in range(CONFIG.num_predicts)],
    )
    spike_detection = SpikeDetection()
    pbar = tqdm(total=CONFIG.num_epochs * len(train_loader), initial=step, disable=not accelerator.is_main_process)
    for epoch in range(initial_epoch, CONFIG.num_epochs):
        pbar.set_description("Training")
        model.train()
        tick = time.perf_counter()
        for past, future, metadata in train_loader:
            records["train/data_time"].update(time.perf_counter() - tick)
            predictions, latent = model(past, future)
            losses, accuracies = criterion(predictions, latent)
            if spike_detection.update(losses.sum()):
                accelerator.print("Loss spike detected!")
                accelerator.save_state(run_dir / "spike", safe_serialization=False)
                accelerator.save(
                    {"metadata": metadata, "past": past, "future": future, "losses": losses, "accuracies": accuracies},
                    run_dir / "spike/data_and_losses.pkl",
                )
                accelerator.set_trigger()
            if accelerator.check_trigger():
                accelerator.end_training()
                return
            accelerator.backward(losses.sum())
            if accelerator.sync_gradients:
                total_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=CONFIG.max_grad_norm)
                records["train/grad_norm"].update(total_norm)
            records["train/params_norm"].update(params_norm(model.parameters()))
            lr = scheduler.get_lr()[0]
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            records["train/loss_mean"].update(losses.detach().mean().item())
            records["train/acc_mean"].update(accuracies.mean().item())
            for i, (loss, acc) in enumerate(zip(losses.detach(), accuracies, strict=True)):
                records[f"train/loss_{i}"].update(loss.item())
                records[f"train/acc_{i}"].update(acc.item())
            records["train/batch_time"].update(time.perf_counter() - tick)
            step += 1
            pbar.set_postfix(epoch=epoch, loss=records["train/loss_mean"].avg, acc=records["train/acc_mean"].avg)
            pbar.update()
            if step % CONFIG.log_interval == 0:
                accelerator.log(records.log() | {"epoch": epoch, "train/lr": lr}, step)
            tick = time.perf_counter()
        scheduler.step()
        # Disable safe serialization because of LSTM, otherwise some weights are not saved
        accelerator.save_state(run_dir / f"epoch_{epoch+1}", safe_serialization=False)
        pbar.set_description("Validation ongoing...")
        val_loss, val_acc = evaluation(model, criterion, val_loader)
        accelerator.log({"val/loss_mean": val_loss, "val/acc_mean": val_acc, "epoch": epoch}, step)
    accelerator.end_training()
