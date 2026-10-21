import csv
import gc
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import CamVidDataset, NUM_CLASSES
from model import get_model
from utils import CombinedLoss, compute_mean_iou, set_seed

DATA_ROOT   = "/home/1d5f/.local/CV_and_AI/CamVid"
RESULTS_DIR = "results"
OPTUNA_DIR  = os.path.join(RESULTS_DIR, "optuna_results")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OPTUNA_DIR,  exist_ok=True)

IMG_SIZE    = 256
NUM_WORKERS = 2
SEED        = 42
N_TRIALS    = 20
EPOCHS      = 20
GRAD_CLIP   = 1.0

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = torch.cuda.is_available()
ENCODERS    = ["vgg16", "resnet34"]

CSV_FIELDS = [
    "trial_number", "lr", "batch_size", "optimizer",
    "weight_decay", "ce_weight", "dice_weight",
    "best_val_miou", "pruned", "pruned_at_epoch",
]

#hyper parameter range to decide the best value for each encoder (vgg16bn and resnet34)
def sample_hyperparameters(trial):
    ce_weight = trial.suggest_float("ce_weight", 0.3, 0.7)
    return {
        "lr":           trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [2, 4]),
        "optimizer":    trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        "ce_weight":    ce_weight,
        "dice_weight":  1.0 - ce_weight,
    }

def write_trial_row(csv_path, trial_num, hp, best_miou, pruned, pruned_at):
    row = {
        "trial_number":   trial_num,
        "lr":             hp["lr"],
        "batch_size":     hp["batch_size"],
        "optimizer":      hp["optimizer"],
        "weight_decay":   hp["weight_decay"],
        "ce_weight":      hp["ce_weight"],
        "dice_weight":    hp["dice_weight"],
        "best_val_miou":  best_miou,
        "pruned":         pruned,
        "pruned_at_epoch": pruned_at if pruned_at is not None else "",
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def build_objective(encoder_name, csv_path):

    def objective(trial):
        set_seed(SEED)
        hp = sample_hyperparameters(trial)

        torch.cuda.empty_cache()
        gc.collect()

        train_ds = CamVidDataset(DATA_ROOT, split="train", img_size=IMG_SIZE, augment=True)
        val_ds   = CamVidDataset(DATA_ROOT, split="val",   img_size=IMG_SIZE, augment=False)

        train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)

        model     = get_model(encoder_name=encoder_name, num_classes=NUM_CLASSES).to(DEVICE)
        criterion = CombinedLoss(ce_weight=hp["ce_weight"], dice_weight=hp["dice_weight"],
                                 num_classes=NUM_CLASSES)

        if hp["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=hp["lr"], weight_decay=hp["weight_decay"])
        else:
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=hp["lr"], weight_decay=hp["weight_decay"])

        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

        best_miou = 0.0
        pruned    = False
        pruned_at = None

        try:
            for epoch in range(EPOCHS):
                model.train()
                for images, labels in train_loader:
                    images = images.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                        loss = criterion(model(images), labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()

                model.eval()
                preds_all, gts_all = [], []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(DEVICE, non_blocking=True)
                        preds_all.append(model(images).argmax(1).cpu())
                        gts_all.append(labels)

                miou      = compute_mean_iou(torch.cat(preds_all), torch.cat(gts_all), NUM_CLASSES)
                best_miou = max(best_miou, miou)

                trial.report(miou, epoch)
                if trial.should_prune():
                    pruned    = True
                    pruned_at = epoch + 1
                    raise optuna.TrialPruned()

        except optuna.TrialPruned:
            pass

        finally:
            write_trial_row(csv_path, trial.number, hp, best_miou, pruned, pruned_at)
            del model, optimizer, criterion
            torch.cuda.empty_cache()
            gc.collect()

        if pruned:
            raise optuna.TrialPruned()

        return best_miou

    return objective

def plot_trial_results(csv_path, encoder_name):
    trials = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            trials.append(row)
    if not trials:
        return

    nums   = [int(r["trial_number"]) for r in trials]
    mious  = [float(r["best_val_miou"]) for r in trials]
    colors = ["red" if r["pruned"] == "True" else "steelblue" for r in trials]

    plt.figure(figsize=(8, 4))
    plt.scatter(nums, mious, c=colors, s=60, zorder=3)
    plt.plot(nums, mious, color="gray", linewidth=0.8, zorder=2)

    best_idx = int(np.argmax(mious))
    plt.scatter(nums[best_idx], mious[best_idx], c="gold", s=120, marker="*", zorder=4)

    plt.legend(handles=[
        Patch(facecolor="steelblue", label="completed"),
        Patch(facecolor="red",       label="pruned"),
        Patch(facecolor="gold",      label="best"),
    ], fontsize=9)
    plt.xlabel("trial number")
    plt.ylabel("best val mIoU")
    plt.title(f"optuna trials — UNet-{encoder_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OPTUNA_DIR, f"trial_results_{encoder_name}.png"), dpi=150)
    plt.close()

def plot_parallel_coordinates(csv_path, encoder_name):
    trials = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            trials.append(row)
    if not trials:
        return

    axes_labels = ["lr", "weight_decay", "ce_weight", "batch_size", "val mIoU"]
    data = np.array([
        [float(r["lr"]), float(r["weight_decay"]), float(r["ce_weight"]),
         float(r["batch_size"]), float(r["best_val_miou"])]
        for r in trials
    ])

    col_range = data.max(axis=0) - data.min(axis=0)
    col_range[col_range == 0] = 1.0
    data_norm   = (data - data.min(axis=0)) / col_range
    mious_norm  = data_norm[:, -1]

    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(axes_labels))
    for i, row in enumerate(data_norm):
        ax.plot(x_pos, row, color=cmap(mious_norm[i]), alpha=0.6, linewidth=1.2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(axes_labels)
    ax.set_ylabel("normalised value")
    ax.set_title(f"parallel coordinates — UNet-{encoder_name}")

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=data[:, -1].min(), vmax=data[:, -1].max()),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="val mIoU")
    plt.tight_layout()
    plt.savefig(os.path.join(OPTUNA_DIR, f"parallel_coords_{encoder_name}.png"), dpi=150)
    plt.close()

def export_full_study_csv(study, encoder_name):
    rows = []
    for t in study.trials:
        row = {"trial_number": t.number, "state": str(t.state)}
        row.update(t.params)
        row["value"] = t.value if t.value is not None else ""
        rows.append(row)
    if not rows:
        return

    all_keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    out_path = os.path.join(OPTUNA_DIR, f"optuna_logs_{encoder_name}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # also append to a combined log at project root
    root_log   = "optuna_logs.csv"
    all_keys_e = ["encoder"] + all_keys
    for row in rows:
        row["encoder"] = encoder_name
    file_exists = os.path.isfile(root_log)
    with open(root_log, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys_e, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"study log -> {out_path}")
    print(f"combined  -> {root_log}")

def main():
    set_seed(SEED)

    for encoder in ENCODERS:
        print(f"\n--- hyperparameter search: UNet-{encoder} ---")

        csv_path = os.path.join(OPTUNA_DIR, f"trials_{encoder}.csv")
        if os.path.isfile(csv_path):
            os.remove(csv_path)

        study = optuna.create_study(
            study_name=f"UNet_{encoder}",
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        )
        study.optimize(build_objective(encoder, csv_path), n_trials=N_TRIALS)

        print(f"\nbest val mIoU : {study.best_value:.4f}")
        best_params = dict(study.best_trial.params)
        best_params["dice_weight"] = 1.0 - best_params["ce_weight"]
        for k, v in best_params.items():
            print(f"  {k:15s}: {v}")

        for save_dir in [RESULTS_DIR, OPTUNA_DIR]:
            with open(os.path.join(save_dir, f"best_params_{encoder}.json"), "w") as f:
                json.dump(best_params, f, indent=4)

        export_full_study_csv(study, encoder)
        plot_trial_results(csv_path, encoder)
        plot_parallel_coordinates(csv_path, encoder)

        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()




# python optimization.py