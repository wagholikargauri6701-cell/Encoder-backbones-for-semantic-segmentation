import os
import json
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
#third party import
from dataset import CamVidDataset, NUM_CLASSES
from model import get_model
from utils import CombinedLoss, compute_mean_iou, compute_pixel_accuracy, set_seed

DATA_ROOT   = "/home/1d5f/.local/CV_and_AI/CamVid"
RESULTS_DIR = "results"
CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

IMG_SIZE = 256
NUM_WORKERS = 2
SEED = 42
EPOCHS = 50
GRAD_CLIP = 1.0
PATIENCE = 10

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = torch.cuda.is_available()

# Best hyperparameters from Optuna (results/best_params_<encoder>.json)
BEST_PARAMS = {
    "vgg16":{
    "lr": 0.00011017042684257373,
    "batch_size": 2,
    "optimizer": "AdamW",
    "weight_decay": 5.431174346560176e-05,
    "ce_weight": 0.623403687933304,
    "dice_weight": 0.37659631206669597,
    },
    "resnet34": {
        "lr":           0.0001838075248544353,
        "batch_size":   2,
        "optimizer":    "Adam",
        "weight_decay": 3.191104348757801e-06,
        "ce_weight":    0.5569991990182276,
        "dice_weight":  0.4430008009817724,
    },
}


#--optimizer--
def get_optimizer(name, params, lr, weight_decay):
    
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

def train_one_encoder(encoder_name):
    
    print(f"  Training UNet — encoder: {encoder_name.upper()}")
   
    set_seed(SEED)

    if encoder_name not in BEST_PARAMS:
        raise ValueError(f"No hyperparameters for '{encoder_name}'. Add to BEST_PARAMS.")

    hp = BEST_PARAMS[encoder_name]

    train_dataset = CamVidDataset(DATA_ROOT, split="train", img_size=IMG_SIZE, augment=True)
    val_dataset   = CamVidDataset(DATA_ROOT, split="val",   img_size=IMG_SIZE, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=hp["batch_size"], shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model     = get_model(encoder_name=encoder_name, num_classes=NUM_CLASSES).to(DEVICE)
    #--loss function--
    criterion = CombinedLoss(ce_weight=hp["ce_weight"], dice_weight=hp["dice_weight"],
                              num_classes=NUM_CLASSES)
    optimizer = get_optimizer(hp["optimizer"], model.parameters(), hp["lr"], hp["weight_decay"])

    # Cosine annealing decays lr from initial value to ~0 by epoch 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    history = {
        "train_loss": [], "val_loss": [], "val_miou": [],
        "val_acc": [], "lr": [], "epoch_time_s": [],
    }
    grad_norms       = []
    best_miou        = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):

        # --- train ---
        model.train()
        running_loss = 0.0
        epoch_start  = time.time()

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                loss = criterion(model(images), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            grad_norms.append(float(grad_norm))
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # --- validate ---
        model.eval()
        val_loss   = 0.0
        preds_all  = []
        labels_all = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds_all.append(outputs.argmax(1).cpu())
                labels_all.append(labels.cpu())

        val_loss  /= len(val_loader)
        preds_all  = torch.cat(preds_all)
        labels_all = torch.cat(labels_all)

        val_miou   = compute_mean_iou(preds_all, labels_all, NUM_CLASSES)
        val_acc    = compute_pixel_accuracy(preds_all, labels_all)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_time_s"].append(epoch_time)

        print(
            f"  epoch {epoch+1:3d}/{EPOCHS} | "
            f"train {train_loss:.4f} | val {val_loss:.4f} | "
            f"mIoU {val_miou:.4f} | acc {val_acc:.4f} | "
            f"lr {current_lr:.2e} | {epoch_time:.1f}s"
        )

        #-- early stoping --
        if val_miou > best_miou:
            best_miou        = val_miou
            patience_counter = 0
            ckpt_path = os.path.join(CKPT_DIR, f"unet_{encoder_name}_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  new best mIoU {best_miou:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            print(f" ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            print(f"\n  early stopping after {PATIENCE} epochs without improvement.")
            break

        scheduler.step()

    last_ckpt = os.path.join(CKPT_DIR, f"unet_{encoder_name}_last.pth")
    torch.save(model.state_dict(), last_ckpt)
    print(f"\n  last checkpoint -> {last_ckpt}")

    with open(os.path.join(RESULTS_DIR, f"history_{encoder_name}.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(RESULTS_DIR, f"grad_norms_{encoder_name}.json"), "w") as f:
        json.dump(grad_norms, f)

    print(f"  best val mIoU : {best_miou:.4f}")
    print(f"  avg epoch time: {sum(history['epoch_time_s'])/len(history['epoch_time_s']):.1f}s")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default="both",
                        choices=["vgg16", "resnet34", "both"])
    args = parser.parse_args()

    encoders = ["vgg16", "resnet34"] if args.encoder == "both" else [args.encoder]
    for enc in encoders:
        train_one_encoder(enc)

if __name__ == "__main__":
    main()

# python train.py --encoder both
# python train.py --encoder vgg16
# python train.py --encoder resnet34
# image size = 256
# vgg16_bn and resnet34 - train batches = 183



